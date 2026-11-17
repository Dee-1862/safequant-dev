# Byte-identity test: auto-gptq's original QuantLinear.pack() vs scripts/gptq_fast_pack.py.
#
#   $AGENV/bin/python scripts/tests/test_gptq_fast_pack.py          (CPU only, ~1-2 min)
#
# For every importable QuantLinear class with the per-column loop, packs the same layer
# with both versions into fresh modules and compares every buffer bit for bit
# (qweight, qzeros, scales, g_idx, bias). Inputs mimic a real GPTQ layer: integer weights
# rebuilt from per-group scales/zeros (what GPTQ hands to pack), act-order g_idx from a
# random permutation (desc_act=True), plus a noisy variant that lands on .5 rounding
# boundaries. Shapes are phi3's (qkv, o, gate_up, down) for the class the builds use.
# Exits 1 on any mismatch.
import importlib
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gptq_fast_pack as fp  # noqa: E402

torch.set_num_threads(4)  # login node: stay light
GS = 128
PHI3 = [(9216, 3072), (3072, 3072), (16384, 3072), (3072, 8192)]  # (out, in)
SMALL = [(512, 1024)]


def make_layer(out_f, in_f, bits, seed, noisy, bias):
    """Linear + scales/zeros/g_idx shaped like auto-gptq's pack_model arguments."""
    g = torch.Generator().manual_seed(seed)
    maxq = 2 ** bits - 1
    n_groups = in_f // GS
    perm = torch.randperm(in_f, generator=g)
    g_idx = torch.empty(in_f, dtype=torch.int32)
    g_idx[perm] = (torch.arange(in_f) // GS).to(torch.int32)          # act-order groups
    scales = torch.rand(out_f, n_groups, generator=g) * 0.02 + 1e-3   # fp32, (out, G)
    zeros = torch.randint(0, maxq + 1, (out_f, n_groups), generator=g).float()
    q = torch.randint(0, maxq + 1, (out_f, in_f), generator=g).float()
    W = (q - zeros[:, g_idx.long()]) * scales[:, g_idx.long()]
    if noisy:  # push values toward .5 boundaries after the divide in pack()
        W = W + (torch.rand(out_f, in_f, generator=g) - 0.5) * scales[:, g_idx.long()]
    lin = nn.Linear(in_f, out_f, bias=bias, dtype=torch.float16)
    lin.weight.data = W.half()
    if bias:
        lin.bias.data = (torch.randn(out_f, generator=g) * 0.1).half()
    return lin, scales, zeros, g_idx


def new_qlayer(cls, bits, in_f, out_f, bias):
    try:
        return cls(bits, GS, in_f, out_f, bias)
    except TypeError:
        return cls(bits, GS, in_f, out_f, bias=bias)


def bits_equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.is_floating_point():
        a, b = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    return torch.equal(a, b)


def main():
    fails, n = [], 0
    for modname, variant in fp.TARGETS:
        try:
            cls = importlib.import_module(modname).QuantLinear
        except Exception as e:
            print(f"skip {modname}: not importable here ({type(e).__name__})")
            continue
        orig = cls.pack
        fast = fp.vectorized_pack(cls, variant)
        if fast is None:
            print(f"FAIL {modname}: loop text not recognized; would not be patched")
            fails.append(modname)
            continue
        uses = modname.endswith("qlinear_cuda")
        shapes = PHI3 if uses else SMALL
        bit_list = [4, 3] if uses else [4]
        for bits in bit_list:
            for (out_f, in_f) in shapes:
                for noisy, bias in ((False, False), (True, True)):
                    try:
                        lin, s, z, gi = make_layer(out_f, in_f, bits, n, noisy, bias)
                        qa, qb = new_qlayer(cls, bits, in_f, out_f, bias), new_qlayer(cls, bits, in_f, out_f, bias)
                    except NotImplementedError as e:
                        print(f"skip {modname} bits={bits} {out_f}x{in_f}: {e}")
                        continue
                    t0 = time.time(); orig(qa, lin, s.clone(), z.clone(), gi.clone()); t1 = time.time()
                    fast(qb, lin, s.clone(), z.clone(), gi.clone()); t2 = time.time()
                    sa, sb = qa.state_dict(), qb.state_dict()
                    diff = [k for k in sa if k not in sb or not bits_equal(sa[k], sb[k])]
                    tag = f"{modname.split('.')[-1]:16s} bits={bits} {out_f:5d}x{in_f:<5d} noisy={int(noisy)} bias={int(bias)}"
                    if diff or set(sa) != set(sb):
                        print(f"FAIL {tag}  differing buffers: {diff}")
                        fails.append(tag)
                    else:
                        print(f"PASS {tag}  buffers={sorted(sa)}  orig {t1 - t0:6.2f}s  fast {t2 - t1:5.2f}s")
                    n += 1
    print(f"\n{n} cases, {len(fails)} failed")
    patched = fp.apply()
    print(f"apply() patches: {patched}")
    sys.exit(1 if fails or not n else 0)


if __name__ == "__main__":
    main()
