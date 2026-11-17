# Compare two GPTQ checkpoint folders tensor by tensor, bit for bit.
#
#   python scripts/tests/compare_gptq_checkpoints.py <ref_dir> <new_dir>
#
# Reads every *.safetensors file in both folders (CPU only, no model load) and reports,
# per tensor: identical / different, plus for packed int32 tensors (qweight, qzeros) the
# fraction of differing 4- or 3-bit fields. Also diffs quantize_config.json.
# Exit 0 only if every tensor is bit-identical and both folders have the same keys.
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_dir(d):
    out = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        if f.is_symlink():
            continue  # fix_autogptq_checkpoints.py links model.safetensors -> gptq_model-*.safetensors
        out.update(load_file(str(f)))
    return out


def nibble_diff(a, b, bits):
    """Fraction of differing `bits`-wide fields in two int32 tensors (4-bit exact, 3-bit approx)."""
    x = (a ^ b).view(torch.int32).flatten().to(torch.int64) & 0xFFFFFFFF
    per = 32 // bits
    mask = (1 << bits) - 1
    diff = sum(((x >> (bits * k)) & mask).ne(0).sum().item() for k in range(per))
    return diff / (x.numel() * per)


def main():
    ref, new = sys.argv[1], sys.argv[2]
    A, B = load_dir(ref), load_dir(new)
    bits = json.loads((Path(ref) / "quantize_config.json").read_text()).get("bits", 4)
    qa = json.loads((Path(ref) / "quantize_config.json").read_text())
    qb = json.loads((Path(new) / "quantize_config.json").read_text())
    cfg_diff = {k: (qa.get(k), qb.get(k)) for k in set(qa) | set(qb) if qa.get(k) != qb.get(k)}
    print(f"ref: {ref}  ({len(A)} tensors)\nnew: {new}  ({len(B)} tensors)")
    print(f"quantize_config differences: {cfg_diff or 'none'}")
    only = sorted(set(A) ^ set(B))
    if only:
        print(f"keys in only one folder: {only[:10]}{' ...' if len(only) > 10 else ''}")
    same, differ = 0, []
    for k in sorted(set(A) & set(B)):
        a, b = A[k], B[k]
        if a.dtype != b.dtype or a.shape != b.shape:
            differ.append((k, "dtype/shape differ")); continue
        ab = a.contiguous().view(torch.uint8) if a.is_floating_point() else a
        bb = b.contiguous().view(torch.uint8) if b.is_floating_point() else b
        if torch.equal(ab, bb):
            same += 1
        elif a.dtype == torch.int32 and (k.endswith("qweight") or k.endswith("qzeros")):
            differ.append((k, f"{100 * nibble_diff(a, b, bits):.4f}% of {bits}-bit fields differ"))
        else:
            d = (a.float() - b.float()).abs()
            differ.append((k, f"max |diff| {d.max().item():.3g}, {100 * d.ne(0).float().mean().item():.4f}% elements"))
    print(f"\nbit-identical tensors: {same}/{len(set(A) & set(B))}")
    for k, msg in differ[:25]:
        print(f"  DIFF {k}: {msg}")
    if len(differ) > 25:
        print(f"  ... {len(differ) - 25} more")
    ok = not differ and not only and not cfg_diff
    print("RESULT: IDENTICAL" if ok else "RESULT: DIFFERENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
