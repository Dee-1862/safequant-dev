# Vectorized replacement for auto-gptq 0.7.1's per-column packing loop.
#
# auto-gptq's QuantLinear.pack() recomputes the integer weights with a Python loop over
# every input column, on the CPU:
#
#     intweight = []
#     for idx in range(self.infeatures):
#         intweight.append(torch.round((W[:, idx] + scale_zeros[self.g_idx[idx]])
#                                      / self.scales[self.g_idx[idx]]).to(torch.int)[:, None])
#     intweight = torch.cat(intweight, dim=1)
#
# That loop dominated the phi3 build on DeltaAI (1 CPU, ~17k iterations per layer).
# apply() swaps ONLY that block for the same expression on the whole matrix:
#
#     intweight = torch.round((W + scale_zeros[self.g_idx].t())
#                             / self.scales[self.g_idx].t()).to(torch.int)
#
# Every element goes through the same ops (add, divide, round, cast) on the same inputs
# with the same dtype promotion; nothing is summed or reordered, so the packed result is
# bit-identical. scripts/tests/test_gptq_fast_pack.py checks that byte for byte against
# the original pack(); the rest of pack() (the numpy bit packing) is auto-gptq's own code.
#
# The block is replaced only when its text matches one of the two known 0.7.1 variants
# exactly (whitespace-normalized); otherwise the class is left untouched.
import importlib
import inspect
import re
import textwrap

# (module, loop variant) for every 0.7.1 QuantLinear whose pack() has the loop
TARGETS = [
    ("auto_gptq.nn_modules.qlinear.qlinear_cuda", "g_idx"),
    ("auto_gptq.nn_modules.qlinear.qlinear_exllama", "g_idx"),
    ("auto_gptq.nn_modules.qlinear.qlinear_triton", "g_idx"),
    ("auto_gptq.nn_modules.qlinear.qlinear_cuda_old", "group_size"),
]

_LOOPS = {
    "g_idx": (
        "intweight = [] for idx in range(self.infeatures): intweight.append( "
        "torch.round((W[:, idx] + scale_zeros[self.g_idx[idx]]) / "
        "self.scales[self.g_idx[idx]]).to(torch.int)[ :, None ] ) "
        "intweight = torch.cat(intweight, dim=1)"
    ),
    "group_size": (
        "intweight = [] for idx in range(self.infeatures): g_idx = idx // self.group_size "
        "intweight.append(torch.round((W[:, idx] + scale_zeros[g_idx]) / "
        "self.scales[g_idx]).to(torch.int)[:, None]) "
        "intweight = torch.cat(intweight, dim=1)"
    ),
}

_VECTORIZED = {
    "g_idx": (
        "intweight = torch.round((W + scale_zeros[self.g_idx].t()) / "
        "self.scales[self.g_idx].t()).to(torch.int)"
    ),
    "group_size": (
        "_sq_g = torch.arange(self.infeatures, device=W.device) // self.group_size\n"
        "{ind}intweight = torch.round((W + scale_zeros[_sq_g].t()) / "
        "self.scales[_sq_g].t()).to(torch.int)"
    ),
}

_BLOCK = re.compile(r"(?P<ind>[ \t]*)intweight = \[\]\n.*?intweight = torch\.cat\(intweight, dim=1\)\n",
                    re.S)


def _norm(s):
    return " ".join(s.split())


def vectorized_pack(cls, variant):
    """
    Build a pack() for `cls` with the column loop replaced; None if the source differs.

    Inputs:
        - cls (type): an auto-gptq QuantLinear class
        - variant (str): "g_idx" | "group_size"

    Outputs:
        - pack (function | None)
    """
    src = textwrap.dedent(inspect.getsource(cls.pack))
    blocks = list(_BLOCK.finditer(src))
    if len(blocks) != 1 or _norm(blocks[0].group(0)) != _norm(_LOOPS[variant]):
        return None
    m = blocks[0]
    ind = m.group("ind")
    new = ind + _VECTORIZED[variant].format(ind=ind) + "\n"
    src = src[:m.start()] + new + src[m.end():]
    mod = importlib.import_module(cls.__module__)
    ns = {}
    exec(compile(src, f"<sq_fast_pack {cls.__module__}>", "exec"), mod.__dict__, ns)
    fn = ns["pack"]
    fn.__sq_fast_pack__ = True
    return fn


def apply():
    """
    Patch every importable auto-gptq QuantLinear whose pack() has the known loop.

    Inputs:
        - None

    Outputs:
        - patched (list[str]): modules patched; skipped ones are not listed
    """
    patched = []
    for modname, variant in TARGETS:
        try:
            cls = importlib.import_module(modname).QuantLinear
        except Exception:
            continue  # kernel-only module that does not import here
        if getattr(cls.pack, "__sq_fast_pack__", False):
            patched.append(modname)
            continue
        fn = vectorized_pack(cls, variant)
        if fn is not None:
            cls.pack = fn
            patched.append(modname)
    return patched
