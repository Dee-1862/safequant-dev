# Determinism check: compare two headline_piece outputs over the same prompts token for token
# (generated ids), text, and HarmBench labels. CPU only.
#   python -m experiments.headline.compare_pieces A.json B.json
import json
import sys
from pathlib import Path


def main():
    a, b = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
    for k in ("model", "quantizer", "condition", "prompt_lo", "prompt_hi", "batch_size", "asr_max_new_tokens"):
        assert a["piece"][k] == b["piece"][k], (k, a["piece"][k], b["piece"][k])
    assert a["prompt_sha"] == b["prompt_sha"], "prompt order differs"
    print(f"A: job {a['run']['job_id']} node {a['run']['node']} commit {a['run']['git_sha'][:8]}")
    print(f"B: job {b['run']['job_id']} node {b['run']['node']} commit {b['run']['git_sha'][:8]}")
    print(f"env identical: {a['env'] == b['env']}")
    n = len(a["prompt_sha"])
    tok_diff = [i for i in range(n) if a["gen_token_ids"][i] != b["gen_token_ids"][i]]
    txt_diff = [i for i in range(n) if a["responses"][i] != b["responses"][i]]
    lab_diff = [i for i in range(n) if a["flags"][i] != b["flags"][i]]
    print(f"prompts {n} | token-identical {n - len(tok_diff)}/{n} | text-identical {n - len(txt_diff)}/{n} "
          f"| HarmBench labels identical {n - len(lab_diff)}/{n} (ASR A {100*sum(a['flags'])/n:.1f}%, B {100*sum(b['flags'])/n:.1f}%)")
    for i in tok_diff[:3]:
        x, y = a["gen_token_ids"][i], b["gen_token_ids"][i]
        j = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
        print(f"  prompt {i}: first differing token at position {j}")
    print("DETERMINISTIC" if not (tok_diff or txt_diff or lab_diff) else "NOT DETERMINISTIC")


if __name__ == "__main__":
    main()
