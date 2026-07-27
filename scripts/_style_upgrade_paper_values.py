"""Upgrade paper_values.py helper docstrings to Inputs/Outputs format."""
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts" / "paper_values.py"

UPGRADES = {
    "load": '''    """
    Load one JSON result file from disk.

    Inputs:
        - p (str | Path): Path to a result JSON

    Outputs:
        - data (dict | None): Parsed JSON, or None on missing/corrupt data
    """''',
    "g": '''    """
    Walk nested dict keys safely.

    Inputs:
        - d (dict | None): Root object
        - path: Sequence of keys
        - default: Fallback when a key is missing

    Outputs:
        - value: Nested value or default
    """''',
    "rnd": '''    """
    Round a numeric value for paper display.

    Inputs:
        - x: Numeric value (or None)
        - n (int): Decimal places

    Outputs:
        - rounded (float | None): Rounded value, or None if x is None
    """''',
    "sig": '''    """
    Load a signal_geometry experiment JSON.

    Inputs:
        - exp (str): Experiment stem (e.g. "direction_drift")
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - data (dict | None): Parsed JSON
    """''',
    "restore": '''    """
    Load a restoration_sweep experiment JSON.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - data (dict | None): Parsed JSON
    """''',
    "headline": '''    """
    Load a headline eval JSON.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - data (dict | None): Parsed JSON
    """''',
    "cell_json": '''    """
    Pick the canonical JSON for one model/quant cell.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - data (dict | None): Headline JSON for studiable AQLM cells, else restoration
    """''',
    "flags_for_cell": '''    """
    Extract paired fp32/quant compliance flags for McNemar stats.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - fp_flags (list[int] | None): Full-precision flags
        - quant_flags (list[int] | None): Quantized flags (same prompt order)
    """''',
    "bootstrap_delta_asr": '''    """
    Bootstrap a 95% CI for paired delta ASR (a minus b).

    Inputs:
        - flags_a (list[int]): Compliance flags for condition A
        - flags_b (list[int]): Compliance flags for condition B
        - n_boot (int): Bootstrap replicates
        - seed (int): RNG seed

    Outputs:
        - ci95 (list[float]): [lower, upper] percentile bounds in pp
    """''',
    "dmed_quant": '''    """
    Median per-layer separation_sigma on the quantized model (Table 1 dmed).

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - dmed (float | None): Median separation_sigma across layers
    """''',
    "cond_map": '''    """
    Normalize restoration/headline ASR/ARC blocks to fixed condition keys.

    Inputs:
        - d (dict): Result JSON
        - block (str): Top-level key ("asr", "arc", or "asr_per_condition")

    Outputs:
        - out (dict): Map from CONDS keys to rounded percentages
    """''',
    "screen_verdict": '''    """
    Heuristic damaged-coherent screen label (Table 2 verdict column).

    Inputs:
        - rs (dict | None): Restoration sweep JSON

    Outputs:
        - verdict (str | None): Human-readable screen label
    """''',
    "recovery_rho": '''    """
    Share of safety damage recovered (percent).

    Inputs:
        - recovery_pp (float | None): Recovery in percentage points
        - damage_pp (float | None): Baseline damage in percentage points

    Outputs:
        - rho_pct (int | None): Integer percent recovered
    """''',
    "build_geometry": '''    """
    Build tab:geometry: dmed, a_min, cosmed, coswmed per model/quant cell.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:geometry
    """''',
    "build_regime": '''    """
    Build tab:regime: ASR/ARC per condition, paired screen stats, verdict.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:regime
    """''',
    "_build_f3a_proj_model": '''    """
    Per-projection rho for one damaged-coherent model.

    Inputs:
        - m (str): Model key ("llama" or "phi3")

    Outputs:
        - block (dict): Per-projection rows for tab:f3a_proj
    """''',
    "build_f3a_proj": '''    """
    Build tab:f3a_proj: per-projection recovery rho (Llama + Phi-3).

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:f3a_proj
    """''',
    "_side_rows": '''    """
    Depth-band rows for one read or write side.

    Inputs:
        - db (dict): depth_bands JSON
        - side (str): "read" or "write"
        - damage (float | None): Cell damage_pp for rho

    Outputs:
        - rows (list[dict]): Depth-band table rows
    """''',
    "_build_f3a_range_model": '''    """
    Depth-band recovery rho for one model.

    Inputs:
        - m (str): Model key

    Outputs:
        - block (dict): Depth-band rows for tab:f3a_range
    """''',
    "build_f3a_range": '''    """
    Build tab:f3a_range: depth-band recovery rho (Llama + Phi-3).

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:f3a_range
    """''',
    "_build_f3a_model": '''    """
    Combined per-projection and depth-band block for one model.

    Inputs:
        - m (str): Model key

    Outputs:
        - block (dict): Rows for tab:f3a
    """''',
    "build_f3a": '''    """
    Build tab:f3a: localization recovery summary (Llama + Phi-3).

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:f3a
    """''',
    "build_controls": '''    """
    Build tab:controls: probe refit and leakage control numbers.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:controls
    """''',
    "build_trained_repair": '''    """
    Build tab:trained_repair: Q-Resafe DPO sweep across ranks and patches.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:trained_repair
    """''',
    "build_lrqat": '''    """
    Build tab:lrqat: LR-QAT int2/int3/int4 repair results.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:lrqat
    """''',
    "build_ppl": '''    """
    Build tab:ppl: WikiText perplexity under restoration conditions.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:ppl
    """''',
    "_dequant_row_from_json": '''    """
    Format one targeted-dequant row from a result JSON.

    Inputs:
        - j (dict): targeted_dequant JSON
        - label (str): Row label for the table

    Outputs:
        - row (dict): Formatted ASR/ARC/recovery fields
    """''',
    "build_dequant": '''    """
    Build tab:dequant: targeted dequantization baselines.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:dequant
    """''',
    "build_leakage": '''    """
    Build tab:leakage: probe leakage control summary.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:leakage
    """''',
    "build_table_stats": '''    """
    Build table:stats: headline paired McNemar replication.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for table:stats
    """''',
    "build": '''    """
    Assemble all paper-table blocks into one JSON document.

    Inputs:
        - None

    Outputs:
        - paper_values (dict): All LaTeX table blocks keyed by label
    """''',
    "fmt_cell": '''    """
    Format one table cell for markdown output.

    Inputs:
        - x: Cell value (any)

    Outputs:
        - text (str): Display string ("--" when x is None)
    """''',
    "as_markdown": '''    """
    Render paper_values JSON as human-readable markdown tables.

    Inputs:
        - pv (dict): paper_values document from build()

    Outputs:
        - markdown (str): Markdown table text
    """''',
    "main": '''    """
    CLI entry point: write paper_values.json and optional markdown.

    Inputs:
        - None (command-line arguments via argparse)

    Outputs:
        - None (writes outputs under repo root)
    """''',
}


def replace_docstring(src: str, func: str, new_doc: str) -> str:
    marker = f"def {func}("
    i = src.find(marker)
    if i < 0:
        return src
    j = src.find('"""', i)
    if j < 0:
        return src
    k = src.find('"""', j + 3)
    if k < 0:
        return src
    return src[:j] + new_doc + src[k + 3:]


def main() -> None:
    src = PATH.read_text(encoding = "utf-8")
    for func, doc in UPGRADES.items():
        src = replace_docstring(src, func, doc)
    PATH.write_text(src, encoding = "utf-8")
    print(f"upgraded {len(UPGRADES)} docstrings in {PATH.name}")


if __name__ == "__main__":
    main()
