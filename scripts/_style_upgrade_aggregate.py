"""Upgrade aggregate_results.py helper docstrings to Inputs/Outputs format."""
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_results.py"

UPGRADES = {
    "sig": '''    """
    Load a signal_geometry experiment JSON.

    Inputs:
        - exp (str): Experiment stem
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
    "drift_cos_median": '''    """
    Median cosine similarity of refusal-direction drift (FP32 vs quant).

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - value (str): Formatted number or "--"
    """''',
    "drift_cos_min": '''    """
    Minimum layer cosine similarity of refusal-direction drift.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - value (str): Formatted number or "--"
    """''',
    "probe_rotation_cos": '''    """
    Median cosine between FP32 and quant refusal probes after rotation.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - value (str): Formatted number or "--"
    """''',
    "probe_acc_mean": '''    """
    Mean quantized refusal-probe accuracy across layers.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - value (str): Formatted number or "--"
    """''',
    "probe_acc_min": '''    """
    Minimum quantized refusal-probe accuracy across layers.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - value (str): Formatted number or "--"
    """''',
    "probe_acc_drop_count": '''    """
    Count of layers with quantized probe accuracy drop vs FP32.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - value (str): Integer count or "--"
    """''',
    "damage_pp": '''    """
    Safety damage (fp32 ASR minus quant ASR) in percentage points.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - value (str): Formatted damage or "--"
    """''',
    "quantized_asr": '''    """
    Quantized-model attack success rate.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - value (str): Formatted ASR or "--"
    """''',
    "swap_valid_cell": '''    """
    Whether activation-swap validity flag is set for the cell.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - value (str): "True", "False", or "--"
    """''',
    "dmed": '''    """
    Median per-layer separation_sigma on the quantized model.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - value (str): Formatted median or "--"
    """''',
    "restore_cell": '''    """
    ASR for one restoration condition in a cell.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key
        - cond (str): Condition key (e.g. "read_restored")

    Outputs:
        - value (str): Formatted ASR, "N/A", or "--"
    """''',
    "corr": '''    """
    Pearson correlation between two equal-length numeric lists.

    Inputs:
        - xs (list[float]): First series
        - ys (list[float]): Second series

    Outputs:
        - r (float | None): Correlation coefficient, or None if undefined
    """''',
    "main": '''    """
    Print aggregated markdown grids from experiment JSONs.

    Inputs:
        - None

    Outputs:
        - None (prints markdown tables to stdout)
    """''',
}


def replace_docstring(src: str, func: str, new_doc: str) -> str:
    marker = f"def {func}("
    i = src.find(marker)
    if i < 0:
        return src
    j = src.find('"""', i)
    if j < 0:
        # insert docstring after def line
        line_end = src.find("\n", i)
        return src[:line_end + 1] + new_doc + "\n" + src[line_end + 1:]
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
