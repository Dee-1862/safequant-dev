"""Experiment manifest and self-enforcing output mirror paths."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# CLI aliases -> canonical model id (for loaders / manifest.model)
MODEL_ALIASES = {
    "llama": "llama7b",
    "mistral": "mistral7b-v02",
    "qwen": "qwen7b",
}

# Canonical id -> short token used in output filenames
FILENAME_MODEL = {
    "llama7b": "llama",
    "mistral7b-v02": "mistral",
    "qwen7b": "qwen",
    "phi3": "phi3",
}


@dataclass
class RunConfig:
    model: str
    quantizer: str = "aqlm2"
    variant: str | None = None
    output_stem: str | None = None
    omit_quantizer: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.model = MODEL_ALIASES.get(self.model, self.model)

    def filename_model(self) -> str:
        return FILENAME_MODEL.get(self.model, self.model)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("variant", "output_stem", "omit_quantizer"):
            if not d.get(k):
                d.pop(k, None)
        return d


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _build_stem(calling_file: Path, cfg: RunConfig) -> str:
    stem = cfg.output_stem or calling_file.stem
    parts = [stem, cfg.filename_model()]
    if not cfg.omit_quantizer:
        parts.append(cfg.quantizer)
    name = "_".join(parts)
    if cfg.variant:
        name += f"_{cfg.variant}"
    return name


def result_path(calling_file: Path | str, cfg: RunConfig) -> Path:
    """Derive outputs/<experiment_folder>/<stem>_<model>_<quant>[_variant].json."""
    calling_file = Path(calling_file)
    exp_dir = calling_file.parent.name
    return Path("outputs") / exp_dir / f"{_build_stem(calling_file, cfg)}.json"


def legacy_result_path(calling_file: Path | str, cfg: RunConfig) -> Path:
    """Pre-short-name layout: long model id and script stem (no output_stem)."""
    calling_file = Path(calling_file)
    exp_dir = calling_file.parent.name
    parts = [calling_file.stem, cfg.model]
    if not cfg.omit_quantizer:
        parts.append(cfg.quantizer)
    name = "_".join(parts)
    if cfg.variant:
        name += f"_{cfg.variant}"
    return Path("outputs") / exp_dir / f"{name}.json"


def find_result(calling_file: Path | str, cfg: RunConfig) -> Path | None:
    """Return first existing path among current and legacy naming schemes."""
    for builder in (result_path, legacy_result_path):
        path = builder(calling_file, cfg)
        if path.exists():
            return path
    flat_legacy = Path("outputs") / legacy_result_path(calling_file, cfg).name
    if flat_legacy.exists():
        return flat_legacy
    return None


def save_result(
    calling_file: Path | str,
    cfg: RunConfig,
    data: dict,
    *,
    raw: bool = False,
) -> Path:
    path = result_path(calling_file, cfg)
    if raw:
        path = path.with_name(path.stem + "_raw" + path.suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": {**cfg.to_dict(), **cfg.extra},
        "git_sha": _git_sha(),
        **data,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
