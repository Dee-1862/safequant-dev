"""Backward-compat shim — library moved to core/residuals.py; analysis to attic/."""
from core.residuals import (  # noqa: F401
    _get_inner,
    extract_last_token_residuals,
    format_prompt,
    get_inner,
    prompt_text,
)
