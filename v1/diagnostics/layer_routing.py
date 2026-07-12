"""Backward-compat shim — library moved to core/residuals.py; analysis to attic/."""
from core.residuals import (  # noqa: F401
    extract_last_token_residuals,
    get_inner,
    prompt_text,
)

# Local aliases for archived scripts only (not exported from core/).
_get_inner = get_inner
format_prompt = prompt_text
