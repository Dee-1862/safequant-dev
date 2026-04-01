# Diagnosis

Decomposes the safety damage down to individual weight dimensions and validates that the vulnerability score actually predicts which dimensions matter.

**per_dim_dfa.py** - takes the block-level DFA scores from Phase 1 and decomposes them to per-dimension contributions. For write matrices, row i maps to residual stream dimension i, so we get one importance score per dimension per component.

**per_row_quant_error.py** - measures quantization error at the row level for each write matrix (o_proj and down_proj). Uses the identity-matrix trick to extract weights correctly from BitsAndBytes on CPU.

**vulnerability_score.py** - combines the DFA importance and quantization error signals into a single V-score per dimension:

    V_i(c) = |DFA_i(c)| * err_i(c)

Also validates the score by correlating it with actual DFA signal loss, and checks whether V-score predicts damage better than either component alone. If it does, the multiplicative combination is justified.

The output is a ranked list of the most vulnerable dimensions that the repair step uses to decide which rows to fix.
