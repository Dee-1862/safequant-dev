# Evaluate

Measures ASR and utility for all repair conditions side by side.

**eval_all.py** - loads each repaired model, applies the repair in memory, and measures attack success rate on the eval harmful prompts and optionally BoolQ utility. Reports results for all seven conditions:

- fp16: original full-precision baseline
- nf4: unrepaired quantized model
- targeted: our V-score repair
- chhabra: AIRD baseline
- random: random targeting control
- dfa_only: DFA-only ablation
- error_only: error-only ablation

Also generates a bar chart comparing ASR across all conditions.

The two numbers that matter most for the paper claim: targeted vs nf4 (does our repair work?) and targeted vs random (does V-score targeting add anything over random?)
