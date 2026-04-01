# Repair

Applies sparse delta-W correction to the NF4 model based on the vulnerability scores from the diagnosis step.

**apply_repairs.py** - the main repair script. Loads the NF4 model, then for each targeting strategy, replaces the selected rows with their FP16 counterparts and saves the result. Five strategies are implemented to allow comparison:

- targeted: our method using V-score
- dfa_only: uses only DFA importance (ablation)
- error_only: uses only quantization error (ablation)
- random: picks random rows (control baseline)
- chhabra: AIRD on the worst-damaged layer (external baseline)

**targeting_validation.py** - validates that the V-score layer selection is meaningful by testing activation steering with top-k vs bottom-k layers. If steering from high-V-score layers reduces ASR more than low-V-score layers, the targeting is identifying the right things.

## How the Repair Works

We freeze the NF4 model and add a sparse correction on top:

    W_repaired = W_nf4 + delta_W

where delta_W is zero everywhere except at the targeted rows, where it equals W_fp16 - W_nf4. This is equivalent to full row replacement but framed as a sparse adapter.

The identity-matrix trick is used throughout to extract NF4 weights correctly on CPU, since BitsAndBytes' .dequantize() returns the wrong shape on CPU.
