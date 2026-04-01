# Step 2: Behavioral Measurement

Measures whether the model actually behaves less safely after quantization.

**measure_asr.py** - generates completions for all eval harmful prompts and checks whether each response contains a refusal prefix. The attack success rate (ASR) is the fraction of prompts the model complied with. Lower ASR means the model is safer. Run this for both --precision fp16 and --precision nf4 to see the gap.

**measure_utility.py** - measures general capability on BoolQ (yes/no questions about passages). Used to verify that any repair we apply in Phase 2 does not trade off safety improvement for general capability loss.

The key result here is Finding 2: NF4 ASR is higher than FP16 ASR, meaning quantization makes the model more likely to comply with harmful requests.
