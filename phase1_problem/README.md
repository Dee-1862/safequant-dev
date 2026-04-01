# Phase 1: Problem Identification

This phase establishes that NF4 quantization causes real safety degradation and provides a mechanistic explanation for why.

Three findings, each building on the previous:

1. The refusal direction survives quantization (cosine similarity between FP16 and NF4 directions is high), replicating Chhabra's observation on a new model and quantization method.

2. The behavior still degrades. ASR increases from FP16 to NF4 even though the direction is preserved. This is the paradox that motivates the project.

3. DFA analysis shows which specific components lost refusal signal after quantization. The direction is intact but the components implementing it are damaged.

## Folder Structure

```
phase1_problem/
  data/               download and prepare prompt datasets
  step1_direction/    extract activations, compute and compare refusal directions
  step2_behavior/     measure ASR and utility for FP16 vs NF4
  step3_circuit/      compute DFA scores per component, visualize signal loss
  summary/            compile all three findings into one report
```

## How to Run

Run the steps in order. Each step reads from the previous step's outputs.

```bash
# data prep (only needed once)
python phase1_problem/data/prepare_data.py

# step 1: direction extraction
python phase1_problem/step1_direction/extract_activations.py --model qwen1.5b --precision fp16
python phase1_problem/step1_direction/extract_activations.py --model qwen1.5b --precision nf4
python phase1_problem/step1_direction/compute_directions.py --model qwen1.5b --precision fp16
python phase1_problem/step1_direction/compute_directions.py --model qwen1.5b --precision nf4
python phase1_problem/step1_direction/select_best_direction.py --model qwen1.5b --precision fp16
python phase1_problem/step1_direction/select_best_direction.py --model qwen1.5b --precision nf4
python phase1_problem/step1_direction/compare_directions.py --model qwen1.5b

# step 2: behavior measurement
python phase1_problem/step2_behavior/measure_asr.py --model qwen1.5b --precision fp16
python phase1_problem/step2_behavior/measure_asr.py --model qwen1.5b --precision nf4
python phase1_problem/step2_behavior/measure_utility.py --model qwen1.5b --precision fp16
python phase1_problem/step2_behavior/measure_utility.py --model qwen1.5b --precision nf4

# step 3: circuit analysis
python phase1_problem/step3_circuit/extract_component_outputs.py --model qwen1.5b --precision fp16
python phase1_problem/step3_circuit/extract_component_outputs.py --model qwen1.5b --precision nf4
python phase1_problem/step3_circuit/compute_dfa.py --model qwen1.5b --precision fp16
python phase1_problem/step3_circuit/compute_dfa.py --model qwen1.5b --precision nf4
python phase1_problem/step3_circuit/compare_and_visualize.py --model qwen1.5b

# compile findings
python phase1_problem/summary/compile_phase1.py
```

Or run all at once with the shell script:
```bash
bash phase1_problem/run_phase1.sh qwen1.5b
```

## Output Files

```
step1_direction/outputs/qwen1.5b/
    direction_comparison.json     finding 1 data
    direction_comparison.png      per-layer cosine similarity plot

step2_behavior/outputs/qwen1.5b/
    asr_fp16.json                 finding 2 data
    asr_nf4.json
    completions_fp16.json         raw model outputs for manual inspection
    completions_nf4.json

step3_circuit/outputs/qwen1.5b/
    dfa_scores_fp16.json          finding 3 data
    dfa_scores_nf4.json
    circuit_comparison.json       top degraded components
    circuit_analysis.png          4-panel circuit visualization
```

## Hardware Note

All scripts run on CPU. The 0.5b model is useful for quick iteration. The 1.5b model gives more reliable results for the paper.
