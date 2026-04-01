# SafeQuant

Research project investigating how NF4 quantization degrades AI safety behavior in instruction-tuned LLMs.

The central finding is that quantization damages the "read side" of the safety circuit more than the "write side." In other words, the model loses its ability to detect that a prompt is harmful, rather than losing the ability to produce a refusal. We call this Read-Side Collapse.

## Project Structure

```
safequant/
  phase1_problem/        identify the problem and measure how bad it is
  phase2_solution/       diagnose where the damage is and repair it
  utils/                 shared utilities used across phases
  config.py              shared configuration (models, paths, NF4 settings)
  requirements.txt       Python dependencies
```

## How to Run

Install dependencies:
```
pip install -r requirements.txt
```

Run Phase 1 first to extract directions and measure behavior:
```
python phase1_problem/data/prepare_data.py
python phase1_problem/step1_direction/extract_activations.py --model qwen1.5b --precision fp16
python phase1_problem/step1_direction/extract_activations.py --model qwen1.5b --precision nf4
python phase1_problem/step2_behavior/measure_asr.py --model qwen1.5b --precision fp16
python phase1_problem/step2_behavior/measure_asr.py --model qwen1.5b --precision nf4
```

Then run Phase 2 to diagnose and repair:
```
python phase2_solution/diagnosis/per_dim_dfa.py --model qwen1.5b
python phase2_solution/diagnosis/per_row_quant_error.py --model qwen1.5b
python phase2_solution/diagnosis/vulnerability_score.py --model qwen1.5b
python phase2_solution/repair/apply_repairs.py --model qwen1.5b --top_k 100
python phase2_solution/evaluate/eval_all.py --model qwen1.5b
```

For the read-side vs write-side restoration experiment:
```
python phase2_solution/experiments/read_write_restoration.py --model qwen1.5b --n_eval 50
```

## Hardware Note

All scripts run on CPU. No GPU required. The NF4 quantization uses BitsAndBytes 0.49.2+ which supports CPU inference with float32 compute dtype.
