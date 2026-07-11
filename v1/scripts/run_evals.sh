#!/bin/bash
echo "Running activation_swap..."
python diagnostics/activation_swap.py --model llama7b --quantizer aqlm2 > outputs/run_activation_swap.log 2>&1

echo "Running probe_causal_ablation..."
python diagnostics/probe_causal_ablation.py --model llama7b --quantizer aqlm2 > outputs/run_probe_causal_ablation.log 2>&1

echo "Running direction_drift sensitivity..."
python diagnostics/direction_drift.py --model llama7b --quantizer aqlm2 --dataset harmbench > outputs/run_direction_drift.log 2>&1

echo "Running refusal_probe for 9 models..."
for q in awq gptq4 gptq3 gptq2; do
  python diagnostics/refusal_probe.py --model mistral7b-v02 --quantizer $q >> outputs/run_refusal_probes.log 2>&1
  python diagnostics/refusal_probe.py --model qwen1.5b --quantizer $q >> outputs/run_refusal_probes.log 2>&1
done
python diagnostics/refusal_probe.py --model llama7b --quantizer awq >> outputs/run_refusal_probes.log 2>&1

echo "Done"
