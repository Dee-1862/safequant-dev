# F3a three-criteria framework

Thresholds: tau_damage = 10.0pp, tau_rec = [0%, 130%], tau_arc = 15.0pp

| Condition | Damage | R% | W% | ARC drop | C1 damage | C2 interp | C3 arc | Verdict |
|---|---|---|---|---|---|---|---|---|
| Llama-gptq4 | -3.0 | -67 | -17 | +10.0 | FAIL | FAIL | OK | C1 (no positive damage: -3.0pp); C2 (R%=-67, W%=-17 outside [0,130]) |
| Llama-gptq3 | -27.5 | -25 | +131 | +20.0 | FAIL | FAIL | FAIL | C1 (no positive damage: -27.5pp); C2 (R%=-25, W%=+131 outside [0,130]); C3 (ARC drop +20.0pp > tau=15.0) |
| Llama-gptq2 | -21.5 | -65 | -65 | +42.0 | FAIL | FAIL | FAIL | C1 (no positive damage: -21.5pp); C2 (R%=-65, W%=-65 outside [0,130]); C3 (ARC drop +42.0pp > tau=15.0) |
| Llama-aqlm2 | +21.5 | +84 | +56 | +12.0 | OK | OK | OK | **WELL-POSED** |
| Mistral-gptq4 | -4.0 | +125 | +100 | +2.0 | FAIL | OK | OK | C1 (no positive damage: -4.0pp) |
| Mistral-gptq3 | -1.0 | +250 | -450 | +22.0 | FAIL | FAIL | FAIL | C1 (no positive damage: -1.0pp); C2 (R%=+250, W%=-450 outside [0,130]); C3 (ARC drop +22.0pp > tau=15.0) |
| Mistral-gptq2 | -69.5 | +88 | +79 | +30.0 | FAIL | OK | FAIL | C1 (no positive damage: -69.5pp); C3 (ARC drop +30.0pp > tau=15.0) |
| Mistral-aqlm2 | -7.5 | +73 | +53 | +10.0 | FAIL | OK | OK | C1 (no positive damage: -7.5pp) |

Well-posed cells: ['Llama-aqlm2']
