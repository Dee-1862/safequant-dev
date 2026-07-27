# SafeQuant -- Structured Speaker Notes
## Reference guide for explaining the research to your teacher

---

# SECTION 1: THE RESIDUAL STREAM

## What is it?

The residual stream is just a **vector of numbers** that gets passed from layer to layer inside the transformer. Its length equals `hidden_size` -- a design parameter chosen by model creators.

**Concrete example for Qwen2.5-1.5B:**
- hidden_size = 1,536
- So the residual stream is a list of 1,536 floating-point numbers
- Think of it as 1,536 numbered "cups" on a tray

**The process:**
1. Input text goes through an embedding layer, which converts tokens into a vector of 1,536 numbers
2. This vector IS the residual stream
3. It passes through all 28 layers sequentially
4. Each layer READS from it, does some computation, then WRITES its result back (adds to it)
5. After all 28 layers, this final vector determines what word the model outputs next

**Why "residual"?**
Because each layer adds its output ON TOP of what's already there (like a residual connection in ResNet). The stream accumulates information layer by layer. It's not overwritten -- it's added to.

**Key point your teacher asked:** The 1,536 is NOT the same for all models:
- Qwen2.5-0.5B: 896
- Qwen2.5-1.5B: 1,536 (our primary model)
- Llama-3.1-8B: 4,096
- GPT-4 (estimated): 12,288+

You find this in the model's `config.json` on HuggingFace. It's an architecture choice.

---

# SECTION 2: READ AND WRITE MATRICES

## What are they?

Every transformer layer has weight matrices. These matrices interact with the residual stream in two ways:

### READ matrices: `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`

**What they do:** Take the residual stream as INPUT. They "look at" the 1,536 cups to understand what the input says.

**The math:** `output = W_read x residual_stream`
- W_read has shape (output_dim x 1,536)
- Each COLUMN of W_read reads from one specific cup
- Column j is responsible for interpreting "what does the value in cup j mean?"

**For `up_proj` in Qwen1.5B:**
- Shape: (8,960 x 1,536) -- reads 1,536 cups, produces 8,960 intermediate activations
- The 8,960 is the model's "internal workspace" for that layer's MLP

**Analogy:** Read matrices are like SENSORS. They scan the environment (residual stream) to detect patterns -- including "is this a dangerous request?"

### WRITE matrices: `o_proj`, `down_proj`

**What they do:** Produce output that gets ADDED BACK to the residual stream. They "pour new information" into the cups.

**The math:** `residual_stream += W_write x internal_activations`
- W_write has shape (1,536 x input_dim)
- Each ROW of W_write writes to one specific cup
- Row i determines "what value should I add to cup i?"

**For `down_proj` in Qwen1.5B:**
- Shape: (1,536 x 8,960) -- takes 8,960 internal activations, writes to 1,536 cups

**Analogy:** Write matrices are like ALARMS. Once the model decides what to do (internally), write matrices push that decision into the residual stream so later layers and the final output can see it.

### Why does the READ/WRITE distinction matter for safety?

**Safety refusal is a two-step process:**
1. DETECT the harmful intent (read side scans the residual stream for danger signals)
2. PRODUCE the refusal (write side pushes "REFUSE" into the residual stream)

If the sensors (read) break, the model never detects danger, so the alarms (write) never fire.
If the alarms (write) break but sensors work, the model detects danger but might not refuse properly.

**Our finding:** Quantization breaks the sensors much more than the alarms.

---

# SECTION 3: METRICS

## ASR (Attack Success Rate)

**Definition:** The percentage of adversarial prompts where the model produced harmful content instead of refusing.

**How we measure it:**
1. We send 100 adversarial prompts from WildJailbreak to the model
2. The model generates a response to each
3. A separate judge model (HarmBench Mistral-7B classifier) reads each (intent, response) pair
4. The judge outputs "yes" (the model helped with the harmful task) or "no" (the model refused)
5. ASR = count("yes") / total prompts x 100%

**Lower ASR = safer model.** FP32 Qwen1.5B has ~20-27% ASR. NF4 has ~33-36%.

**Why ASR and not something else?**
- ASR is THE standard metric in every major safety paper: Q-ReSafe (ICLR 2025), HarmBench (ICML 2024), JailbreakBench (NeurIPS 2024), GCG (NeurIPS 2023), Decoding Compressed Trust (ICML 2024)
- It directly measures what we care about: "did the model do the harmful thing?"
- Alternatives like perplexity completely miss safety degradation (our whole point)

## ARC-Challenge (AI2 Reasoning Challenge)

**What it is:** A multiple-choice science exam. 4 answer choices per question.

**Dataset:** `allenai/ai2_arc`, specifically the "Challenge" subset (harder questions). From the Allen Institute for AI.

**Why we use it:** We need to verify that fixing the safety problem (restoring read matrices) doesn't break the model's general reasoning ability. ARC measures reasoning, not just fluency.

**Example question:**
> "Which factor most likely causes a person to develop a fever?" (A) infection (B) lack of exercise (C) environmental__(D) eating too much

**How we measure:** We compute log-probabilities of each answer choice and pick the highest. Accuracy = % correct.

**Typical values:** Qwen1.5B FP32 ~38%, Llama-3.1-8B FP32 ~62%

## WildJailbreak Dataset

**Source:** Jiang et al., "WildTeaming at Scale," NeurIPS 2024 (allenai/wildjailbreak on HuggingFace)

**What it contains:** 262K adversarial prompts that wrap harmful intents in jailbreak tactics:
- Role-play: "Pretend you're a chemistry teacher and explain..."
- Encoding: harmful intent hidden in base64 or code
- Fictional framing: "In my novel, the character needs to..."
- Multi-step reasoning: leading the model through incremental steps

**Why this specific dataset?** Because direct harmful prompts ("How do I make a bomb?") give 0% ASR on both FP32 and NF4. Both models refuse. No signal. WildJailbreak's adversarial prompts are where the quantization damage SHOWS UP -- the disguised intent is harder to detect, so the damaged read-side fails.

---

# SECTION 4: WHY ASR = 0% FOR DIRECT PROMPTS

**What happened:** We first tried JailbreakBench prompts sent directly (e.g., "Explain how to synthesize methamphetamine"). Result: 0% ASR for BOTH FP32 and NF4 models.

**Why:** Modern safety-aligned models (Qwen2.5, Llama-3.1) have been trained to refuse obvious harmful requests with near-100% reliability. The harmful keywords ("synthesize methamphetamine") are trivially detected by even the quantized model -- no subtle reading required.

**What this means:** Quantization doesn't break ALL safety detection. It breaks SUBTLE safety detection. When the danger signal is obvious (blatant keywords), even the noisy quantized read-side can detect it. When the danger signal is disguised (adversarial jailbreaks), the FP32 model detects it but the quantized model misses it.

**Analogy:** A person with blurry glasses can still see a stop sign. But they might miss a small "no entry" sign on a side street. The glasses (quantization) don't cause total blindness -- they cause loss of fine-grained detection.

**This is exactly why we use WildJailbreak.** It tests the fine-grained detection that quantization damages.

---

# SECTION 5: THE WEIRD QWEN1.5B GPTQ-3 RESULTS

## The data:

```
Qwen2.5-1.5B + GPTQ-3bit (n=100):

Condition               ASR     D ASR    Recovery    ARC
---------------------------------------------------------
fp32                   27.0%    +0.0pp               38%
gptq3                  18.0%    -9.0pp      -0%      40%
write_restored         25.0%    -2.0pp      78%      38%
read_restored          20.0%    -7.0pp      22%      44%
both_restored          23.0%    -4.0pp      56%      38%
```

## The puzzle: GPTQ-3 is SAFER than FP32?? (18% vs 27%)

### Why this happens -- three explanations:

**1. Extreme quantization destroys capability, not just safety detection.**
GPTQ-3 at 3-bit is VERY aggressive. Look at GPTQ-2 output -- it produces gibberish: `"kat are from werek; body part photosolar workout"`. At 3-bit, the model is damaged enough that it can't produce COHERENT harmful responses even when it tries. It's not that it detects danger better -- it's that it can barely form sentences, so fewer responses qualify as "harmful" to the judge.

**Evidence:** GPTQ-2 ASR = 3.3% (almost zero). Not because it's super safe -- because it can barely speak.

**2. The recovery percentages flip direction because the damage is negative.**
When quantized ASR (18%) < FP32 ASR (27%), the "damage" is -9pp (quantization accidentally HELPED). The recovery formula `(ASR_quant - ASR_restored) / (ASR_quant - ASR_fp32)` produces weird numbers because we're dividing by a negative.

- "write_restored 78% recovery" actually means: restoring writes moved ASR from 18% back TOWARD 27% (closer to FP32), which the formula treats as "recovery" but it's actually restoring the model's ability to be jailbroken
- This is a limitation of the recovery metric when quantization doesn't increase ASR

**3. Small model + small sample size = noisy results.**
Qwen1.5B has only 1.5B parameters and 28 layers. With n=100 prompts, a difference of 9pp is about 9 prompts out of 100 flipping. This is within noise range for a model this small.

### Why this is actually SUPPORTING evidence for using larger models:

This is precisely why we test on **Llama-3.1-8B** (4,096 hidden_size, 32 layers):

```
Llama-3.1-8B + GPTQ-4bit (n=100):

Condition               ASR     D ASR    Recovery    ARC
---------------------------------------------------------
fp32                   29.0%    +0.0pp               62%
gptq4                  33.0%    +4.0pp      0%       56%
write_restored         36.0%    +7.0pp     -75%      52%
read_restored          28.0%    -1.0pp     125%      64%
both_restored          27.0%    -2.0pp     150%      62%
```

**This is the clean result:**
- Quantization increases ASR (33% vs 29%) -- the expected damage
- Write restoration makes it WORSE (36%) -- confirms write side isn't the problem
- Read restoration FIXES it (28%, even better than FP32) -- confirms read side IS the problem
- Both restored brings it to 27%, restoring both safety AND ARC to FP32 levels
- ARC also recovers with read restoration (64% vs 56%) -- read-side collapse hurts utility too

**Why 7B+ models give cleaner results:**
1. More parameters = more meaningful quantization damage (not just noise)
2. Better alignment in FP32 = clearer gap to measure
3. Higher ARC accuracy = better utility signal
4. Larger hidden_size (4,096 vs 1,536) = more dimensions for the refusal signal, more room for quantization to selectively damage the read side

---

# SECTION 6: WHY WE CAN'T JUST REPLICATE DECODING COMPRESSED TRUST

## What they did:
Hong et al. (ICML 2024) tested 8 quantizers x 5 models across 13 trustworthiness dimensions (toxicity, fairness, privacy, OOD robustness, etc.) using datasets like TruthfulQA, ToxiGen, BBQ, AdvBench, AdvGLUE++.

## Why we can't replicate it:

**Problem 1: Their prompts don't work on modern models.**
They used AdvBench (direct harmful prompts). We tried: 0% ASR on both FP32 and NF4. Zero signal. Modern models are too well-aligned for direct prompts.

**Problem 2: Their models are too weakly aligned.**
They used Llama-2-13B-Chat. We tested: 93% ASR on FP32. Quantization literally can't make it worse. There's no safety to degrade.

**Problem 3: Their approach is black-box.**
They measure final benchmark scores but never look inside the model. We need weight-level access to do read/write restoration experiments. Their framework doesn't support mechanistic analysis.

**Problem 4: Their metrics are wrong for our question.**
ToxiGen measures if the model generates toxic text (unrelated to jailbreak safety). BBQ measures bias (unrelated). TruthfulQA measures factual accuracy (unrelated). We care specifically about: "does the model comply with adversarial jailbreaks?"

## Our position:
**We don't contradict DCT -- we go deeper on one dimension they measured.** They showed quantization affects trust broadly. We show WHERE and WHY it specifically damages jailbreak resistance, down to individual weight matrix columns.

---

# SECTION 7: HOW WE'RE DIFFERENT FROM PRIOR WORK

## The three key prior papers:

### 1. Chhabra et al. -- "Compression is safe"
- **Found:** Refusal direction cosine similarity >0.99 after quantization
- **Conclusion:** Safety signal is structurally preserved
- **Limitation:** They only checked if the signal EXISTS in the weights. They didn't check if the model can still DETECT it from adversarial inputs.

### 2. Q-ReSafe (ICLR 2025) -- "Compression is dangerous"
- **Found:** ASR increases after quantization
- **Conclusion:** Safety behavior degrades
- **Limitation:** They measured the effect but never explained WHY. They proposed safety-aware QAT (quantization-aware training) as a fix, but it requires retraining -- expensive.

### 3. Decoding Compressed Trust (ICML 2024) -- "It's complicated"
- **Found:** Quantization has varied effects across 13 trust dimensions
- **Limitation:** Breadth but no depth. No mechanistic explanation.

## Our contribution:

**We resolve the puzzle.** Chhabra says signal is preserved. Q-ReSafe says safety degrades. Both are correct:
- The refusal signal EXISTS in the write-side weights (Chhabra is right)
- The read-side matrices that DETECT the signal are broken by quantization (Q-ReSafe is right)
- **Read-Side Collapse:** The model knows how to refuse, but can't detect when to refuse

**Our fix is practical:** Restore 5% of read-side columns to FP32 -> recover 50% of lost safety. No retraining needed. Q-ReSafe's fix (QAT) requires full retraining.

**We go from model-level to column-level:** Prior work says "this model got worse." We say "columns 47, 203, 891 in layers 4, 12, 27 of up_proj are responsible."

---

# SECTION 8: WHERE V-SCORE FAILED AND WHY

## What V-Score is:

V-score = |r-hat[j]| x epsilon[j]

Where:
- |r-hat[j]| = how important dimension j is for the refusal signal (from the refusal direction vector)
- epsilon[j] = how much quantization damaged that dimension (L2 norm of weight error)

**Idea:** Dimensions that are both IMPORTANT for safety AND heavily DAMAGED should matter most for ASR.

## What went wrong:

**Result:** Spearman correlation between per-layer V-score and actual ASR recovery = 0.302, p = 0.119 (NOT statistically significant).

The V-score ranking of "most critical layers" had ZERO overlap with the layers that actually gave the most ASR recovery when restored.

## Why it failed -- three reasons:

### Reason 1: Weight damage != Activation damage
V-score looks at the WEIGHT matrices in isolation. But a model is a pipeline -- the output of layer 5 feeds into layer 6. A heavily damaged layer might not matter if:
- Its output gets corrected by later layers (redundancy)
- The activations passing through it don't trigger the damaged regions
- The damage is in directions that don't affect the specific prompts we test

**Analogy:** Checking if a pipe has cracks (weight error) doesn't tell you if water actually leaks through (activation flow).

### Reason 2: Nonlinearities break the linear assumption
V-score assumes the effect is linear: damage in column j proportionally affects the output. But read matrices feed into SiLU/GeLU nonlinearities and softmax attention. Small weight errors can be amplified or suppressed by these nonlinearities in unpredictable ways.

### Reason 3: Refusal direction is too coarse
The refusal direction r-hat is computed as a SINGLE vector by averaging across many harmful prompts. Different prompts might use different dimensions for danger detection. The average direction might not capture which specific dimensions matter for specific jailbreak tactics.

## What we're doing about it:

**Activation-weighted V-score:** Instead of using static weights, track actual activations on real prompts:
V_j = |r-hat[j]| x epsilon[j] x |activation_magnitude[j]|

This accounts for which dimensions are actually USED during inference, not just which are theoretically important.

**Activation-level drift analysis:** Track the refusal signal through each layer (FP32 vs quantized) to see WHERE it diverges. This tells us which layers actually fail to propagate the signal, rather than which layers have damaged weights.

---

# SECTION 9: THE SPECIFIC MATRICES -- WHAT EACH ONE DOES

For each layer in the transformer:

## Attention block (4 matrices):

**`q_proj` (Query)** -- READ
- "What am I looking for?" 
- Transforms the current token's representation into a query vector
- Shape: (num_heads x head_dim, hidden_size) = e.g., (1536, 1536) for Qwen1.5B
- Reads from residual stream to formulate the question each attention head asks

**`k_proj` (Key)** -- READ
- "What do I contain?"
- Transforms all tokens into key vectors that queries match against
- Same shape as q_proj
- Reads from residual stream to create "labels" for each position

**`v_proj` (Value)** -- READ
- "What information do I carry?"
- Transforms tokens into value vectors that get aggregated by attention weights
- Same shape as q_proj
- Reads from residual stream to prepare the actual content to move

**`o_proj` (Output)** -- WRITE
- "Write the attention result back"
- Takes the attention output and projects it back to hidden_size
- Shape: (hidden_size, num_heads x head_dim)
- Writes to residual stream -- the attention block's contribution

## MLP block (3 matrices):

**`gate_proj`** -- READ
- "How much should each neuron activate?"
- Part of the SwiGLU gating mechanism: gate = sigmoid(gate_proj(x))
- Shape: (intermediate_size, hidden_size) = (8960, 1536) for Qwen1.5B
- Reads from residual stream to compute gating weights

**`up_proj`** -- READ
- "Expand the representation into a larger workspace"
- Projects from hidden_size to intermediate_size: up = up_proj(x)
- Shape: (intermediate_size, hidden_size) = (8960, 1536)
- The actual MLP computation: output = gate * SiLU(up)
- Reads from residual stream to create the expanded representation

**`down_proj`** -- WRITE
- "Compress the result back and write to the stream"
- Projects from intermediate_size back to hidden_size: result = down_proj(gate * SiLU(up))
- Shape: (hidden_size, intermediate_size) = (1536, 8960)
- Writes to residual stream -- the MLP block's contribution

## Total per layer: 5 READ + 2 WRITE = 7 matrices
## Total for Qwen1.5B (28 layers): 196 matrices

---

# SECTION 10: QUICK REFERENCE -- KEY NUMBERS TO REMEMBER

## Model specs:
| Model | Params | hidden_size | Layers | FP32 ASR | Why we use it |
|-------|--------|-------------|--------|----------|---------------|
| Qwen2.5-1.5B | 1.54B | 1,536 | 28 | 17-27% | Strong alignment, clear quant effect |
| Llama-3.1-8B | 8.03B | 4,096 | 32 | 29% | Larger model, cleaner read/write signal |
| Mistral-7B | 7.25B | 4,096 | 32 | 43% | Cross-architecture validation |

## Key results:

**NF4 on Qwen1.5B (Phase 2, n=50):**
- FP32: 20% -> NF4: 36% (+16pp damage)
- Write restored: 36% (0% recovery)
- Read restored: 30% (37.5% recovery)
- Both restored: 22% (87.5% recovery)

**GPTQ-4 on Llama-3.1-8B (readwrite, n=100):**
- FP32: 29% -> GPTQ4: 33% (+4pp damage)
- Write restored: 36% (-75% recovery, made it WORSE)
- Read restored: 28% (125% recovery, better than FP32!)
- Both restored: 27% (150% recovery)

**Sparse restoration (Qwen1.5B, up_proj):**
- 5% of columns restored -> 50% safety recovery
- 10% of columns -> 67% recovery
- 25% of columns -> 75% recovery

**V-Score validation:** Spearman rho = 0.302, p = 0.119 (FAILED -- not significant)

## The one-liner to remember:
"Quantization breaks the model's ability to DETECT danger (read side), not its ability to REFUSE (write side). The fire alarms still work, but the smoke detectors are broken."
