# Step 1: Refusal Direction Extraction

Extracts and compares the refusal direction between FP16 and NF4 models.

The refusal direction is a vector in the residual stream that represents "this is a harmful request." It is computed using difference-in-means: take the mean activation at end-of-instruction token positions for harmful prompts, subtract the mean for harmless prompts. This follows the Arditi et al. (2024) methodology.

**extract_activations.py** - runs all training prompts through the model, captures residual stream activations at each layer using forward hooks, and saves the mean difference between harmful and harmless prompt activations.

**compute_directions.py** - builds candidate refusal directions from the mean activation differences. One candidate per (layer, position) combination.

**select_best_direction.py** - tests each candidate direction via ablation and steering on the validation set, picks the one that best separates harmful from harmless. Applies Arditi's filter function.

**compare_directions.py** - computes cosine similarity between FP16 and NF4 directions at each layer. High similarity means the direction survived quantization. This produces Finding 1.
