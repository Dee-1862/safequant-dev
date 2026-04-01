# Step 3: Circuit Analysis

Identifies which attention and MLP components are responsible for the refusal signal, and which ones lost that signal after quantization.

**extract_component_outputs.py** - runs harmful prompts through both models and records the output of each attention block and MLP block at the last token position.

**compute_dfa.py** - computes a DFA (Direct Feature Attribution) score for each component. DFA measures how much the component's mean output projects onto the refusal direction. A high DFA score means the component is actively pushing the residual stream toward refusal.

**compare_and_visualize.py** - compares DFA scores between FP16 and NF4 for every component, finds the ones that dropped the most, and generates a 4-panel visualization. This produces Finding 3: specific components lost significant refusal signal after quantization.

The DFA scores from this step feed directly into Phase 2 where we use them to compute per-dimension vulnerability scores.
