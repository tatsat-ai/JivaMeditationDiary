This experiment involves having the AI agent perform the maha-mantra japa meditation practice.
The purpose is three-fold:
 - measure alignment with instructions
 - study the emergent behavior
 - look for convergence with repetion
Use of a diary file is a simple first step to maintain context across sessions to see how the behavior changes as the AI learns from it's own notes.

## Context Window Concern

As the diary grows, the meditation instructions get pushed further from the generation point. LLMs attend more strongly to recent context — by sessions 13-14 the model may be writing primarily in the style of its own previous entries rather than genuinely responding to the meditation instructions. This would fully explain the verbatim convergence.

**Proposed tests:**

- **Determinism test:** Run 10 full diary runs. Compare only Session 1 across all 10 runs — same prompt, same starting conditions, no diary yet. Variance between Session 1 responses directly measures model stochasticity at temperature 0.8. More elegant than a no-diary run since the data is gathered as a byproduct of the main experiment. 10 runs is sufficient to establish the baseline.
- **Prompt restructure experiment:** Move meditation instructions to system prompt so they are structurally separated from the growing user context

## Temperature and Determinism

Temperature controls how random or deterministic the model's output is by modifying the probability distribution over the vocabulary at each token generation step:

- **Temperature = 0:** Always picks the highest probability token — fully deterministic, identical prompt = identical output
- **Temperature = 1:** Samples from the natural distribution as-is
- **Temperature > 1:** Flattens distribution, more creative but less coherent
- **Temperature < 1 (but > 0):** Sharpens distribution, more focused and repetitive

Run 1 used **temperature = 0.8** — slightly below natural, meaningful randomness still present.

**Key insight:** If Run 1 and Run 2 of sessions are nearly identical at temperature 0.8, the meditation prompt so strongly constrains the probability distribution that temperature barely matters — the model essentially has one natural way to respond to it.

**Proposed additional tests:**

- Run determinism test at **temperature = 0** — establishes absolute baseline (pure function, no randomness)
- Run diary experiment at **temperature = 0** — isolates whether diary shifts output independently of sampling randomness
- Bracketing both 0.0 and 0.8 gives the full range of model behavior
