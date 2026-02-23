# ADR-004: Commitment layer definition (0.85 threshold, 3 consecutive layers)

## Status: Accepted

## Context

We need an operational definition of "when does the model commit to a cross-lingual representation" -- i.e., at which layer the residual stream activations for the same concept in English and a target language become stably similar. A single threshold crossing is noisy due to layer-to-layer fluctuations in cosine similarity.

## Options Considered

| Option | Pro | Con |
|--------|-----|-----|
| First layer above threshold | Simple; single parameter | Noisy; a single spike at layer 5 followed by a drop would be counted as commitment |
| Majority vote across window | Robust to isolated spikes | Harder to interpret; variable window sizes produce different results |
| N consecutive layers above threshold | Intuitive ("stays above 0.85 for 3 layers"); robust to transient spikes; two clear parameters | Sensitive to exact threshold and N choice; may miss gradual convergence |

## Decision

Commitment layer is defined as the first layer `l` where cosine similarity >= 0.85 for 3 consecutive layers (`l`, `l+1`, `l+2`). This balances robustness against transient noise with sensitivity to genuine convergence.

Results are reported at multiple thresholds (0.75, 0.85, 0.90) in the analysis to demonstrate sensitivity to this methodological choice. The 0.85/3 combination is used as the primary reporting threshold.

## Consequences

- **Enables:** Clear, reproducible definition of commitment timing; directly comparable across variants and languages; simple to compute and explain.
- **Constrains:** The threshold is a methodological choice, not ground truth. Curves that rise gradually to 0.84 and plateau are reported as "never committed" (-1), which may undercount true alignment for some language pairs. Sensitivity analysis at alternative thresholds mitigates this.
