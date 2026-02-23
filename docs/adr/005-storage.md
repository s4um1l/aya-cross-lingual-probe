# ADR-005: float16 activation storage over float32

## Status: Accepted

## Context

Activation extraction produces one tensor per stimulus per variant: shape (37, 2048) representing residual stream vectors at each of 37 layers (embedding + 36 hidden). For 700 stimuli across 3 variants, storage format directly affects disk usage and I/O speed during analysis.

## Options Considered

| Option | Pro | Con |
|--------|-----|-----|
| float32 | Full precision; no quantization error | ~318MB total (3 variants x 700 stimuli x 37 x 2048 x 4 bytes); slower I/O |
| float16 | ~159MB total (2x smaller); faster I/O; matches model inference dtype | Small precision loss (~0.1% relative error on cosine similarity for large-norm vectors) |

## Decision

Store activations as float16 (numpy `.npy` files). Residual stream activations in transformer models are large-norm vectors (typical L2 norm 50-200), so the relative precision loss from float16 is negligible for cosine similarity computation. Analysis code casts to float32 at load time for numerical stability in the dot product.

## Consequences

- **Enables:** ~159MB total storage (fits easily on disk); faster numpy load/save; matches the dtype used during model inference (float16), avoiding unnecessary precision that the model never had.
- **Constrains:** Activations very close to zero (norm < 1e-4) may lose meaningful precision. This is not a concern for residual stream vectors but would matter for attention pattern storage (not used in this analysis).
