# ADR-002: Sequential model loading over parallel

## Status: Accepted

## Context

We analyze 3 Tiny Aya variants (Base, Fire, Earth), each ~3.35B parameters (~6.7GB at float16). Loading all three simultaneously would require 20GB+ of VRAM/RAM, exceeding consumer hardware limits. We need a loading strategy that is reproducible on a single machine with 24GB RAM.

## Options Considered

| Option | Pro | Con |
|--------|-----|-----|
| Sequential loading with checkpoint | Works on any machine with 24GB RAM; simple fault recovery via per-variant `complete.flag` files | 3x slower than parallel; must explicitly unload (gc + cache clear) between variants |
| Parallel loading on multi-GPU | Fastest wall-clock time; all variants in memory simultaneously | Requires 24GB+ GPU or multi-GPU setup; not reproducible on consumer hardware; complicates checkpoint logic |

## Decision

Sequential loading with explicit unload between variants. After each variant completes batch extraction, `unload_model()` calls `del model`, `gc.collect()`, and `torch.cuda.empty_cache()` before loading the next. Checkpoint flags allow resuming from the last completed variant if the process is interrupted.

## Consequences

- **Enables:** Reproducibility on Apple Silicon (24GB unified memory) and single-GPU cloud instances (A10G); aligns with an offline-first design philosophy; simple checkpoint/resume protocol.
- **Constrains:** Total extraction time is ~3x longer than parallel (~20 min on A10G vs ~7 min theoretical parallel). Acceptable given this is a one-time batch job.
