# ADR-001: HuggingFace Transformers over TransformerLens / raw PyTorch hooks

## Status: Accepted (revised)

## Context

We need clean activation capture across all residual stream layers of Tiny Aya variants (36 transformer blocks + embedding layer = 37 total) without modifying the forward pass. The framework must return structured activation tensors for downstream cosine similarity analysis.

## Options Considered

| Option | Pro | Con |
|--------|-----|-----|
| TransformerLens | Standard tool for mech-interp; well-documented hook API; `run_with_cache` returns all activations | Does not support Cohere2 architecture (`convert_hf_model_config` fails); ~200MB dependency |
| HuggingFace Transformers `output_hidden_states=True` | Native support for Cohere2; no extra dependency; returns all hidden states in a single forward pass | No named hook API; returns tuple of tensors rather than named cache; slightly less ergonomic for causal interventions |
| Raw PyTorch hooks | No extra dependency; full control over hook registration | Manual lifecycle management; error-prone for 37-layer capture; no standardized activation naming |

## Decision

Use HuggingFace Transformers with `output_hidden_states=True`. TransformerLens was the original choice but does not support Cohere2ForCausalLM (Tiny Aya's architecture). HF Transformers natively supports Cohere2 and returns all hidden states as a tuple from a single forward pass, which is sufficient for our cosine similarity analysis.

## Consequences

- **Enables:** Single-call activation extraction for all 37 layers; native Cohere2 support; no additional dependencies beyond transformers.
- **Constrains:** No named hook API; causal intervention experiments (activation patching) would require manual hook registration or a different framework. This is acceptable since this analysis is observational, not interventional.
