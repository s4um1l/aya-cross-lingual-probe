"""
HuggingFace transformers wrapper for sequential Tiny Aya variant loading.

Uses AutoModelForCausalLM + output_hidden_states=True for activation extraction
instead of TransformerLens, since TL does not support Cohere2 architecture.

NEVER load more than one model simultaneously -- 24GB RAM limit on Apple Silicon.
Each variant must be fully unloaded before loading the next.

Architecture: Cohere2ForCausalLM (3.35B params, ~256k vocab, GQA, sliding window)
Models are gated on HuggingFace -- ensure you have accepted the license agreement
at each model page before running.

Usage:
    from model_loader import load_model, unload_model, VARIANT_CONFIGS
    model, tokenizer = load_model("base")
    # ... do work ...
    unload_model(model)
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ── Variant configurations ─────────────────────────────────────────────────
VARIANT_CONFIGS: dict[str, dict] = {
    "base": {
        "hf_name": "CohereLabs/tiny-aya-base",
        "n_layers": None,  # auto-detected from config
        "d_model": None,   # auto-detected from config
    },
    "fire": {
        "hf_name": "CohereLabs/tiny-aya-fire",
        "n_layers": None,
        "d_model": None,
    },
    "earth": {
        "hf_name": "CohereLabs/tiny-aya-earth",
        "n_layers": None,
        "d_model": None,
    },
}

# Track currently loaded model to prevent simultaneous loading
_current_model: Optional[str] = None


def _get_device() -> str:
    """Determine best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(variant: str) -> tuple:
    """
    Load a single Tiny Aya variant using HuggingFace transformers.

    Args:
        variant: One of 'base', 'fire', 'earth'.

    Returns:
        Tuple of (model, tokenizer) ready for activation extraction.

    Raises:
        ValueError: If variant is not recognized.
        RuntimeError: If another model is already loaded.
    """
    global _current_model

    if variant not in VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: {list(VARIANT_CONFIGS.keys())}"
        )

    if _current_model is not None:
        raise RuntimeError(
            f"Model '{_current_model}' is already loaded. "
            f"Call unload_model() before loading '{variant}'."
        )

    config = VARIANT_CONFIGS[variant]
    hf_name = config["hf_name"]
    device = _get_device()

    logger.info(f"Loading variant '{variant}' from {hf_name} on {device}...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        dtype=torch.float16,
        device_map=device if device == "cuda" else None,
    )

    if device != "cuda":
        model = model.to(device)

    model.eval()

    # Auto-detect architecture parameters
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    # Update config with detected values
    VARIANT_CONFIGS[variant]["n_layers"] = n_layers
    VARIANT_CONFIGS[variant]["d_model"] = d_model

    logger.info(
        f"Loaded '{variant}': n_layers={n_layers}, d_model={d_model}, "
        f"device={device}, dtype=float16"
    )

    _current_model = variant
    return model, tokenizer


def unload_model(model) -> None:
    """
    Explicitly unload a model and free memory.

    Args:
        model: The model to unload.
    """
    global _current_model

    variant_name = _current_model or "unknown"
    logger.info(f"Unloading model '{variant_name}'...")

    try:
        if hasattr(model, "to"):
            model.to("cpu")
    except Exception:
        pass

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass

    _current_model = None
    logger.info(f"Model '{variant_name}' unloaded, memory freed.")


def extract_hidden_states(
    model, tokenizer, text: str
) -> torch.Tensor:
    """
    Run a forward pass and return hidden states at all layers.

    Uses output_hidden_states=True to get residual stream activations
    at every layer without hooks.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        text: Input text to process.

    Returns:
        Tensor of shape (n_layers+1, d_model) -- last token activation at each layer.
        Index 0 = embedding output, index i = output of transformer layer i.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is a tuple of (n_layers+1) tensors, each (batch, seq_len, hidden_size)
    # Index 0 = embedding output, Index 1..n = transformer layer outputs
    hidden_states = outputs.hidden_states

    # Extract last token position from each layer
    last_token_states = torch.stack(
        [hs[0, -1, :] for hs in hidden_states]
    )  # (n_layers+1, d_model)

    return last_token_states


def get_activation_shape(variant: str) -> tuple[int, int]:
    """
    Return expected activation shape (n_layers + 1, d_model) for a variant.

    Args:
        variant: One of 'base', 'fire', 'earth'.

    Returns:
        Tuple of (n_layers + 1, d_model).

    Raises:
        ValueError: If architecture params haven't been detected yet.
    """
    config = VARIANT_CONFIGS[variant]
    n_layers = config.get("n_layers")
    d_model = config.get("d_model")

    if n_layers is None or d_model is None:
        raise ValueError(
            f"Architecture params for '{variant}' not yet detected. "
            f"Load the model first to auto-detect n_layers and d_model."
        )

    return (n_layers + 1, d_model)


if __name__ == "__main__":
    """Quick smoke test: print variant configs."""
    import sys

    logging.basicConfig(level=logging.INFO)

    print("Variant configurations:")
    for name, cfg in VARIANT_CONFIGS.items():
        print(f"  {name}: {cfg['hf_name']}")

    if "--load" in sys.argv:
        variant = sys.argv[sys.argv.index("--load") + 1]
        print(f"\nAttempting to load '{variant}'...")
        model, tokenizer = load_model(variant)
        print(f"Success! n_layers={model.config.num_hidden_layers}, d_model={model.config.hidden_size}")
        shape = get_activation_shape(variant)
        print(f"Expected activation shape: {shape}")

        # Quick forward pass test
        test_text = "The child has a fever."
        states = extract_hidden_states(model, tokenizer, test_text)
        print(f"Hidden states shape: {states.shape}")

        unload_model(model)
        print("Model unloaded.")
