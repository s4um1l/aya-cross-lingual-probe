"""
Modal app definition for GPU-accelerated batch activation extraction.

Uses HuggingFace transformers with output_hidden_states=True (not TransformerLens,
which does not support Cohere2 architecture).

- Image: Debian-slim with Python deps
- Volume: aya-activations for persisting .npy files across runs
- GPU: A10G (24GB VRAM -- fits 3.35B float16 model)
- Timeout: 30 min per function call

Usage:
    # Called by batch_runner.py -- not invoked directly
    from modal_config import extract_activations_remote
"""

from __future__ import annotations

import modal

# ── Modal App ─────────────────────────────────────────────────────────────
app = modal.App("aya-mech-interp")

# ── Image: Debian-slim + Python deps + CUDA ───────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.45,<5",
        "numpy>=1.26,<2",
        "tqdm>=4.66",
        "accelerate>=0.27",
    )
)

# ── Volume: persistent storage for activation .npy files ──────────────────
volume = modal.Volume.from_name("aya-activations", create_if_missing=True)
VOLUME_MOUNT = "/activations"

# ── HuggingFace token secret ──────────────────────────────────────────────
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])


@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret],
    timeout=30 * 60,
    retries=1,
)
def extract_activations_remote(
    variant: str,
    stimuli: list[dict],
    hf_name: str,
) -> dict:
    """
    Run activation extraction on Modal GPU for a batch of stimuli.

    Uses output_hidden_states=True to capture residual stream at all layers.
    Saves float16 .npy files to Modal Volume.

    Args:
        variant: Model variant name (base/fire/earth).
        stimuli: List of stimulus dicts with 'stimulus_id' and 'text' keys.
        hf_name: HuggingFace model name for this variant.

    Returns:
        Dict with 'processed', 'skipped', 'errors' counts and 'variant'.
    """
    import gc
    from pathlib import Path

    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    variant_dir = Path(VOLUME_MOUNT) / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    print(f"[Modal] Loading variant '{variant}' from {hf_name} on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"[Modal] Loaded: n_layers={n_layers}, d_model={d_model}")

    # ── Process stimuli ───────────────────────────────────────────────
    processed = 0
    skipped = 0
    errors = 0

    for stim in tqdm(stimuli, desc=f"[Modal] {variant}"):
        stim_id = stim["stimulus_id"]
        text = stim["text"]
        out_path = variant_dir / f"{stim_id}_resid.npy"

        if out_path.exists():
            skipped += 1
            continue

        try:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # hidden_states: tuple of (n_layers+1) tensors, each (batch, seq_len, hidden_size)
            hidden_states = outputs.hidden_states

            # Extract last token at each layer, stack to (n_layers+1, d_model)
            last_token_states = torch.stack(
                [hs[0, -1, :] for hs in hidden_states]
            )

            activation = last_token_states.cpu().float().half().numpy()
            np.save(str(out_path), activation)
            processed += 1

        except Exception as e:
            print(f"[Modal] Error processing '{stim_id}': {e}")
            errors += 1

    # ── Cleanup ───────────────────────────────────────────────────────
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Commit volume changes
    volume.commit()

    result = {
        "variant": variant,
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "total": len(stimuli),
    }
    print(f"[Modal] Done: {result}")
    return result
