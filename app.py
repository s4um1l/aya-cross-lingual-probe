"""
Interactive Gradio demo: Tiny Aya Cross-Lingual Concept Representation Explorer.

User types a sentence, selects a model variant, and sees layer-by-layer language
probability evolution as a bar chart.  A "Play animation" button auto-advances
the layer slider from 0 to 32.

Two operating modes:
    1. Live mode   — loads a model and computes activations in real-time
                     (for HF Space deployment with GPU).
    2. Demo mode   — uses pre-computed sample data so it works without GPU/model.
                     Activate with the --demo flag.  Default for local testing.

Usage:
    uv run app.py --demo       # demo mode (no GPU required)
    uv run app.py              # live mode  (requires model weights + GPU)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go

# ── Project paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# ── Language metadata ──────────────────────────────────────────────────────
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "sw": "Swahili",
    "am": "Amharic",
    "fr": "French",
    "es": "Spanish",
    "ar": "Arabic",
    "yo": "Yoruba",
    "ta": "Tamil",
}

LANGUAGES = list(LANGUAGE_NAMES.keys())
VARIANTS = ["Base", "Fire", "Earth"]
NUM_LAYERS = 33  # layers 0-32

# ── Pre-loaded examples ───────────────────────────────────────────────────
EXAMPLES = [
    ["The child has a fever.", "Base"],
    ["\u092c\u091a\u094d\u091a\u0947 \u0915\u094b \u092c\u0941\u0916\u093e\u0930 \u0939\u0948\u0964", "Fire"],
    ["Mtoto ana homa.", "Earth"],
]


# ── Demo data generation ──────────────────────────────────────────────────

def _sigmoid(x: float, center: float, steepness: float) -> float:
    """Numerically safe sigmoid."""
    z = steepness * (x - center)
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _generate_demo_probs(
    text: str,
    variant: str,
) -> np.ndarray:
    """
    Generate plausible layer-by-layer language probability data for demo mode.

    Returns array of shape (NUM_LAYERS, len(LANGUAGES)) where each row sums
    to 1.0 and shows a realistic shift: early layers are uncertain / biased
    toward English, mid layers show the true language emerging, and final
    layers strongly favour the correct language.

    The variant affects the 'commitment speed':
        - Fire commits to South Asian languages earlier (Hindi, Bengali, Tamil)
        - Earth commits to African languages earlier (Swahili, Amharic, Yoruba)
        - Base is the median
    """
    # Simple heuristic to guess the dominant language from the text
    dominant = _guess_language(text)

    # Commitment layer varies by variant and language family
    south_asian = {"hi", "bn", "ta"}
    african = {"sw", "am", "yo"}

    base_commit = 18
    if dominant in south_asian:
        if variant == "Fire":
            commit_layer = 12
        elif variant == "Earth":
            commit_layer = 20
        else:
            commit_layer = base_commit
    elif dominant in african:
        if variant == "Earth":
            commit_layer = 12
        elif variant == "Fire":
            commit_layer = 20
        else:
            commit_layer = base_commit
    else:
        commit_layer = base_commit - 2  # en/fr/es/ar converge quickly for all

    probs = np.zeros((NUM_LAYERS, len(LANGUAGES)), dtype=np.float64)
    dominant_idx = LANGUAGES.index(dominant) if dominant in LANGUAGES else 0

    # Seed for consistency per input
    rng = np.random.RandomState(abs(hash(text + variant)) % (2**31))

    for layer in range(NUM_LAYERS):
        # Dominant language probability follows a sigmoid
        p_dom = 0.10 + 0.85 * _sigmoid(layer, commit_layer, 0.45)

        # Distribute remaining probability among other languages
        remaining = 1.0 - p_dom
        noise = rng.dirichlet(np.ones(len(LANGUAGES) - 1) * 0.5)
        row = np.zeros(len(LANGUAGES))
        row[dominant_idx] = p_dom

        other_idxs = [i for i in range(len(LANGUAGES)) if i != dominant_idx]
        for j, oi in enumerate(other_idxs):
            row[oi] = remaining * noise[j]

        # Early layers: English gets a boost (model's default)
        if layer < commit_layer // 2 and dominant != "en":
            en_idx = LANGUAGES.index("en")
            boost = 0.25 * (1.0 - layer / (commit_layer / 2))
            row[en_idx] += boost
            row /= row.sum()

        probs[layer] = row / row.sum()

    return probs


def _guess_language(text: str) -> str:
    """Very rough heuristic language detection for demo purposes."""
    # Check for specific scripts
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return "hi"
        if 0x0980 <= cp <= 0x09FF:
            return "bn"
        if 0x0B80 <= cp <= 0x0BFF:
            return "ta"
        if 0x1200 <= cp <= 0x137F:
            return "am"
        if 0x0600 <= cp <= 0x06FF:
            return "ar"

    text_lower = text.lower()

    # Swahili markers
    sw_markers = ["mtoto", "mgonjwa", "ana ", "homa", "daktari", "hospitali", "maji"]
    if any(m in text_lower for m in sw_markers):
        return "sw"

    # Yoruba markers (tone marks)
    yo_markers = ["\u1ecd", "\u1eb9", "\u0301", "\u0300", "\u0304"]
    if any(m in text for m in yo_markers):
        return "yo"

    # French markers
    fr_markers = ["l'enfant", "le patient", "la m\u00e8re", "le m\u00e9decin",
                  "fièvre", "hôpital"]
    if any(m in text_lower for m in fr_markers):
        return "fr"

    # Spanish markers
    es_markers = ["el niño", "el paciente", "la madre", "enfermo", "fiebre",
                  "hospital"]
    if any(m in text_lower for m in es_markers):
        return "es"

    return "en"


# ── Live mode (model loading) ─────────────────────────────────────────────

_live_model = None
_live_tokenizer = None
_live_variant = None


def _load_model_live(variant: str):
    """Load a model for live inference."""
    global _live_model, _live_tokenizer, _live_variant

    if _live_variant == variant and _live_model is not None:
        return

    # Unload previous model
    if _live_model is not None:
        import gc
        import torch
        del _live_model
        _live_model = None
        _live_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    from model_loader import load_model, VARIANT_CONFIGS  # noqa: F401

    print(f"Loading model variant: {variant}")
    _live_model = load_model(variant.lower())
    _live_tokenizer = _live_model.tokenizer
    _live_variant = variant


def _compute_live_probs(text: str, variant: str) -> np.ndarray:
    """
    Run a forward pass through the model and extract residual stream
    activations, then project onto the unembedding matrix to get
    per-layer language probabilities.
    """
    import torch

    _load_model_live(variant)
    model = _live_model

    # Tokenize
    tokens = model.to_tokens(text, prepend_bos=True)

    # Run with cache to get residual stream at every layer
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: "resid_post" in n)

    # Collect residual stream activations: take the mean across token positions
    n_layers = model.cfg.n_layers
    resid_layers = []
    for layer_idx in range(n_layers):
        key = f"blocks.{layer_idx}.hook_resid_post"
        # shape: (batch, seq_len, d_model) -> mean over seq -> (d_model,)
        resid = cache[key][0].mean(dim=0)
        resid_layers.append(resid)

    # Also add embed layer (layer 0)
    embed_key = "hook_embed"
    if embed_key in cache:
        embed_resid = cache[embed_key][0].mean(dim=0)
        resid_layers.insert(0, embed_resid)

    # Project each layer's residual onto the unembedding to get logits
    unembed = model.W_U  # (d_model, d_vocab)

    # For each layer, compute logits and extract language-representative tokens
    probs = np.zeros((len(resid_layers), len(LANGUAGES)), dtype=np.float64)

    # Use representative tokens for each language
    lang_tokens = _get_language_tokens(model)

    for layer_idx, resid in enumerate(resid_layers):
        logits = resid @ unembed  # (d_vocab,)
        # Softmax over language tokens
        lang_logits = torch.stack([logits[tid].mean() for tid in lang_tokens])
        lang_probs = torch.softmax(lang_logits, dim=0).detach().cpu().numpy()
        probs[layer_idx] = lang_probs

    return probs


def _get_language_tokens(model) -> list:
    """
    Get representative token IDs for each language.

    Uses common words in each language as proxies.
    """
    import torch

    # Representative words per language
    lang_words = {
        "en": ["the", "child", "has"],
        "hi": ["\u092c\u091a\u094d\u091a\u093e", "\u0939\u0948"],
        "bn": ["\u09b6\u09bf\u09b6\u09c1", "\u0986\u099b\u09c7"],
        "sw": ["mtoto", "ana"],
        "am": ["\u1205\u133b\u1291", "\u12a0\u1208"],
        "fr": ["enfant", "le"],
        "es": ["niño", "el"],
        "ar": ["\u0627\u0644\u0637\u0641\u0644", "\u0645\u0646"],
        "yo": ["\u1ecd\u1e41\u1ecd", "ni"],
        "ta": ["\u0b95\u0bc1\u0bb4\u0ba8\u0bcd\u0ba4\u0bc8", "\u0b87\u0bb0\u0bc1"],
    }

    tokenizer = model.tokenizer
    all_token_ids = []
    for lang in LANGUAGES:
        words = lang_words.get(lang, ["the"])
        token_ids = []
        for word in words:
            ids = tokenizer.encode(word, add_special_tokens=False)
            token_ids.extend(ids)
        if not token_ids:
            token_ids = [0]
        all_token_ids.append(torch.tensor(token_ids, device=model.cfg.device))

    return all_token_ids


# ── Unified probability function ──────────────────────────────────────────

def get_layer_probs(
    text: str,
    variant: str,
    demo_mode: bool = True,
) -> np.ndarray:
    """
    Get language probabilities for all layers.

    Args:
        text: Input sentence.
        variant: Model variant name.
        demo_mode: If True, use synthetic demo data.

    Returns:
        Array of shape (NUM_LAYERS, len(LANGUAGES)).
    """
    if demo_mode:
        return _generate_demo_probs(text, variant)
    else:
        return _compute_live_probs(text, variant)


# ── Visualization ─────────────────────────────────────────────────────────

def make_bar_chart(probs_row: np.ndarray, layer: int) -> go.Figure:
    """
    Create a Plotly bar chart showing top-5 language probabilities at a
    given layer.
    """
    # Get top-5 by probability
    sorted_idxs = np.argsort(probs_row)[::-1][:5]
    labels = [LANGUAGE_NAMES[LANGUAGES[i]] for i in sorted_idxs]
    values = [float(probs_row[i]) for i in sorted_idxs]

    colors = [
        "#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6",
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors[: len(labels)],
                text=[f"{v:.1%}" for v in values],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"Top-5 Language Probabilities at Layer {layer}",
            x=0.5,
            font=dict(size=16),
        ),
        yaxis=dict(
            title="Probability",
            range=[0, 1.05],
            tickformat=".0%",
        ),
        xaxis=dict(title="Language"),
        height=400,
        margin=dict(t=60, b=60, l=60, r=30),
        plot_bgcolor="white",
        font=dict(size=13),
    )

    return fig


# ── Gradio app builder ────────────────────────────────────────────────────

def build_app(demo_mode: bool = True) -> gr.Blocks:
    """
    Construct the Gradio Blocks app.

    Args:
        demo_mode: If True, uses pre-computed sample data.
    """
    # Cache for the current session's probabilities
    _cache: dict[str, np.ndarray] = {}

    def _cache_key(text: str, variant: str) -> str:
        return f"{text}|||{variant}"

    def compute_and_show(text: str, variant: str, layer: int):
        """Compute probabilities (or use cache) and return bar chart."""
        if not text.strip():
            return make_bar_chart(np.ones(len(LANGUAGES)) / len(LANGUAGES), layer)

        key = _cache_key(text, variant)
        if key not in _cache:
            _cache[key] = get_layer_probs(text, variant, demo_mode=demo_mode)

        probs = _cache[key]
        layer = min(layer, probs.shape[0] - 1)
        return make_bar_chart(probs[layer], layer)

    def play_animation(text: str, variant: str):
        """Generator that yields frames for layers 0 through 32."""
        if not text.strip():
            for layer in range(NUM_LAYERS):
                yield layer, make_bar_chart(
                    np.ones(len(LANGUAGES)) / len(LANGUAGES), layer
                )
            return

        key = _cache_key(text, variant)
        if key not in _cache:
            _cache[key] = get_layer_probs(text, variant, demo_mode=demo_mode)

        probs = _cache[key]
        for layer in range(min(NUM_LAYERS, probs.shape[0])):
            fig = make_bar_chart(probs[layer], layer)
            yield layer, fig
            time.sleep(0.15)

    mode_label = "DEMO MODE (pre-computed data)" if demo_mode else "LIVE MODE (model inference)"

    with gr.Blocks(
        title="Inside Tiny Aya: Cross-Lingual Concept Explorer",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            f"""
            # Inside Tiny Aya: Cross-Lingual Concept Explorer

            Explore how Tiny Aya model variants process multilingual input across
            transformer layers.  Watch language probabilities evolve from the
            embedding layer (0) through to the final layer (32).

            **Mode:** {mode_label}
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Input Sentence",
                    placeholder="Type a sentence in any of the 10 supported languages...",
                    lines=2,
                )
                variant_dropdown = gr.Dropdown(
                    choices=VARIANTS,
                    value="Base",
                    label="Model Variant",
                )
            with gr.Column(scale=1):
                layer_slider = gr.Slider(
                    minimum=0,
                    maximum=32,
                    step=1,
                    value=0,
                    label="Layer",
                )
                play_btn = gr.Button("Play Animation (0 -> 32)", variant="primary")

        chart_output = gr.Plot(label="Language Probabilities")

        # Pre-loaded examples
        gr.Examples(
            examples=EXAMPLES,
            inputs=[text_input, variant_dropdown],
            label="Pre-loaded Examples",
        )

        gr.Markdown(
            """
            ---
            **Supported languages:** English, Hindi, Bengali, Swahili, Amharic,
            French, Spanish, Arabic, Yoruba, Tamil

            **What to look for:**
            - Early layers (0-8): probabilities are diffuse, often biased toward English
            - Mid layers (8-20): the model begins committing to the input language
            - Final layers (24-32): strong commitment to the correct language
            - Fire variant commits earlier for South Asian languages (Hindi, Bengali, Tamil)
            - Earth variant commits earlier for African languages (Swahili, Amharic, Yoruba)
            """
        )

        # Event handlers
        text_input.change(
            fn=compute_and_show,
            inputs=[text_input, variant_dropdown, layer_slider],
            outputs=chart_output,
        )
        variant_dropdown.change(
            fn=compute_and_show,
            inputs=[text_input, variant_dropdown, layer_slider],
            outputs=chart_output,
        )
        layer_slider.change(
            fn=compute_and_show,
            inputs=[text_input, variant_dropdown, layer_slider],
            outputs=chart_output,
        )
        play_btn.click(
            fn=play_animation,
            inputs=[text_input, variant_dropdown],
            outputs=[layer_slider, chart_output],
        )

    return app


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tiny Aya Cross-Lingual Concept Explorer — Gradio app"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run in demo mode with pre-computed sample data (no GPU required).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to serve the app on (default: 7860).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a public share link.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    demo_mode = args.demo
    if not demo_mode:
        # Auto-fallback: if transformers/torch not available, use demo mode
        try:
            import torch  # noqa: F401
            import transformer_lens  # noqa: F401
        except ImportError:
            print(
                "WARNING: torch or transformer_lens not available. "
                "Falling back to demo mode."
            )
            demo_mode = True

    app = build_app(demo_mode=demo_mode)
    app.launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
