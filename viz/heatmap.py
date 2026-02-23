"""
Primary visualization: concept alignment heatmap.

X-axis: 3 model variants (Base, Fire, Earth)
Y-axis: 10 languages (full names, not codes)
Color: Average commitment layer (early = warmer color) OR average peak
       cosine similarity — uses commitment layer by default.
Size: exactly 1200x675px at 144dpi

Usage:
    uv run viz/heatmap.py          # generate from real data
    uv run viz/heatmap.py --demo   # generate with synthetic sample data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CURVES_PATH = RESULTS_DIR / "alignment_curves.json"
ASSETS_DIR = ROOT / "assets"
OUTPUT_PATH = ASSETS_DIR / "concept_alignment_heatmap.png"

# ── Constants ─────────────────────────────────────────────────────────────
DPI = 144
FIGSIZE = (1200 / DPI, 675 / DPI)

VARIANTS = ["base", "fire", "earth"]
VARIANT_LABELS = ["Base", "Fire", "Earth"]

LANG_CODES = ["hi", "bn", "sw", "am", "fr", "es", "ar", "yo", "ta"]
LANG_NAMES = {
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
LANG_LABELS = [LANG_NAMES[lc] for lc in LANG_CODES]

CONCEPT_IDS = [f"C{i:02d}" for i in range(1, 21)]

DEFAULT_THRESHOLD = 0.85
DEFAULT_CONSECUTIVE = 3


# ── Data loading ──────────────────────────────────────────────────────────


def extract_commitment_layer(
    curve: list[float],
    threshold: float = DEFAULT_THRESHOLD,
    consecutive: int = DEFAULT_CONSECUTIVE,
) -> int:
    """First layer where cosine_sim >= threshold for N consecutive layers."""
    if len(curve) < consecutive:
        return -1

    run_start: int | None = None
    run_length = 0

    for idx, val in enumerate(curve):
        if val >= threshold:
            if run_start is None:
                run_start = idx
            run_length += 1
            if run_length >= consecutive:
                return run_start
        else:
            run_start = None
            run_length = 0

    return -1


def load_real_data() -> np.ndarray:
    """
    Load alignment_curves.json and compute average commitment layer
    per (language, variant).

    Returns:
        2-D array of shape (n_languages, n_variants) with avg commitment
        layers.  Cells where no concept committed are set to NaN.
    """
    with open(CURVES_PATH) as f:
        data = json.load(f)

    curves = data["curves"]
    matrix = np.full((len(LANG_CODES), len(VARIANTS)), np.nan)

    for v_idx, variant in enumerate(VARIANTS):
        variant_curves = curves.get(variant, {})

        for l_idx, lang in enumerate(LANG_CODES):
            commit_layers: list[int] = []

            for concept_id in CONCEPT_IDS:
                concept_data = variant_curves.get(concept_id, {})
                lang_curve = concept_data.get(lang, [])

                if not lang_curve:
                    continue

                cl = extract_commitment_layer(lang_curve)
                if cl >= 0:
                    commit_layers.append(cl)

            if commit_layers:
                matrix[l_idx, v_idx] = np.mean(commit_layers)

    return matrix


def generate_demo_data() -> np.ndarray:
    """
    Generate synthetic commitment-layer matrix for demo/testing.

    Simulates a plausible pattern where:
    - Fire commits earlier for South Asian languages (Hindi, Bengali, Tamil)
    - Earth commits earlier for African languages (Swahili, Amharic, Yoruba)
    - Base is generally later across the board
    """
    rng = np.random.default_rng(42)

    # Base: generally commits in layers 18-26
    base = rng.uniform(18, 26, size=len(LANG_CODES))

    # Fire: earlier for South Asian (12-18), similar to base elsewhere
    fire = base.copy()
    south_asian_idx = [LANG_CODES.index(lc) for lc in ["hi", "bn", "ta"]]
    for idx in south_asian_idx:
        fire[idx] = rng.uniform(12, 18)

    # Earth: earlier for African (12-18), similar to base elsewhere
    earth = base.copy()
    african_idx = [LANG_CODES.index(lc) for lc in ["sw", "am", "yo"]]
    for idx in african_idx:
        earth[idx] = rng.uniform(12, 18)

    # High-resource languages commit a bit earlier everywhere
    high_res_idx = [LANG_CODES.index(lc) for lc in ["fr", "es"]]
    for idx in high_res_idx:
        base[idx] -= rng.uniform(2, 5)
        fire[idx] -= rng.uniform(2, 5)
        earth[idx] -= rng.uniform(2, 5)

    matrix = np.column_stack([base, fire, earth])
    return np.round(matrix, 1)


# ── Plotting ──────────────────────────────────────────────────────────────


def render_heatmap(matrix: np.ndarray, output_path: Path, demo: bool = False) -> None:
    """
    Render the concept alignment heatmap and save to disk.

    Args:
        matrix: Shape (n_languages, n_variants) with avg commitment layers.
        output_path: Where to save the PNG.
        demo: If True, add a small "DEMO DATA" watermark.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Use plasma colormap (perceptually uniform), reversed so early = warm
    cmap = sns.color_palette("plasma_r", as_cmap=True)

    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        xticklabels=VARIANT_LABELS,
        yticklabels=LANG_LABELS,
        cbar_kws={"label": "Avg commitment layer (lower = earlier alignment)"},
        linewidths=0.5,
        linecolor="white",
        vmin=8,
        vmax=28,
    )

    # Title and subtitle
    ax.set_title(
        "Does Tiny Aya share concepts across languages?",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    fig.text(
        0.5,
        0.94,
        "Layer of peak cross-lingual alignment by variant and language",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    # Style
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    if demo:
        fig.text(
            0.98,
            0.02,
            "DEMO DATA",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#AAAAAA",
            style="italic",
        )

    fig.patch.set_facecolor("white")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=DPI,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate concept alignment heatmap.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic sample data instead of real results.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output path (default: assets/concept_alignment_heatmap.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else OUTPUT_PATH

    if args.demo:
        print("Generating heatmap with DEMO data...")
        matrix = generate_demo_data()
    else:
        if not CURVES_PATH.exists():
            print(f"ERROR: {CURVES_PATH} not found.")
            print("Run concept_alignment.py first, or use --demo for sample data.")
            sys.exit(1)
        print("Generating heatmap from real data...")
        matrix = load_real_data()

    render_heatmap(matrix, output_path, demo=args.demo)
    print(f"Saved: {output_path}")
    print(f"Size:  {1200}x{675}px at {DPI}dpi")


if __name__ == "__main__":
    main()
