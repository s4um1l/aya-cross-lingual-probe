"""
Secondary visualization: alignment curve line plots.

Shows how cosine similarity evolves across layers for specific concepts.
Compares Base vs Fire vs Earth for the same concept-language pair.

Generates 3 plots:
  1. C01 (fever) in Hindi: Base vs Fire vs Earth
  2. C18 (malaria) in Swahili: Base vs Fire vs Earth
  3. Average across all concepts in Hindi: Base vs Fire vs Earth

Usage:
    uv run viz/alignment_curves.py          # generate from real data
    uv run viz/alignment_curves.py --demo   # generate with synthetic data
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

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CURVES_PATH = RESULTS_DIR / "alignment_curves.json"
ASSETS_DIR = ROOT / "assets"

# ── Constants ─────────────────────────────────────────────────────────────
DPI = 144
FIGSIZE = (1200 / DPI, 675 / DPI)

VARIANTS = ["base", "fire", "earth"]
VARIANT_LABELS = {"base": "Base", "fire": "Fire", "earth": "Earth"}
VARIANT_COLORS = {"base": "#4C72B0", "fire": "#DD8452", "earth": "#55A868"}

N_LAYERS_DEFAULT = 33  # 32 transformer layers + embedding

CONCEPT_IDS = [f"C{i:02d}" for i in range(1, 21)]

# Plot specifications: (concept_id, language_code, output_filename, title)
PLOT_SPECS = [
    (
        "C01",
        "hi",
        "alignment_curve_fever_hi.png",
        "Alignment Curve: fever (C01) in Hindi",
    ),
    (
        "C18",
        "sw",
        "alignment_curve_malaria_sw.png",
        "Alignment Curve: malaria (C18) in Swahili",
    ),
    (
        None,
        "hi",
        "alignment_curve_avg_hi.png",
        "Alignment Curve: Average across all concepts in Hindi",
    ),
]


# ── Data loading ──────────────────────────────────────────────────────────


def load_real_curves() -> dict:
    """Load alignment_curves.json and return the curves dict."""
    with open(CURVES_PATH) as f:
        data = json.load(f)
    return data["curves"]


def _demo_curve(
    rng: np.random.Generator,
    n_layers: int,
    commit_layer: int,
    final_sim: float,
) -> list[float]:
    """
    Generate a plausible alignment curve that rises from low similarity
    to a plateau near final_sim, with the main rise around commit_layer.
    """
    x = np.arange(n_layers, dtype=np.float64)

    # Sigmoid-shaped curve centered at commit_layer
    steepness = rng.uniform(0.25, 0.45)
    base_curve = final_sim / (1.0 + np.exp(-steepness * (x - commit_layer)))

    # Add small noise
    noise = rng.normal(0, 0.015, size=n_layers)
    curve = np.clip(base_curve + noise, 0.0, 1.0)

    return [round(float(v), 6) for v in curve]


def generate_demo_curves() -> dict:
    """
    Generate synthetic alignment curves that show plausible patterns.

    Base commits later (~layer 22), Fire earlier for Hindi (~layer 14),
    Earth earlier for Swahili (~layer 14).
    """
    rng = np.random.default_rng(42)
    n_layers = N_LAYERS_DEFAULT

    curves: dict = {}

    for variant in VARIANTS:
        variant_data: dict = {}

        for concept_id in CONCEPT_IDS:
            concept_curves: dict = {}

            for lang in ["hi", "bn", "sw", "am", "fr", "es", "ar", "yo", "ta"]:
                # Determine commit layer based on variant + language
                if variant == "fire" and lang in ("hi", "bn", "ta"):
                    commit = int(rng.uniform(12, 17))
                    final = rng.uniform(0.88, 0.95)
                elif variant == "earth" and lang in ("sw", "am", "yo"):
                    commit = int(rng.uniform(12, 17))
                    final = rng.uniform(0.88, 0.95)
                elif lang in ("fr", "es"):
                    commit = int(rng.uniform(15, 20))
                    final = rng.uniform(0.90, 0.96)
                else:
                    commit = int(rng.uniform(19, 25))
                    final = rng.uniform(0.82, 0.92)

                concept_curves[lang] = _demo_curve(rng, n_layers, commit, final)

            variant_data[concept_id] = concept_curves

        curves[variant] = variant_data

    return curves


def get_single_curve(
    curves: dict, variant: str, concept_id: str, lang: str
) -> list[float] | None:
    """Extract a single curve from the nested dict."""
    return curves.get(variant, {}).get(concept_id, {}).get(lang)


def get_average_curve(
    curves: dict, variant: str, lang: str
) -> list[float] | None:
    """Compute average alignment curve across all concepts for a language."""
    all_curves: list[np.ndarray] = []

    variant_data = curves.get(variant, {})
    for concept_id in CONCEPT_IDS:
        concept_data = variant_data.get(concept_id, {})
        lang_curve = concept_data.get(lang)
        if lang_curve:
            all_curves.append(np.array(lang_curve))

    if not all_curves:
        return None

    avg = np.mean(all_curves, axis=0)
    return [round(float(v), 6) for v in avg]


# ── Plotting ──────────────────────────────────────────────────────────────


def render_curve_plot(
    curves_by_variant: dict[str, list[float]],
    title: str,
    output_path: Path,
    demo: bool = False,
) -> None:
    """
    Render a line plot comparing Base vs Fire vs Earth alignment curves.

    Args:
        curves_by_variant: {variant_name: [sim_l0, sim_l1, ...]}
        title: Plot title.
        output_path: Where to save the PNG.
        demo: If True, add a small "DEMO DATA" note.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    for variant in VARIANTS:
        curve = curves_by_variant.get(variant)
        if curve is None:
            continue

        layers = list(range(len(curve)))
        ax.plot(
            layers,
            curve,
            label=VARIANT_LABELS[variant],
            color=VARIANT_COLORS[variant],
            linewidth=2,
            alpha=0.9,
        )

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Cosine Similarity (en vs target)", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add threshold reference line
    ax.axhline(y=0.85, color="#999999", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(
        0.5,
        0.86,
        "commitment threshold (0.85)",
        fontsize=7,
        color="#999999",
        ha="left",
    )

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
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=DPI,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)


def generate_all_plots(curves: dict, demo: bool = False) -> list[Path]:
    """
    Generate all 3 alignment curve plots.

    Returns:
        List of output paths.
    """
    outputs: list[Path] = []

    for concept_id, lang, filename, title in PLOT_SPECS:
        output_path = ASSETS_DIR / filename

        curves_by_variant: dict[str, list[float]] = {}

        for variant in VARIANTS:
            if concept_id is not None:
                # Single concept curve
                curve = get_single_curve(curves, variant, concept_id, lang)
            else:
                # Average across all concepts
                curve = get_average_curve(curves, variant, lang)

            if curve is not None:
                curves_by_variant[variant] = curve

        if not curves_by_variant:
            print(f"WARNING: No data for {filename}. Skipping.")
            continue

        render_curve_plot(curves_by_variant, title, output_path, demo=demo)
        outputs.append(output_path)
        print(f"Saved: {output_path}")

    return outputs


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate alignment curve line plots.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic sample data instead of real results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.demo:
        print("Generating alignment curves with DEMO data...")
        curves = generate_demo_curves()
    else:
        if not CURVES_PATH.exists():
            print(f"ERROR: {CURVES_PATH} not found.")
            print("Run concept_alignment.py first, or use --demo for sample data.")
            sys.exit(1)
        print("Generating alignment curves from real data...")
        curves = load_real_curves()

    outputs = generate_all_plots(curves, demo=args.demo)
    print(f"\nGenerated {len(outputs)} plots.")


if __name__ == "__main__":
    main()
