"""
Generate annotated alignment curve for Hindi with RISE / PEAK / COLLAPSE labels.

Designed for X article and LinkedIn article hero visual — makes the
rise-peak-collapse pattern name immediately visual without reading text.

Usage:
    uv run viz/annotated_curve.py          # from real data
    uv run viz/annotated_curve.py --demo   # from synthetic data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402

from alignment_curves import (  # noqa: E402
    VARIANTS,
    VARIANT_LABELS,
    VARIANT_COLORS,
    CURVES_PATH,
    load_real_curves,
    generate_demo_curves,
    get_average_curve,
)

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"

DPI = 144
FIGSIZE = (1200 / DPI, 750 / DPI)


def render_annotated_hindi(curves: dict, demo: bool = False) -> Path:
    """
    Render Hindi average alignment curve with RISE / PEAK / COLLAPSE annotations.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Plot all three variants
    curves_by_variant: dict[str, list[float]] = {}
    for variant in VARIANTS:
        curve = get_average_curve(curves, variant, "hi")
        if curve is not None:
            curves_by_variant[variant] = curve

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
            linewidth=2.5,
            alpha=0.9,
        )

    n_layers = len(curves_by_variant.get("fire", curves_by_variant.get("base", [])))

    # ── Get actual curve values for smart annotation placement ───────────
    fire_curve = curves_by_variant.get("fire")
    base_curve = curves_by_variant.get("base")
    earth_curve = curves_by_variant.get("earth")

    # Find actual peak layer and values for Fire
    if fire_curve:
        peak_layer = int(np.argmax(fire_curve))
        fire_peak_val = fire_curve[peak_layer]
    else:
        peak_layer = 20
        fire_peak_val = 0.89

    base_at_peak = base_curve[peak_layer] if base_curve and peak_layer < len(base_curve) else 0.71
    collapse_end = n_layers - 1 if n_layers > 0 else 36

    # ── Phase region shading ──────────────────────────────────────────────
    ax.axvspan(2, peak_layer, alpha=0.05, color="#2ecc71", zorder=0)
    ax.axvspan(peak_layer, peak_layer + 3, alpha=0.07, color="#f39c12", zorder=0)
    ax.axvspan(30, collapse_end, alpha=0.05, color="#e74c3c", zorder=0)

    # ── RISE annotation (above curves, no overlap) ────────────────────────
    ax.annotate(
        "RISE",
        xy=(10, 1.02),
        fontsize=15,
        fontweight="bold",
        color="#27ae60",
        ha="center",
        va="bottom",
    )
    ax.annotate(
        "building shared representations",
        xy=(10, 1.00),
        fontsize=7,
        color="#27ae60",
        ha="center",
        va="top",
        style="italic",
    )

    # ── PEAK annotation (above curves) ────────────────────────────────────
    ax.annotate(
        "PEAK",
        xy=(peak_layer + 1, 1.02),
        fontsize=15,
        fontweight="bold",
        color="#e67e22",
        ha="center",
        va="bottom",
    )
    ax.annotate(
        f"L{peak_layer}",
        xy=(peak_layer + 1, 1.00),
        fontsize=7,
        color="#e67e22",
        ha="center",
        va="top",
        style="italic",
    )
    # Arrow pointing down to the actual peak on Fire curve
    ax.annotate(
        "",
        xy=(peak_layer, fire_peak_val + 0.01),
        xytext=(peak_layer + 1, 0.98),
        arrowprops=dict(
            arrowstyle="->",
            color="#e67e22",
            lw=1.5,
            alpha=0.6,
            connectionstyle="arc3,rad=0.2",
        ),
    )

    # ── COLLAPSE annotation (in the empty space below the falling curves) ─
    ax.annotate(
        "COLLAPSE",
        xy=(33.5, 0.18),
        fontsize=15,
        fontweight="bold",
        color="#c0392b",
        ha="center",
        va="bottom",
    )
    ax.annotate(
        "language-specific output",
        xy=(33.5, 0.16),
        fontsize=7,
        color="#c0392b",
        ha="center",
        va="top",
        style="italic",
    )

    # ── +15% gap bracket at peak layer ────────────────────────────────────
    if fire_curve and base_curve:
        # Place bracket right at peak where the gap is the reported +15%
        bracket_x = peak_layer
        fire_at_bracket = fire_curve[bracket_x] if bracket_x < len(fire_curve) else fire_peak_val
        base_at_bracket = base_curve[bracket_x] if bracket_x < len(base_curve) else base_at_peak

        # Offset bracket slightly right of the peak point so it doesn't overlap the line
        bx_display = bracket_x + 1.5
        ax.plot(
            [bx_display, bx_display],
            [base_at_bracket, fire_at_bracket],
            color="#555555",
            lw=1.3,
            alpha=0.8,
        )
        # Small horizontal ticks at ends
        tick_w = 0.5
        ax.plot([bx_display - tick_w, bx_display + tick_w], [fire_at_bracket, fire_at_bracket],
                color="#555555", lw=1.3, alpha=0.8)
        ax.plot([bx_display - tick_w, bx_display + tick_w], [base_at_bracket, base_at_bracket],
                color="#555555", lw=1.3, alpha=0.8)

        ax.annotate(
            "+15%",
            xy=(bx_display + 1, (fire_at_bracket + base_at_bracket) / 2),
            fontsize=10,
            fontweight="bold",
            color="#555555",
            ha="left",
            va="center",
        )

    # ── Standard plot elements ────────────────────────────────────────────
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Cosine Similarity (en → hi)", fontsize=10)
    ax.set_ylim(-0.05, 1.12)
    ax.set_title(
        "Cross-Lingual Alignment: Hindi (avg. 20 medical concepts)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.2, linestyle="--")

    if demo:
        fig.text(
            0.98, 0.02, "DEMO DATA",
            ha="right", va="bottom", fontsize=7,
            color="#AAAAAA", style="italic",
        )

    fig.patch.set_facecolor("white")
    fig.tight_layout()

    output_path = ASSETS_DIR / "annotated_rise_peak_collapse.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=DPI,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate annotated rise-peak-collapse alignment curve.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic sample data instead of real results.",
    )
    args = parser.parse_args()

    if args.demo:
        print("Generating annotated curve with DEMO data...")
        curves = generate_demo_curves()
    else:
        if not CURVES_PATH.exists():
            print(f"ERROR: {CURVES_PATH} not found.")
            print("Run concept_alignment.py first, or use --demo for sample data.")
            sys.exit(1)
        print("Generating annotated curve from real data...")
        curves = load_real_curves()

    output = render_annotated_hindi(curves, demo=args.demo)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
