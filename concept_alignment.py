"""
Computes cosine similarity of residual stream activations across languages
for each concept, at each layer, for each model variant.

Primary output: results/alignment_curves.json
Secondary output: results/commitment_matrix.csv (from commitment_layer extraction)

The alignment curve for a given (concept, language_pair, variant) is a vector
of cosine similarities at each layer between the English activation and the
target-language activation.  The "commitment layer" is the first layer where
similarity >= threshold for `consecutive` layers in a row.

Usage:
    uv run concept_alignment.py                     # full analysis, all variants
    uv run concept_alignment.py --variant base      # single variant
    uv run concept_alignment.py --threshold 0.85    # override threshold
    uv run concept_alignment.py --consecutive 3     # override consecutive count
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Project paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROBES_PATH = DATA_DIR / "concept_probes.json"
MANIFEST_PATH = DATA_DIR / "stimulus_manifest.json"
ACTIVATIONS_DIR = ROOT / "activations"
RESULTS_DIR = ROOT / "results"

# ── Constants ──────────────────────────────────────────────────────────────
ALL_VARIANTS = ["base", "fire", "earth"]
ALL_LANGUAGES = ["en", "hi", "bn", "sw", "am", "fr", "es", "ar", "yo", "ta"]
TARGET_LANGUAGES = [lang for lang in ALL_LANGUAGES if lang != "en"]
CONCEPT_IDS = [f"C{i:02d}" for i in range(1, 21)]

DEFAULT_THRESHOLD = 0.85
DEFAULT_CONSECUTIVE = 3


# ── Core functions ─────────────────────────────────────────────────────────


def load_activation(variant: str, stimulus_id: str) -> np.ndarray:
    """
    Load a single activation file.

    Args:
        variant: Model variant name (base/fire/earth).
        stimulus_id: Stimulus identifier (e.g. 'probe_C01_en').

    Returns:
        Activation array of shape (n_layers+1, d_model).

    Raises:
        FileNotFoundError: If the activation file does not exist.
    """
    path = ACTIVATIONS_DIR / variant / f"{stimulus_id}_resid.npy"
    if not path.exists():
        raise FileNotFoundError(f"Activation file not found: {path}")

    data = np.load(str(path)).astype(np.float32)
    return data


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].  Returns 0.0 if either vector has
        near-zero norm.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_alignment_curves(variant: str) -> dict:
    """
    For each of 20 concepts, for each language pair (en, L), for each layer,
    compute the cosine similarity between the English activation and the
    target-language activation.

    Args:
        variant: Model variant name.

    Returns:
        Dict shaped: {concept_id: {lang: [sim_l0, sim_l1, ..., sim_ln]}}

    Raises:
        FileNotFoundError: If activation files are missing.
    """
    curves: dict[str, dict[str, list[float]]] = {}

    for concept_id in CONCEPT_IDS:
        en_stim_id = f"probe_{concept_id}_en"

        try:
            en_act = load_activation(variant, en_stim_id)
        except FileNotFoundError:
            logger.warning(
                f"Missing English activation for {concept_id} "
                f"(variant={variant}). Skipping concept."
            )
            continue

        n_layers_plus_1 = en_act.shape[0]
        concept_curves: dict[str, list[float]] = {}

        for lang in TARGET_LANGUAGES:
            lang_stim_id = f"probe_{concept_id}_{lang}"

            try:
                lang_act = load_activation(variant, lang_stim_id)
            except FileNotFoundError:
                logger.warning(
                    f"Missing activation for {concept_id}/{lang} "
                    f"(variant={variant}). Skipping."
                )
                continue

            # Compute per-layer cosine similarity
            sims: list[float] = []
            for layer_idx in range(n_layers_plus_1):
                sim = compute_cosine_similarity(
                    en_act[layer_idx], lang_act[layer_idx]
                )
                sims.append(round(sim, 6))

            concept_curves[lang] = sims

        if concept_curves:
            curves[concept_id] = concept_curves

    return curves


def extract_commitment_layer(
    curve: list[float],
    threshold: float = DEFAULT_THRESHOLD,
    consecutive: int = DEFAULT_CONSECUTIVE,
) -> int:
    """
    First layer l where cosine_sim >= threshold for `consecutive` layers
    in a row.

    This measures when the model 'locks in' to the correct language
    representation for that concept.

    Args:
        curve: List of cosine similarity values, one per layer.
        threshold: Minimum similarity value.
        consecutive: Number of consecutive layers above threshold.

    Returns:
        Layer index of the first qualifying layer, or -1 if never committed.
    """
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


def build_commitment_matrix(
    all_curves: dict[str, dict],
    threshold: float = DEFAULT_THRESHOLD,
    consecutive: int = DEFAULT_CONSECUTIVE,
) -> pd.DataFrame:
    """
    Build a DataFrame with columns:
        concept_id, concept, language, base_layer, fire_layer, earth_layer

    Each row represents one (concept, language) pair with the commitment
    layer for each variant.

    Args:
        all_curves: Dict shaped {variant: {concept_id: {lang: [sims]}}}.
        threshold: Commitment threshold.
        consecutive: Consecutive layers required.

    Returns:
        DataFrame with commitment layer data.
    """
    # Load concept names from probes
    concept_names = _load_concept_names()

    rows: list[dict] = []
    for concept_id in CONCEPT_IDS:
        concept_name = concept_names.get(concept_id, concept_id)

        for lang in TARGET_LANGUAGES:
            row: dict = {
                "concept_id": concept_id,
                "concept": concept_name,
                "language": lang,
            }

            for variant in ALL_VARIANTS:
                variant_curves = all_curves.get(variant, {})
                concept_curves = variant_curves.get(concept_id, {})
                lang_curve = concept_curves.get(lang, [])

                if lang_curve:
                    commit_layer = extract_commitment_layer(
                        lang_curve, threshold, consecutive
                    )
                else:
                    commit_layer = -1

                row[f"{variant}_layer"] = commit_layer

            rows.append(row)

    return pd.DataFrame(rows)


# ── Helpers ────────────────────────────────────────────────────────────────


def _load_concept_names() -> dict[str, str]:
    """Load concept names from concept_probes.json."""
    if not PROBES_PATH.exists():
        return {}

    with open(PROBES_PATH) as f:
        probes = json.load(f)

    return {cid: entry["concept"] for cid, entry in probes.items()}


def _detect_n_layers(variant: str) -> int | None:
    """
    Detect the number of layers from the first available activation file
    for a variant.

    Returns:
        Number of layers (n_layers+1 from file shape), or None if no files found.
    """
    variant_dir = ACTIVATIONS_DIR / variant
    if not variant_dir.exists():
        return None

    for npy_file in sorted(variant_dir.glob("probe_*_resid.npy"))[:1]:
        data = np.load(str(npy_file))
        return data.shape[0]

    return None


def _check_activations_exist() -> bool:
    """Check if any activation data exists."""
    if not ACTIVATIONS_DIR.exists():
        return False

    for variant in ALL_VARIANTS:
        variant_dir = ACTIVATIONS_DIR / variant
        if variant_dir.exists():
            npy_files = list(variant_dir.glob("probe_*_resid.npy"))
            if npy_files:
                return True

    return False


# ── Output writers ─────────────────────────────────────────────────────────


def write_alignment_curves(
    all_curves: dict[str, dict],
    n_layers: int,
    threshold: float,
    consecutive: int,
) -> Path:
    """
    Write alignment_curves.json with metadata.

    Args:
        all_curves: {variant: {concept_id: {lang: [sims]}}}.
        n_layers: Total layer count (n_layers+1).
        threshold: Threshold used for commitment.
        consecutive: Consecutive count used.

    Returns:
        Path to the written file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "variants": list(all_curves.keys()),
            "languages": TARGET_LANGUAGES,
            "concepts": len(CONCEPT_IDS),
            "layers": n_layers,
            "threshold_used": threshold,
            "consecutive_layers": consecutive,
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "curves": all_curves,
    }

    out_path = RESULTS_DIR / "alignment_curves.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Wrote alignment curves to {out_path}")
    return out_path


def write_commitment_matrix(df: pd.DataFrame) -> Path:
    """
    Write commitment_matrix.csv.

    Args:
        df: DataFrame with commitment layer data.

    Returns:
        Path to the written file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = RESULTS_DIR / "commitment_matrix.csv"
    df.to_csv(out_path, index=False)

    logger.info(f"Wrote commitment matrix ({len(df)} rows) to {out_path}")
    return out_path


# ── Main orchestrator ──────────────────────────────────────────────────────


def run_analysis(
    variants: list[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    consecutive: int = DEFAULT_CONSECUTIVE,
) -> None:
    """
    Run the full concept alignment analysis.

    Args:
        variants: List of variants to analyze (default: all).
        threshold: Commitment threshold.
        consecutive: Consecutive layers for commitment.
    """
    # Check for activation data
    if not _check_activations_exist():
        print("\n" + "=" * 60)
        print("ERROR: No activation data found.")
        print("=" * 60)
        print()
        print("Activation files are required for the alignment analysis.")
        print("Run the batch extraction first:")
        print()
        print("  uv run batch_runner.py --local     # local run")
        print("  uv run batch_runner.py             # Modal GPU run")
        print()
        print("After batch_runner.py completes, verify with:")
        print()
        print("  Check activations/ directory for missing files.")
        print()
        print("Then re-run this script.")
        print("=" * 60)
        sys.exit(1)

    if variants is None:
        variants = ALL_VARIANTS

    # Detect layer count from available data
    n_layers = None
    for variant in variants:
        n_layers = _detect_n_layers(variant)
        if n_layers is not None:
            break

    if n_layers is None:
        print("ERROR: Could not detect layer count from activation files.")
        sys.exit(1)

    print("=" * 60)
    print("CONCEPT ALIGNMENT ANALYSIS")
    print("=" * 60)
    print(f"Variants:     {variants}")
    print(f"Concepts:     {len(CONCEPT_IDS)}")
    print(f"Languages:    {TARGET_LANGUAGES}")
    print(f"Layers:       {n_layers}")
    print(f"Threshold:    {threshold}")
    print(f"Consecutive:  {consecutive}")
    print()

    # Compute alignment curves for each variant
    all_curves: dict[str, dict] = {}

    for variant in variants:
        variant_dir = ACTIVATIONS_DIR / variant
        if not variant_dir.exists():
            logger.warning(f"Variant directory missing: {variant_dir}. Skipping.")
            continue

        print(f"Computing alignment curves for '{variant}'...")
        curves = compute_alignment_curves(variant)
        all_curves[variant] = curves

        concept_count = len(curves)
        total_curves = sum(len(lang_curves) for lang_curves in curves.values())
        print(f"  {concept_count} concepts, {total_curves} language curves")

    if not all_curves:
        print("ERROR: No alignment curves computed. Check activation files.")
        sys.exit(1)

    # Write alignment curves JSON
    print()
    curves_path = write_alignment_curves(
        all_curves, n_layers, threshold, consecutive
    )
    print(f"Alignment curves: {curves_path}")

    # Build and write commitment matrix
    commitment_df = build_commitment_matrix(all_curves, threshold, consecutive)
    matrix_path = write_commitment_matrix(commitment_df)
    print(f"Commitment matrix: {matrix_path} ({len(commitment_df)} rows)")

    # Print summary statistics
    print()
    print("-" * 60)
    print("COMMITMENT LAYER SUMMARY")
    print("-" * 60)

    for variant in variants:
        col = f"{variant}_layer"
        if col not in commitment_df.columns:
            continue

        committed = commitment_df[commitment_df[col] >= 0]
        uncommitted = commitment_df[commitment_df[col] < 0]

        if len(committed) > 0:
            avg = committed[col].mean()
            median = committed[col].median()
            print(
                f"  {variant}: avg={avg:.1f}, median={median:.0f}, "
                f"committed={len(committed)}/{len(commitment_df)}, "
                f"uncommitted={len(uncommitted)}"
            )
        else:
            print(f"  {variant}: no commitments found")

    # Language breakdown for Hindi (primary finding)
    print()
    print("-" * 60)
    print("HINDI COMMITMENT BY VARIANT (Primary finding)")
    print("-" * 60)

    hindi_rows = commitment_df[commitment_df["language"] == "hi"]
    for variant in variants:
        col = f"{variant}_layer"
        if col not in hindi_rows.columns:
            continue

        committed = hindi_rows[hindi_rows[col] >= 0]
        if len(committed) > 0:
            avg = committed[col].mean()
            print(f"  {variant}: avg Hindi commitment layer = {avg:.1f}")
        else:
            print(f"  {variant}: no Hindi commitments")

    print()
    print("=" * 60)
    print("Analysis complete. Results written to results/ directory.")
    print("=" * 60)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute cross-lingual concept alignment curves and "
            "commitment matrices from residual stream activations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run concept_alignment.py                    # all variants\n"
            "  uv run concept_alignment.py --variant base     # base only\n"
            "  uv run concept_alignment.py --threshold 0.90   # stricter\n"
            "  uv run concept_alignment.py --consecutive 5    # more robust\n"
        ),
    )
    parser.add_argument(
        "--variant",
        choices=ALL_VARIANTS,
        default=None,
        help="Analyze a single variant only (default: all).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Commitment threshold (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--consecutive",
        type=int,
        default=DEFAULT_CONSECUTIVE,
        help=f"Consecutive layers for commitment (default: {DEFAULT_CONSECUTIVE}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for concept_alignment CLI."""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    variants = [args.variant] if args.variant else None

    try:
        run_analysis(
            variants=variants,
            threshold=args.threshold,
            consecutive=args.consecutive,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
