"""
Sequential batch runner: loads each model variant, runs all stimuli,
saves float16 activations to disk, unloads before next model.

Dual-mode operation:
    --local     Run on local CPU/MPS using model_loader.py directly
    (default)   Run on Modal via modal_config.py (A10G GPU)
    --smoke-test Only process 5 stimuli with base model (for testing)
    --variant   Run a specific variant only (base|fire|earth)

Activation extraction uses HuggingFace transformers output_hidden_states=True
to capture residual stream at every layer (Cohere2 architecture not supported
by TransformerLens).

Output format:
    activations/{variant}/{stimulus_id}_resid.npy
    Shape: (n_layers+1, d_model) -- one vector per layer per stimulus
    dtype: float16
    Uses LAST token position's activation

Checkpoint protocol:
    activations/{variant}/complete.flag written LAST after all stimuli done.
    Re-running skips variants with complete.flag.
    Within a variant, skip existing .npy files.

Usage:
    uv run batch_runner.py --local                    # full local run
    uv run batch_runner.py --local --smoke-test       # 5 stimuli, base only
    uv run batch_runner.py --local --variant fire     # local, fire only
    uv run batch_runner.py                            # full Modal run
    uv run batch_runner.py --smoke-test               # Modal smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Project paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "stimulus_manifest.json"
ACTIVATIONS_DIR = ROOT / "activations"

# ── All variant names ──────────────────────────────────────────────────────
ALL_VARIANTS = ["base", "fire", "earth"]


def load_stimuli(smoke_test: bool = False) -> list[dict]:
    """
    Load all stimuli from stimulus_manifest.json.

    Args:
        smoke_test: If True, return only the first 5 stimuli.

    Returns:
        List of stimulus dicts with 'stimulus_id' and 'text' keys.
    """
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Stimulus manifest not found: {MANIFEST_PATH}\n"
            f"Run 'uv run build_stimuli.py' first."
        )

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    stimuli = manifest.get("probes", []) + manifest.get("flores", [])

    if smoke_test:
        stimuli = stimuli[:5]
        logger.info(f"Smoke test mode: using {len(stimuli)} stimuli only.")

    logger.info(f"Loaded {len(stimuli)} stimuli from {MANIFEST_PATH}")
    return stimuli


def is_variant_complete(variant: str) -> bool:
    """Check if a variant has a complete.flag file."""
    flag_path = ACTIVATIONS_DIR / variant / "complete.flag"
    return flag_path.exists()


def write_complete_flag(variant: str) -> None:
    """Write complete.flag after all stimuli for a variant are processed."""
    flag_path = ACTIVATIONS_DIR / variant / "complete.flag"
    flag_path.write_text(
        f"completed_at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
    )
    logger.info(f"Wrote complete flag: {flag_path}")


# ── Local extraction ──────────────────────────────────────────────────────


def extract_activations_local(
    variant: str,
    stimuli: list[dict],
) -> dict:
    """
    Run activation extraction locally using HuggingFace transformers.

    Uses output_hidden_states=True to capture residual stream at all layers.

    Args:
        variant: Model variant name (base/fire/earth).
        stimuli: List of stimulus dicts.

    Returns:
        Dict with processing statistics.
    """
    from tqdm import tqdm

    from model_loader import extract_hidden_states, load_model, unload_model

    variant_dir = ACTIVATIONS_DIR / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading variant '{variant}' locally...")
    model, tokenizer = load_model(variant)

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    logger.info(f"Loaded: n_layers={n_layers}, d_model={d_model}")

    # Process stimuli
    processed = 0
    skipped = 0
    errors = 0

    for stim in tqdm(stimuli, desc=f"[local] {variant}"):
        stim_id = stim["stimulus_id"]
        text = stim["text"]
        out_path = variant_dir / f"{stim_id}_resid.npy"

        # Checkpoint: skip existing files
        if out_path.exists():
            skipped += 1
            continue

        try:
            # Extract hidden states at all layers
            hidden_states = extract_hidden_states(model, tokenizer, text)

            # Convert to float16 numpy
            activation = hidden_states.cpu().to(dtype=hidden_states.dtype).float().half().numpy()

            # Save
            np.save(str(out_path), activation)
            processed += 1

        except Exception as e:
            logger.error(f"Error processing '{stim_id}': {e}")
            errors += 1

    # Unload model to free memory
    unload_model(model)

    result = {
        "variant": variant,
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "total": len(stimuli),
    }
    logger.info(f"Local extraction done: {result}")
    return result


# ── Modal extraction ──────────────────────────────────────────────────────


def extract_activations_modal(
    variant: str,
    stimuli: list[dict],
) -> dict:
    """
    Run activation extraction on Modal GPU.

    Args:
        variant: Model variant name (base/fire/earth).
        stimuli: List of stimulus dicts.

    Returns:
        Dict with processing statistics.
    """
    try:
        from modal_config import app, extract_activations_remote
    except ImportError:
        raise ImportError(
            "Modal is not installed. Install with: uv add modal\n"
            "Or use --local flag to run locally."
        )

    from model_loader import VARIANT_CONFIGS

    hf_name = VARIANT_CONFIGS[variant]["hf_name"]

    logger.info(
        f"Submitting {len(stimuli)} stimuli for '{variant}' to Modal..."
    )

    # Run within Modal App context
    with app.run():
        result = extract_activations_remote.remote(
            variant=variant,
            stimuli=stimuli,
            hf_name=hf_name,
        )

    logger.info(f"Modal extraction done: {result}")

    # Download results from Modal Volume to local activations/
    _sync_volume_to_local(variant)

    return result


def _sync_volume_to_local(variant: str) -> None:
    """Download .npy files from Modal Volume to local activations/ directory."""
    import subprocess

    local_dir = ACTIVATIONS_DIR / variant
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Syncing Modal volume 'aya-activations/{variant}/' to {local_dir}...")

    try:
        # Download to parent dir since modal volume get creates the variant subdir
        result = subprocess.run(
            [
                "modal", "volume", "get",
                "aya-activations",
                f"{variant}/",
                str(local_dir.parent),
                "--force",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            logger.warning(
                f"modal volume get returned non-zero: {result.stderr.strip()}"
            )
        else:
            npy_count = len(list(local_dir.glob("*.npy")))
            logger.info(f"Synced {npy_count} .npy files to {local_dir}")

    except FileNotFoundError:
        logger.warning(
            "Modal CLI not found. Install with: uv add modal\n"
            "You can manually download files from the 'aya-activations' volume."
        )
    except subprocess.TimeoutExpired:
        logger.warning("Volume sync timed out after 5 minutes.")


# ── Main orchestrator ─────────────────────────────────────────────────────


def run_batch(
    local: bool = False,
    smoke_test: bool = False,
    variant_filter: str | None = None,
) -> None:
    """
    Orchestrate batch activation extraction across all variants.

    Args:
        local: If True, run on local CPU/MPS. Otherwise use Modal.
        smoke_test: If True, only process 5 stimuli with base model.
        variant_filter: If set, only process this single variant.
    """
    mode = "local" if local else "Modal"
    logger.info(f"Starting batch run (mode={mode}, smoke_test={smoke_test})")

    # Load stimuli
    stimuli = load_stimuli(smoke_test=smoke_test)
    logger.info(f"Total stimuli to process: {len(stimuli)}")

    # Determine which variants to process
    if smoke_test:
        variants = ["base"]
        logger.info("Smoke test: running base variant only.")
    elif variant_filter:
        if variant_filter not in ALL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant_filter}'. "
                f"Choose from: {ALL_VARIANTS}"
            )
        variants = [variant_filter]
    else:
        variants = ALL_VARIANTS

    # Process each variant sequentially
    all_results = []
    for variant in variants:
        if is_variant_complete(variant):
            logger.info(
                f"Variant '{variant}' already complete (complete.flag found). Skipping."
            )
            continue

        logger.info(f"Processing variant '{variant}'...")
        start_time = time.time()

        if local:
            result = extract_activations_local(variant, stimuli)
        else:
            result = extract_activations_modal(variant, stimuli)

        elapsed = time.time() - start_time
        result["elapsed_seconds"] = round(elapsed, 1)
        all_results.append(result)

        # Write complete.flag if no errors
        if result["errors"] == 0:
            variant_dir = ACTIVATIONS_DIR / variant
            expected_count = len(stimuli)
            actual_count = len(list(variant_dir.glob("*_resid.npy")))

            if actual_count >= expected_count:
                write_complete_flag(variant)
            else:
                logger.warning(
                    f"Variant '{variant}': expected {expected_count} files, "
                    f"found {actual_count}. NOT writing complete.flag."
                )
        else:
            logger.warning(
                f"Variant '{variant}' had {result['errors']} error(s). "
                f"NOT writing complete.flag. Re-run to retry."
            )

    # Summary
    print("\n" + "=" * 60)
    print("BATCH RUN SUMMARY")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Stimuli per variant: {len(stimuli)}")
    print(f"Variants requested: {variants}")
    print()

    for r in all_results:
        print(f"  {r['variant']}:")
        print(f"    Processed: {r['processed']}")
        print(f"    Skipped:   {r['skipped']}")
        print(f"    Errors:    {r['errors']}")
        print(f"    Time:      {r.get('elapsed_seconds', '?')}s")

    if not all_results:
        print("  (All variants already complete -- nothing to do.)")

    print()
    for v in variants:
        flag = "YES" if is_variant_complete(v) else "NO"
        npy_dir = ACTIVATIONS_DIR / v
        npy_count = len(list(npy_dir.glob("*_resid.npy"))) if npy_dir.exists() else 0
        print(f"  {v}: complete.flag={flag}, .npy files={npy_count}")

    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Batch activation extraction for Tiny Aya variants. "
            "Extracts residual stream activations at all layers for each stimulus."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run batch_runner.py --local --smoke-test    # quick local test\n"
            "  uv run batch_runner.py --local                 # full local run\n"
            "  uv run batch_runner.py --local --variant fire  # local, fire only\n"
            "  uv run batch_runner.py                         # full Modal run\n"
            "  uv run batch_runner.py --variant base          # Modal, base only\n"
        ),
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally on CPU/MPS instead of Modal GPU.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Only process 5 stimuli with base model (for testing).",
    )
    parser.add_argument(
        "--variant",
        choices=ALL_VARIANTS,
        default=None,
        help="Run a specific variant only (base|fire|earth).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for batch_runner CLI."""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        run_batch(
            local=args.local,
            smoke_test=args.smoke_test,
            variant_filter=args.variant,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Progress saved (checkpoint protocol).")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Batch run failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
