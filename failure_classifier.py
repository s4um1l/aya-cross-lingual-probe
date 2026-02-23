"""
Identifies inputs where a regional variant underperforms Base.
Builds failure taxonomy for language routing edge cases.

Categories:
    CODE_SWITCH:        input contains mixed-language tokens / multiple scripts
    TRANSLITERATION:    Hindi written in Latin script (Hinglish), etc.
    FORMAL_REGISTER:    formal/medical terminology vs conversational
    LOW_RESOURCE_VARIANT: dialect within a language family
    UNICODE_EDGE:       mixed script directions, special characters

A failure is defined as:
    - variant_commitment > base_commitment + 4 layers  (variant significantly worse)
    - OR variant_commitment == -1 and base_commitment != -1  (variant never commits)

Usage:
    uv run failure_classifier.py                     # full analysis
    uv run failure_classifier.py --verbose           # debug logging
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Project paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROBES_PATH = DATA_DIR / "concept_probes.json"
MANIFEST_PATH = DATA_DIR / "stimulus_manifest.json"
RESULTS_DIR = ROOT / "results"
COMMITMENT_MATRIX_PATH = RESULTS_DIR / "commitment_matrix.csv"

# ── Constants ──────────────────────────────────────────────────────────────
ALL_VARIANTS = ["base", "fire", "earth"]
REGIONAL_VARIANTS = ["fire", "earth"]

# Failure threshold: variant must be this many layers worse than base
FAILURE_DELTA = 4

# Script ranges for language detection
SCRIPT_RANGES: dict[str, str] = {
    "hi": "Devanagari",
    "bn": "Bengali",
    "ta": "Tamil",
    "am": "Ethiopic",
    "ar": "Arabic",
    "yo": "Latin",
    "sw": "Latin",
    "fr": "Latin",
    "es": "Latin",
    "en": "Latin",
}

# Expected scripts for non-Latin languages
NON_LATIN_LANGUAGES = {"hi", "bn", "ta", "am", "ar"}

# Formal / medical terminology markers
FORMAL_MARKERS_EN = {
    "diagnosis", "diagnosed", "prognosis", "clinical", "pathology",
    "etiology", "contraindicated", "prophylaxis", "hemorrhage",
    "intravenous", "subcutaneous", "intramuscular", "pharmaceutical",
    "antibiotic", "antiviral", "antimicrobial", "ventilation",
    "resuscitation", "triage", "epidemiology", "comorbidity",
}

# Extended formal markers across languages (common medical Latin/Greek roots)
FORMAL_MARKERS_CROSS = {
    "syndrome", "therapy", "chronic", "acute", "carcinoma",
    "benign", "malignant", "trauma", "surgical", "anesthesia",
}


# ── Script detection utilities ─────────────────────────────────────────────


def _get_script_categories(text: str) -> set[str]:
    """
    Return the set of Unicode script categories present in text.

    Filters out Common and Inherited categories (punctuation, digits, etc.)
    to focus on actual script blocks.

    Args:
        text: Input text.

    Returns:
        Set of script names (e.g. {'Latin', 'Devanagari'}).
    """
    scripts = set()
    for char in text:
        if char.isspace() or char in '.,;:!?()[]{}"\'-/\\@#$%^&*_+=~`|<>':
            continue
        try:
            cat = unicodedata.category(char)
            # Skip number/symbol categories
            if cat.startswith(("N", "S", "P", "Z")):
                continue
            name = unicodedata.name(char, "")
            # Extract script from character name
            if "DEVANAGARI" in name:
                scripts.add("Devanagari")
            elif "BENGALI" in name:
                scripts.add("Bengali")
            elif "TAMIL" in name:
                scripts.add("Tamil")
            elif "ETHIOPIC" in name:
                scripts.add("Ethiopic")
            elif "ARABIC" in name:
                scripts.add("Arabic")
            elif "LATIN" in name or cat == "Ll" or cat == "Lu":
                scripts.add("Latin")
            elif "CJK" in name:
                scripts.add("CJK")
            elif "CYRILLIC" in name:
                scripts.add("Cyrillic")
            else:
                # Fallback: try to infer from general category
                if cat.startswith("L"):
                    # It's a letter, try code point range
                    cp = ord(char)
                    if 0x0900 <= cp <= 0x097F:
                        scripts.add("Devanagari")
                    elif 0x0980 <= cp <= 0x09FF:
                        scripts.add("Bengali")
                    elif 0x0B80 <= cp <= 0x0BFF:
                        scripts.add("Tamil")
                    elif 0x1200 <= cp <= 0x137F:
                        scripts.add("Ethiopic")
                    elif 0x0600 <= cp <= 0x06FF:
                        scripts.add("Arabic")
                    elif 0x0041 <= cp <= 0x024F:
                        scripts.add("Latin")
                    else:
                        scripts.add(f"Unknown({hex(cp)})")
        except ValueError:
            continue

    return scripts


def detect_code_switching(text: str) -> bool:
    """
    Check if text contains tokens from multiple scripts.

    Code-switching is when a single sentence contains words in
    different scripts (e.g., Hindi words mixed with English words
    in Devanagari + Latin script).

    Args:
        text: Input text.

    Returns:
        True if multiple scripts detected.
    """
    scripts = _get_script_categories(text)
    # Filter out minor script presence (single chars, numbers)
    significant_scripts = set()
    for script in scripts:
        # Count characters in this script
        count = 0
        for char in text:
            if char.isspace():
                continue
            char_scripts = _get_script_categories(char)
            if script in char_scripts:
                count += 1
        # At least 2 characters to be significant
        if count >= 2:
            significant_scripts.add(script)

    return len(significant_scripts) > 1


def detect_transliteration(text: str, expected_lang: str) -> bool:
    """
    Check if text is in Latin script when the expected language uses
    a different script.

    For example, Hindi (expected: Devanagari) written as "mera bachcha
    bahut sick ho gaya" (Latin script = Hinglish).

    Args:
        text: Input text.
        expected_lang: ISO 639-1 language code.

    Returns:
        True if text appears to be transliterated (Latin script when
        non-Latin expected).
    """
    if expected_lang not in NON_LATIN_LANGUAGES:
        return False

    scripts = _get_script_categories(text)

    # If the text is predominantly Latin but the language expects
    # a non-Latin script, it is likely transliterated
    expected_script = SCRIPT_RANGES.get(expected_lang, "Latin")

    if "Latin" in scripts and expected_script not in scripts:
        return True

    # Mixed case: mostly Latin with a few native script chars
    if "Latin" in scripts and expected_script in scripts:
        latin_count = sum(
            1 for c in text if c.isalpha() and ord(c) < 0x0250
        )
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha > 0 and latin_count / total_alpha > 0.7:
            return True

    return False


def detect_formal_register(text: str) -> bool:
    """
    Check for medical/formal terminology markers.

    Looks for Latinate medical terms that may be unfamiliar to
    conversationally-tuned models.

    Args:
        text: Input text.

    Returns:
        True if formal medical terminology detected.
    """
    text_lower = text.lower()
    words = set(re.findall(r"[a-zA-Z]+", text_lower))

    formal_matches = words & (FORMAL_MARKERS_EN | FORMAL_MARKERS_CROSS)
    return len(formal_matches) >= 1


def detect_unicode_edge(text: str) -> bool:
    """
    Check for Unicode edge cases: mixed RTL/LTR, combining characters,
    zero-width joiners, special punctuation, etc.

    Args:
        text: Input text.

    Returns:
        True if Unicode edge cases detected.
    """
    has_rtl = False
    has_ltr = False
    has_combining = False
    has_zwj = False

    for char in text:
        cp = ord(char)
        cat = unicodedata.category(char)

        # RTL characters (Arabic, Hebrew)
        bidi = unicodedata.bidirectional(char)
        if bidi in ("R", "AL", "AN"):
            has_rtl = True
        elif bidi == "L":
            has_ltr = True

        # Combining marks
        if cat.startswith("M"):
            has_combining = True

        # Zero-width characters
        if cp in (0x200B, 0x200C, 0x200D, 0xFEFF):
            has_zwj = True

    # Mixed directionality is an edge case
    if has_rtl and has_ltr:
        return True

    # Unusual combining character usage
    if has_combining and has_zwj:
        return True

    return False


def detect_low_resource_variant(text: str, language: str) -> bool:
    """
    Check if text might be a low-resource dialect or regional variant.

    This is a heuristic check: if the text uses non-standard character
    combinations for the language, or if the language itself is low-resource
    (Yoruba, Amharic for many NLP tasks).

    Args:
        text: Input text.
        language: ISO 639-1 code.

    Returns:
        True if text may be a low-resource variant.
    """
    low_resource_langs = {"yo", "am", "ta", "bn"}

    if language in low_resource_langs:
        # These languages are inherently lower-resource in most LLMs
        return True

    return False


# ── Failure classification ─────────────────────────────────────────────────


def classify_failure(
    stimulus_id: str,
    text: str,
    language: str,
    base_commit: int,
    variant_commit: int,
    variant: str,
) -> dict | None:
    """
    Determine if this (stimulus, variant) pair is a failure case, and
    classify the failure into one of 5 categories.

    A failure is defined as:
        - variant_commit > base_commit + FAILURE_DELTA  (variant significantly worse)
        - OR variant_commit == -1 and base_commit != -1  (variant never commits)

    Args:
        stimulus_id: e.g. 'probe_C01_hi'.
        text: The stimulus text.
        language: ISO 639-1 language code.
        base_commit: Base model commitment layer (-1 if never committed).
        variant_commit: Regional variant commitment layer.
        variant: Regional variant name ('fire' or 'earth').

    Returns:
        Failure record dict, or None if not a failure.
    """
    # Check failure condition
    is_failure = False

    if variant_commit == -1 and base_commit != -1:
        is_failure = True
    elif variant_commit != -1 and base_commit != -1:
        if variant_commit > base_commit + FAILURE_DELTA:
            is_failure = True

    if not is_failure:
        return None

    # Classify the failure
    category = "UNKNOWN"
    notes = ""

    if detect_code_switching(text):
        category = "CODE_SWITCH"
        scripts = _get_script_categories(text)
        notes = f"Mixed scripts detected: {scripts}"
    elif detect_transliteration(text, language):
        category = "TRANSLITERATION"
        expected = SCRIPT_RANGES.get(language, "Latin")
        notes = f"Expected {expected} script, found Latin (transliterated)"
    elif detect_formal_register(text):
        category = "FORMAL_REGISTER"
        text_lower = text.lower()
        words = set(re.findall(r"[a-zA-Z]+", text_lower))
        matches = words & (FORMAL_MARKERS_EN | FORMAL_MARKERS_CROSS)
        notes = f"Formal terms: {matches}"
    elif detect_unicode_edge(text):
        category = "UNICODE_EDGE"
        notes = "Mixed directionality or special Unicode features"
    elif detect_low_resource_variant(text, language):
        category = "LOW_RESOURCE_VARIANT"
        notes = f"Low-resource language: {language}"
    else:
        category = "UNKNOWN"
        notes = "No specific category matched; manual review needed"

    # Build text preview (first 80 chars)
    text_preview = text[:80] + ("..." if len(text) > 80 else "")

    return {
        "stimulus_id": stimulus_id,
        "language": language,
        "text_preview": text_preview,
        "base_commit": base_commit,
        f"{variant}_commit": variant_commit,
        "failure_category": category,
        "notes": notes,
    }


# ── Orchestrator ───────────────────────────────────────────────────────────


def _load_stimulus_texts() -> dict[str, dict]:
    """
    Load stimulus texts from the manifest file.

    Returns:
        Dict mapping stimulus_id to {'text': ..., 'language': ...,
        'concept_id': ..., 'concept': ...}.
    """
    stimuli: dict[str, dict] = {}

    # Load from concept probes directly
    if PROBES_PATH.exists():
        with open(PROBES_PATH) as f:
            probes = json.load(f)

        for concept_id, entry in probes.items():
            for lang, text in entry.get("translations", {}).items():
                stim_id = f"probe_{concept_id}_{lang}"
                stimuli[stim_id] = {
                    "text": text,
                    "language": lang,
                    "concept_id": concept_id,
                    "concept": entry.get("concept", ""),
                }

    # Also load from manifest for FLORES stimuli
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

        for stim in manifest.get("flores", []):
            stim_id = stim["stimulus_id"]
            if stim_id not in stimuli:
                stimuli[stim_id] = {
                    "text": stim["text"],
                    "language": stim.get("language", "en"),
                    "concept_id": stim.get("concept_id"),
                    "concept": stim.get("concept"),
                }

    return stimuli


def run_failure_classification() -> None:
    """
    Run failure classification across all stimuli and variants.

    Reads the commitment matrix from concept_alignment.py output,
    compares each regional variant against base, and classifies
    failure cases.
    """
    # Check for required input files
    if not COMMITMENT_MATRIX_PATH.exists():
        print("\n" + "=" * 60)
        print("ERROR: Commitment matrix not found.")
        print("=" * 60)
        print()
        print(f"Expected: {COMMITMENT_MATRIX_PATH}")
        print()
        print("Run concept_alignment.py first to generate the matrix:")
        print("  uv run concept_alignment.py")
        print("=" * 60)
        sys.exit(1)

    # Load data
    commitment_df = pd.read_csv(COMMITMENT_MATRIX_PATH)
    stimuli = _load_stimulus_texts()

    print("=" * 60)
    print("FAILURE CLASSIFIER")
    print("=" * 60)
    print(f"Commitment matrix: {len(commitment_df)} rows")
    print(f"Stimulus texts loaded: {len(stimuli)}")
    print()

    # Collect failure cases
    all_failures: list[dict] = []

    for _, row in commitment_df.iterrows():
        concept_id = row["concept_id"]
        concept = row.get("concept", "")
        language = row["language"]
        base_layer = int(row["base_layer"])

        stim_id = f"probe_{concept_id}_{language}"
        stim_info = stimuli.get(stim_id, {})
        text = stim_info.get("text", "")

        # Check each regional variant against base
        for variant in REGIONAL_VARIANTS:
            variant_col = f"{variant}_layer"
            if variant_col not in row:
                continue

            variant_layer = int(row[variant_col])

            failure = classify_failure(
                stimulus_id=stim_id,
                text=text,
                language=language,
                base_commit=base_layer,
                variant_commit=variant_layer,
                variant=variant,
            )

            if failure is not None:
                # Add concept info
                failure["concept"] = concept
                failure["concept_id"] = concept_id

                # Add the other variant's commitment for the full row
                for v in ALL_VARIANTS:
                    col = f"{v}_layer"
                    if col in row:
                        failure[f"{v}_commit"] = int(row[col])

                all_failures.append(failure)

    # Build output DataFrame
    if all_failures:
        # Normalize columns across all failures
        output_rows: list[dict] = []
        for f in all_failures:
            output_rows.append({
                "stimulus_id": f.get("stimulus_id", ""),
                "concept": f.get("concept", ""),
                "language": f.get("language", ""),
                "text_preview": f.get("text_preview", ""),
                "base_commit": f.get("base_commit", -1),
                "fire_commit": f.get("fire_commit", -1),
                "earth_commit": f.get("earth_commit", -1),
                "failure_category": f.get("failure_category", "UNKNOWN"),
                "notes": f.get("notes", ""),
            })

        failure_df = pd.DataFrame(output_rows)
    else:
        failure_df = pd.DataFrame(columns=[
            "stimulus_id", "concept", "language", "text_preview",
            "base_commit", "fire_commit", "earth_commit",
            "failure_category", "notes",
        ])

    # Write output
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "failure_cases.csv"
    failure_df.to_csv(out_path, index=False)

    print(f"Total failure cases found: {len(failure_df)}")
    print(f"Output: {out_path}")

    # Print category breakdown
    if len(failure_df) > 0:
        print()
        print("-" * 60)
        print("FAILURE CATEGORY BREAKDOWN")
        print("-" * 60)
        category_counts = failure_df["failure_category"].value_counts()
        for cat, count in category_counts.items():
            print(f"  {cat}: {count}")

        # Print language breakdown
        print()
        print("-" * 60)
        print("FAILURES BY LANGUAGE")
        print("-" * 60)
        lang_counts = failure_df["language"].value_counts()
        for lang, count in lang_counts.items():
            print(f"  {lang}: {count}")

        # Print a few example failures
        print()
        print("-" * 60)
        print("SAMPLE FAILURES (first 5)")
        print("-" * 60)
        for _, row in failure_df.head(5).iterrows():
            print(
                f"  {row['stimulus_id']} ({row['language']}): "
                f"base={row['base_commit']}, fire={row['fire_commit']}, "
                f"earth={row['earth_commit']} "
                f"[{row['failure_category']}]"
            )
            print(f"    Text: {row['text_preview']}")
            print(f"    Notes: {row['notes']}")
            print()
    else:
        print()
        print("No failure cases found. This could mean:")
        print("  - Regional variants perform similarly to base")
        print("  - Threshold may need adjustment")
        print("  - Activation data may not be available")

    print("=" * 60)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Classify failure cases where regional variants underperform "
            "the Base model in cross-lingual concept alignment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run failure_classifier.py             # full analysis\n"
            "  uv run failure_classifier.py --verbose   # debug output\n"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for failure_classifier CLI."""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        run_failure_classification()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Failure classification failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
