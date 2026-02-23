"""
Build stimulus manifest from concept probes and FLORES-200 health sentences.

Combines hand-verified concept probes (20 concepts x 10 languages) with
FLORES-200 health-domain parallel sentences to create the full stimulus set
for activation extraction.

Usage:
    uv run build_stimuli.py
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ── Project paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROBES_PATH = DATA_DIR / "concept_probes.json"
MANIFEST_PATH = DATA_DIR / "stimulus_manifest.json"
FLORES_CACHE_DIR = DATA_DIR / "flores_cache"

# ── Language code mapping: ISO 639-1 → FLORES-200 ─────────────────────────
FLORES_LANG_CODES: dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "sw": "swh_Latn",
    "am": "amh_Ethi",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "ar": "arb_Arab",
    "yo": "yor_Latn",
    "ta": "tam_Taml",
}

# ── Health domain keywords (English) for FLORES filtering ─────────────────
HEALTH_KEYWORDS_EN: list[str] = [
    "disease", "illness", "fever", "pain", "hospital", "doctor", "medicine",
    "treatment", "symptom", "infection", "child", "mother", "birth", "death",
    "food", "water", "body", "blood", "cough", "breathing", "diarrhea", "skin",
]

TARGET_FLORES_PER_LANG = 50


def load_concept_probes() -> dict:
    """Load and validate concept_probes.json."""
    with open(PROBES_PATH) as f:
        probes = json.load(f)

    langs = list(FLORES_LANG_CODES.keys())
    assert len(probes) == 20, f"Expected 20 concepts, got {len(probes)}"
    for cid, entry in probes.items():
        for lang in langs:
            assert lang in entry["translations"], f"{cid} missing language {lang}"
    return probes


def load_flores_health_sentences() -> dict[str, list[dict]]:
    """
    Load FLORES-200 devtest split and filter to health-domain sentences.

    Uses English text to identify health-relevant sentence indices, then
    retrieves parallel sentences in all 10 languages.

    Returns:
        Dict mapping ISO 639-1 code to list of sentence dicts.
    """
    print("Loading FLORES-200 dataset...")
    flores = load_dataset(
        "facebook/flores",
        "all",
        split="devtest",
        cache_dir=str(FLORES_CACHE_DIR),
        trust_remote_code=True,
    )

    # Step 1: Find health-relevant sentence indices using English text
    en_key = f"sentence_{FLORES_LANG_CODES['en']}"
    health_indices: list[int] = []

    print("Filtering for health-domain sentences...")
    for idx, row in enumerate(flores):
        en_text = row.get(en_key, "").lower()
        if any(kw in en_text for kw in HEALTH_KEYWORDS_EN):
            health_indices.append(idx)

    print(f"Found {len(health_indices)} health-domain sentences in English")

    # Step 2: Cap at TARGET_FLORES_PER_LANG per language (take first N)
    selected_indices = health_indices[:TARGET_FLORES_PER_LANG]
    if len(health_indices) < TARGET_FLORES_PER_LANG:
        warnings.warn(
            f"Only found {len(health_indices)} health sentences, "
            f"target was {TARGET_FLORES_PER_LANG}. Using all available.",
            stacklevel=2,
        )

    # Step 3: Extract parallel sentences for all languages
    result: dict[str, list[dict]] = {lang: [] for lang in FLORES_LANG_CODES}

    for lang_iso, lang_flores in FLORES_LANG_CODES.items():
        col_key = f"sentence_{lang_flores}"
        for rank, idx in enumerate(selected_indices):
            row = flores[idx]
            text = row.get(col_key, "")
            if text:
                result[lang_iso].append({
                    "flores_index": idx,
                    "text": text,
                    "language": lang_iso,
                    "source": "flores-200-devtest",
                    "rank": rank,
                })

    return result


def build_probe_stimuli(probes: dict) -> list[dict]:
    """Convert concept probes into flat stimulus list."""
    stimuli: list[dict] = []
    for cid, entry in sorted(probes.items()):
        for lang, text in entry["translations"].items():
            stimuli.append({
                "stimulus_id": f"probe_{cid}_{lang}",
                "type": "concept_probe",
                "concept_id": cid,
                "concept": entry["concept"],
                "category": entry["category"],
                "language": lang,
                "text": text,
            })
    return stimuli


def build_flores_stimuli(flores_sentences: dict[str, list[dict]]) -> list[dict]:
    """Convert FLORES sentences into flat stimulus list."""
    stimuli: list[dict] = []
    for lang, sentences in sorted(flores_sentences.items()):
        for sent in sentences:
            stimuli.append({
                "stimulus_id": f"flores_{lang}_{sent['rank']:03d}",
                "type": "flores",
                "concept_id": None,
                "concept": None,
                "category": "health_domain",
                "language": lang,
                "text": sent["text"],
                "flores_index": sent["flores_index"],
            })
    return stimuli


def validate_manifest(manifest: dict) -> None:
    """Run validation checks on the final manifest."""
    probes = manifest["probes"]
    flores = manifest["flores"]

    # Check 1: All 20 concepts x 10 languages present in probes
    concepts = set()
    langs_per_concept: dict[str, set] = {}
    for p in probes:
        cid = p["concept_id"]
        concepts.add(cid)
        if cid not in langs_per_concept:
            langs_per_concept[cid] = set()
        langs_per_concept[cid].add(p["language"])

    assert len(concepts) == 20, f"Expected 20 concepts, got {len(concepts)}"
    for cid, langs in langs_per_concept.items():
        assert len(langs) == 10, f"{cid} has {len(langs)} languages, expected 10"

    # Check 2: FLORES >= 50 per language (warn if fewer)
    flores_counts: dict[str, int] = {}
    for f_item in flores:
        lang = f_item["language"]
        flores_counts[lang] = flores_counts.get(lang, 0) + 1

    for lang, count in sorted(flores_counts.items()):
        if count < TARGET_FLORES_PER_LANG:
            warnings.warn(
                f"FLORES: {lang} has only {count} sentences "
                f"(target: {TARGET_FLORES_PER_LANG})",
                stacklevel=2,
            )

    print("\n── Validation Summary ──")
    print(f"  Probe stimuli:  {len(probes)} (20 concepts x 10 languages)")
    print(f"  FLORES stimuli: {len(flores)}")
    print(f"  Total stimuli:  {len(probes) + len(flores)}")
    print(f"  Languages:      {sorted(flores_counts.keys())}")
    print(f"  FLORES per lang:")
    for lang, count in sorted(flores_counts.items()):
        status = "OK" if count >= TARGET_FLORES_PER_LANG else "WARN"
        print(f"    {lang}: {count} [{status}]")
    print("  Validation: PASS")


def main() -> None:
    """Build and save the stimulus manifest."""
    print("=" * 60)
    print("Building stimulus manifest")
    print("=" * 60)

    # Load concept probes
    print("\n[1/4] Loading concept probes...")
    probes = load_concept_probes()
    probe_stimuli = build_probe_stimuli(probes)
    print(f"  Loaded {len(probe_stimuli)} probe stimuli")

    # Load FLORES-200
    print("\n[2/4] Loading FLORES-200 health sentences...")
    flores_sentences = load_flores_health_sentences()
    flores_stimuli = build_flores_stimuli(flores_sentences)
    print(f"  Loaded {len(flores_stimuli)} FLORES stimuli")

    # Build manifest
    print("\n[3/4] Building manifest...")
    manifest = {
        "metadata": {
            "total_stimuli": len(probe_stimuli) + len(flores_stimuli),
            "flores_sentences": len(flores_stimuli),
            "concept_probes": len(probe_stimuli),
            "languages": len(FLORES_LANG_CODES),
            "concepts": 20,
            "flores_per_language_target": TARGET_FLORES_PER_LANG,
            "health_keywords": HEALTH_KEYWORDS_EN,
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "flores": flores_stimuli,
        "probes": probe_stimuli,
    }

    # Validate
    print("\n[4/4] Validating manifest...")
    validate_manifest(manifest)

    # Write output
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nManifest written to: {MANIFEST_PATH}")
    print(f"File size: {MANIFEST_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
