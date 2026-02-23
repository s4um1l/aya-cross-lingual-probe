---
language:
  - en
  - hi
  - bn
  - sw
  - am
  - fr
  - es
  - ar
  - yo
  - ta
license: apache-2.0
task_categories:
  - feature-extraction
tags:
  - mechanistic-interpretability
  - cross-lingual
  - multilingual
  - medical
  - probing
pretty_name: Tiny Aya Cross-Lingual Medical Concept Probes
size_categories:
  - n<1K
---

# Tiny Aya Cross-Lingual Medical Concept Probes

## Dataset Description

20 medical concepts expressed as full sentences in 10 languages, designed for probing cross-lingual concept representations in multilingual LLMs. Each concept is a complete declarative sentence preserving the same semantic structure across all languages.

### Purpose

These probe sentences serve as stimuli for mechanistic interpretability analysis -- specifically, extracting residual stream activations from Tiny Aya model variants (Base, Fire, Earth; 3.35B parameters each) and measuring cross-lingual alignment via cosine similarity at every transformer layer.

## Languages

| Code | Language | Script |
|------|----------|--------|
| en | English | Latin |
| hi | Hindi | Devanagari |
| bn | Bengali | Bengali |
| sw | Swahili | Latin |
| am | Amharic | Ethiopic |
| fr | French | Latin |
| es | Spanish | Latin |
| ar | Arabic | Arabic |
| yo | Yoruba | Latin |
| ta | Tamil | Tamil |

## Dataset Structure

**Size:** 200 probe sentences (20 concepts x 10 languages)

**Format:** JSON with concept IDs (C01-C20) as keys. Each entry contains:

- `concept` -- English concept label
- `category` -- semantic category
- `translations` -- object mapping language code to full sentence
- `flores_verified` -- boolean indicating FLORES-200 verification status

### Concept Categories

| Category | Concepts |
|----------|----------|
| symptom | fever, cough, diarrhea, pain |
| entity | child, mother, doctor, medicine, hospital, water, newborn |
| action | breathing, eating, sleeping |
| severity | dangerous |
| state | sick, healthy |
| disease | malaria, infection, dehydration |

### Example

```json
{
  "C01": {
    "concept": "fever",
    "category": "symptom",
    "translations": {
      "en": "The child has a fever.",
      "hi": "बच्चे को बुखार है।",
      "sw": "Mtoto ana homa.",
      "am": "ህፃኑ ትኩሳት አለበት።",
      "fr": "L'enfant a de la fièvre.",
      "...": "..."
    },
    "flores_verified": true
  }
}
```

## Data Collection

Translations were authored with reference to the FLORES-200 parallel corpus and hand-verified for semantic equivalence across all 10 languages. Full sentences (rather than isolated tokens) are used because transformer residual streams require sufficient context for stable cross-lingual representations.

## Use Case

**Mechanistic interpretability** -- activation extraction and cross-lingual alignment analysis across transformer layers. This dataset was used to study how regional fine-tuning affects concept representation in Tiny Aya variants, revealing a universal rise-peak-collapse alignment architecture where models build shared cross-lingual representations in mid-network layers (L18-20) then dismantle them at the final layers.

## Limitations

- Small sample (N=20 concepts) limited to the medical domain
- Translations cover 10 languages; results may not generalize to other languages
- Designed specifically for probing Aya-family models; utility for other architectures is untested

## Citation

For methodology and findings, see the research report in this repository:

```
REPORT.md — "Tiny Aya Builds Shared Concepts Mid-Network Then Destroys Them
at Output Layers — And Regional Fine-Tuning Determines How Well It Builds Them"
Saumil Srivastava, February 2026
```

## License

Apache 2.0
