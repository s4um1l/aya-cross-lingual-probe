# ADR-003: FLORES-200 over machine-translated stimuli

## Status: Accepted

## Context

Cross-lingual alignment analysis requires parallel sentences that are semantically equivalent across all 10 target languages. The quality of these translations directly determines whether measured cosine similarity reflects genuine concept alignment or translation artifacts.

## Options Considered

| Option | Pro | Con |
|--------|-----|-----|
| FLORES-200 (human-translated) | Professional human translations; 200+ languages; verified parallel corpus; standard benchmark in multilingual NLP | Fixed domain coverage; requires keyword filtering for health domain; limited to existing sentences |
| Machine translation (e.g., Google Translate) | Unlimited sentence generation; any domain; cheap and fast | Translation quality varies by language pair; introduces MT artifacts that may confuse alignment measurement; no verification guarantee |

## Decision

Use FLORES-200 as the primary stimulus source. Health-domain sentences are selected via keyword filtering on the English devtest split, then parallel sentences in all 10 languages are extracted. Concept probes (20 medical concepts) are hand-verified against FLORES translations rather than machine-generated.

## Consequences

- **Enables:** High-confidence semantic equivalence across languages; reproducible stimulus set (FLORES is versioned and public); alignment measurements reflect model behavior rather than translation noise.
- **Constrains:** Limited to ~50 health-domain sentences per language from the FLORES devtest split. Concept probes must be manually verified, which is labor-intensive for 20 concepts x 10 languages.
