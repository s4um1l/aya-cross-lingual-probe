# Production Delta

## What this is

Mechanistic interpretability analysis of Tiny Aya variants (Base, Fire, Earth) using layer-wise residual stream cosine similarity analysis. This is a research artifact that produces:

- **Cross-lingual alignment curves:** cosine similarity of concept activations between English and 9 target languages across all 37 transformer layers (embedding + 36 hidden), for 3 model variants.
- **Commitment matrices:** the layer at which each (concept, language, variant) triple "locks in" to a stable cross-lingual representation.
- **Failure taxonomy:** five categories of inputs where regional variants underperform Base, derived from commitment layer analysis.
- **Edge case test suite:** 32 hand-crafted inputs covering all failure categories, formatted for language routing integration.
- **Medical concept probe dataset:** 20 concepts in 10 languages, verified against FLORES-200 parallel sentences.

Hardware used: Apple Silicon Mac (local data prep, analysis, visualization) + Modal A10G GPU (activation extraction). Total compute time: approximately 20 minutes on GPU for batch extraction, plus 2-3 hours for local analysis and visualization.

## What production would require

- **Activation extraction at scale.** We use HuggingFace Transformers' `output_hidden_states=True` to capture all layer activations in a single forward pass. While simpler than hook-based approaches, this is still a research pattern: it doubles memory usage (storing all hidden states) and cannot run inside optimized serving frameworks. Production activation capture would require custom vLLM operators, Triton kernels, or instrumented model serving that exposes intermediate states without the memory overhead of retaining all hidden states.

- **Continuous probing across model updates.** This analysis is a point-in-time snapshot. When Tiny Aya variants are updated (new fine-tuning data, architecture changes, quantization), the commitment layer patterns may shift. A production system would need an automated regression pipeline that re-runs alignment analysis after each model update and alerts if commitment patterns degrade for critical language pairs.

- **Statistical significance testing.** With n=20 concepts, our findings describe trends but do not meet the bar for publication-grade statistical claims. A production evaluation would need 200+ concepts across multiple domains, with bootstrapped confidence intervals on commitment layer differences and correction for multiple comparisons across language pairs.

- **Human evaluation of translations.** The concept probes were hand-verified against FLORES-200 parallel sentences, but native speaker validation is still required. A mistranslated probe produces a spurious alignment curve. Production-grade probes would require back-translation verification by native speakers for all 10 languages.

- **Model serving architecture.** The Gradio app loads models per-request, which is acceptable for a demo but not for production. A deployed system would require persistent model serving (e.g., vLLM, TGI, or Triton Inference Server) with pre-cached activation snapshots for common inputs. The layer-by-layer animation would query a cache rather than running inference.

- **Edge case coverage.** The 32 edge cases in `p1_edge_cases.json` are hand-crafted based on the failure taxonomy, not derived from empirical analysis of real user inputs. Production would require mining actual CHW queries for code-switching, transliteration, and dialect patterns, then validating the router's behavior on these real-world inputs.

- **Latency constraints.** Residual stream analysis of all 37 layers adds approximately 3-5x overhead to a standard forward pass. A production router cannot afford this latency on every request. Instead, activation probing should run offline on representative samples, with the routing rules derived from the analysis applied as lightweight classifiers at inference time.

## What would NOT change

- **The research questions and methodology are valid at any scale.** Cross-lingual alignment via cosine similarity in the residual stream is a well-established mechanistic interpretability technique. The questions ("does regional fine-tuning create shared concepts earlier?" and "where do regional variants fail?") remain the right questions for routing decisions.

- **FLORES-200 as a stimulus source.** FLORES-200 provides human-translated parallel sentences across 200+ languages. It is the gold standard for controlled cross-lingual evaluation and would remain the stimulus source at production scale.

- **The cosine similarity alignment metric.** While cosine similarity has limitations (geometric proximity vs. functional equivalence), it is the standard metric for residual stream analysis and provides interpretable, comparable values across layers and variants.

- **The concept probe format.** The structure of `concept_probes.json` (concept ID, category, parallel translations, verification flag) is reusable for any multilingual model. The 20 medical concepts can be extended to 200+ without changing the format or analysis pipeline.

- **The failure taxonomy categories.** The five failure categories (TRANSLITERATION, CODE_SWITCH, FORMAL_REGISTER, LOW_RESOURCE_VARIANT, UNICODE_EDGE) describe language phenomena, not scale phenomena. Code-switching is code-switching whether you test with 20 or 20,000 inputs. The categories would gain more examples at scale but would not change in kind.

- **The commitment layer definition.** The operational definition (first layer with similarity >= threshold for N consecutive layers) is a clean, reproducible metric. The specific threshold (0.85) and consecutive count (3) are parameters that may be tuned, but the definition itself is sound.

## Gap summary

| Aspect | Research (this project) | Production requirement |
|--------|------------------------|----------------------|
| Concept count | 20 | 200+ |
| Languages | 10 | 20+ (add Mandarin, Portuguese, Indonesian, etc.) |
| Stimuli | 700 (probes + FLORES) | 10,000+ (real user queries) |
| Activation extraction | HF Transformers `output_hidden_states` | Custom vLLM operator / Triton kernel |
| Model serving | Per-request Gradio loading | Persistent vLLM / TGI with activation cache |
| Statistical testing | Descriptive (trends, averages) | Bootstrapped CIs, multiple comparison correction |
| Translation verification | FLORES cross-reference | Native speaker back-translation |
| Failure detection | Rule-based heuristics | ML-based classifier trained on real failures |
| Update cadence | One-time analysis | Automated regression on model updates |
| Latency budget | Unbounded (research) | < 50ms for routing decision |
