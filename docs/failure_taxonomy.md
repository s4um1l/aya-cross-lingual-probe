# Failure Taxonomy

Categorization of inputs where a regional Tiny Aya variant (Fire or Earth) underperforms the Base model in cross-lingual concept alignment. These categories identify systematic routing edge cases for the language router.

A "failure" is defined as: the regional variant's commitment layer is at least 4 layers later than Base's commitment layer for the same stimulus, indicating significantly worse cross-lingual concept alignment.

---

## 1. CODE_SWITCH

**Definition:** Input contains tokens from two or more languages within a single sentence. The model receives conflicting language signals and may oscillate between language-specific processing paths rather than committing to one.

**Why it causes routing failures:** Regional variants (Fire, Earth) are fine-tuned to expect monolingual input in their target languages. Mixed-language input activates competing language representations in early layers, delaying or preventing commitment to a stable cross-lingual representation. Base, having broader multilingual coverage without regional specialization, handles the ambiguity more gracefully.

**Example inputs:**
- `"My bachcha has bukhar since kal"` (English + Hindi)
- `"Le mtoto is très malade"` (English + Swahili + French)
- `"Doctor sahab ne kaha take medicine daily"` (Hindi + English)

**Recommended handling in the language router:** Detect mixed-language tokens using a script/language-ID classifier on sub-sentence spans. If more than one language is detected with confidence > 0.3, route to Base instead of a regional variant.

---

## 2. TRANSLITERATION

**Definition:** Text in one language written using the script of another language -- most commonly Hindi, Bengali, or Tamil written in Latin script (e.g., Hinglish, Romanized Bengali). The tokenizer processes the input as if it were a Latin-script language, but the semantic content belongs to a different language.

**Why it causes routing failures:** Transliterated text tokenizes very differently from native-script text. Fire (optimized for Devanagari Hindi) receives Latin-script tokens it was not fine-tuned on, producing activations that are neither English nor Hindi in the residual stream. The model may never commit to a cross-lingual representation because the token embeddings are out of distribution. Base handles this slightly better because it has seen more diverse tokenization patterns during pretraining.

**Example inputs:**
- `"mera bachcha bahut zyada sick ho gaya"` (Hinglish)
- `"amar shishur jor hoyeche"` (Romanized Bengali)
- `"kuzhanthaiku kaichal irukku"` (Romanized Tamil)

**Recommended handling in the language router:** Use a script detection preprocessor. If the detected script does not match the expected script for the detected language (e.g., Latin script + Hindi semantics), route to Base. Optionally, transliterate to native script before routing to the regional variant.

---

## 3. FORMAL_REGISTER

**Definition:** Input uses formal, technical, or medical terminology that diverges from the conversational register the regional variant was fine-tuned on. This includes WHO-standard medical terms, pharmaceutical names, and clinical language.

**Why it causes routing failures:** Regional variants like Fire and Earth are fine-tuned on community-level health content in the target languages -- conversational, CHW-style interactions. Formal medical terminology (e.g., "acute febrile illness" instead of "high fever") maps to English-dominant token embeddings even when the surrounding sentence is in Hindi or Swahili. The variant then partially processes the sentence as English, delaying commitment.

**Example inputs:**
- `"bacche mein acute respiratory distress syndrome ke lakshan hain"` (Hindi with English medical terms)
- `"mtoto ana dalili za severe acute malnutrition"` (Swahili with English clinical terms)
- `"le patient presente une insuffisance renale aigue"` (French medical register)

**Recommended handling in the language router:** Maintain a terminology watchlist of formal/clinical terms. If the input contains terms from the watchlist that are borrowed from English into the target language, prefer Base or add the formal terms to a domain-specific vocabulary for the regional variant. Long-term: fine-tune regional variants on formal medical corpora.

---

## 4. LOW_RESOURCE_VARIANT

**Definition:** Input is in a dialect, regional variant, or closely related language that the regional model groups with a higher-resource sibling. For example, Bhojpuri routed as Hindi, Sheng routed as Swahili, or Darija routed as Arabic.

**Why it causes routing failures:** The regional variant may have never seen the dialect during fine-tuning. It tokenizes the input using the closest language's vocabulary, but the resulting embeddings are noisy. Commitment layers shift later because the model's internal language detector cannot confidently classify the dialect, causing the residual stream to hover near (but not above) the commitment threshold.

**Example inputs:**
- `"hamaar bachcha ke bukhar baa"` (Bhojpuri, often classified as Hindi)
- `"mtoto amefall sick sana"` (Sheng -- Swahili/English urban slang)
- `"dereyet w'alad andek homa"` (Algerian Darija, classified as Arabic)
- `"pillaiki jvaram unnadi"` (Telugu, may be misrouted if close to Tamil)

**Recommended handling in the language router:** Add dialect/variant detection as a second-stage classifier after primary language ID. For known low-resource variants, route to Base rather than the regional variant. Build a mapping of (dialect -> recommended variant) based on empirical commitment layer analysis.

---

## 5. UNICODE_EDGE

**Definition:** Input contains unusual Unicode characteristics: mixed script directions (RTL + LTR), Devanagari or Arabic numerals mixed with Latin text, zero-width joiners/non-joiners, or other special characters that affect tokenization.

**Why it causes routing failures:** Tokenizers handle Unicode edge cases inconsistently. A sentence mixing Arabic (RTL) with English numbers (LTR) may be split into unexpected token boundaries. Zero-width characters can cause tokens to be split or merged in ways that produce out-of-distribution embeddings. These tokenization artifacts propagate through the residual stream, delaying or preventing the model from recognizing the underlying concept.

**Example inputs:**
- `"الطفل عمره 3 سنوات ولديه حمى"` (Arabic with Western Arabic numerals)
- `"बच्चे का तापमान १०३°F है"` (Hindi with Devanagari numerals + Latin degree symbol)
- `"omo naa\u200bni iba"` (Yoruba with zero-width space inside a word)
- `"குழந்தை\u200cக்கு 40°C காய்ச்சல்"` (Tamil with ZWNJ + Latin temperature)

**Recommended handling in the language router:** Normalize Unicode before routing: convert all numerals to ASCII, strip zero-width characters, normalize whitespace. If the input contains mixed script directions, identify the dominant script and route based on that. Add Unicode normalization (NFC) as a preprocessing step in the routing pipeline.
