# HC60 — hand-crafted feature set (Core v2)

**Module:** `v2/src/core_hc60_features.py`  
**Count:** exactly **60** numeric features, prefix `hc60_`.  
**Design:** independent of legacy `hc_*` / TF-IDF; English-oriented regex/lexicon proxies; safe on empty/short text (zeros / bounded values).

## Groups (see `HC60_GROUP_INDICES`)

| Group | Index range | Role |
|-------|-------------|------|
| length_structure | 0–9 | Length, sentence/word stats |
| surface | 10–16 | Character composition |
| punctuation | 17–25 | Punctuation-type densities (per character) |
| lexical_diversity | 26–35 | TTR variants, Yule K, repetition |
| functional_proxies | 36–50 | Regex/lexicon ratios (per token) |
| readability_discourse | 51–59 | textstat readability + stopwords / burstiness / discourse |

## Tokenization

- **Words:** ASCII letters + apostrophe inside word (`_WORD_RE`).
- **Sentences:** split on `.` `!` `?` runs (heuristic; OK for email/SMS/QA mix).

## Readability

Uses `textstat` when installed; on failure or empty text → zeros.

## NLTK stopwords

English stopword list from `nltk.corpus.stopwords`; if unavailable, empty set (stopword ratio → 0).

## Reproducibility

Feature extraction is **deterministic** given text (no randomness). Downstream training uses `random_state=42` in experiment suite.
