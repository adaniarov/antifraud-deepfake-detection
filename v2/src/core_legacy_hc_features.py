"""
Legacy v1-style hand-crafted stylometric features (68), aligned with
`notebooks/02_features/03_feature_engineering.ipynb`, implemented with NLTK
(no spaCy) for v2 lockfile compatibility.

Groups: lexical surface (12), POS ratios (17), function-word ratios (20),
punctuation ratios (10), lexical complexity (9).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

FUNCTION_WORDS = [
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "shall",
    "should",
]

POS_TAGS = [
    "NOUN",
    "VERB",
    "ADJ",
    "ADV",
    "PRON",
    "DET",
    "ADP",
    "CONJ",
    "CCONJ",
    "SCONJ",
    "NUM",
    "PART",
    "INTJ",
    "AUX",
    "PROPN",
    "PUNCT",
    "SYM",
]

PUNCT_CHARS = [".", ",", "!", "?", ":", ";", "-", "'", '"', "("]

PUNCT_SAFE_NAMES = {
    ".": "period",
    ",": "comma",
    "!": "excl",
    "?": "question",
    ":": "colon",
    ";": "semicolon",
    "-": "dash",
    "'": "apos",
    '"': "dquote",
    "(": "paren",
}


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _penn_to_univ(pos: str, tok: str) -> str | None:
    if not tok or len(tok) == 1 and not tok.isalnum():
        if tok in PUNCT_CHARS or tok in ".?!,:;\"'()-[]{}":
            return "PUNCT"
    if pos in (".", ",", ":", "''", "``", "-LRB-", "-RRB-", "''"):
        return "PUNCT"
    if pos == "SYM":
        return "SYM"
    if pos.startswith("NN"):
        return "NOUN"
    if pos == "MD":
        return "AUX"
    if pos.startswith("VB"):
        return "VERB"
    if pos.startswith("JJ"):
        return "ADJ"
    if pos.startswith("RB") or pos == "WRB":
        return "ADV"
    if pos == "WDT":
        return "DET"
    if pos.startswith("PRP") or pos in ("WP", "WP$", "EX"):
        return "PRON"
    if pos in ("DT", "PDT"):
        return "DET"
    if pos == "IN":
        return "ADP"
    if pos == "CC":
        return "CCONJ"
    if pos in ("TO",):
        return "PART"
    if pos == "UH":
        return "INTJ"
    if pos == "RP":
        return "PART"
    if pos.startswith("NNP"):
        return "PROPN"
    if pos == "CD":
        return "NUM"
    if pos.startswith("FW"):
        return "X"
    return None


def compute_ttr(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_corrected_ttr(tokens: list[str]) -> float:
    n = len(tokens)
    if n == 0:
        return 0.0
    return len(set(tokens)) / np.sqrt(2 * n)


def compute_hapax_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    hapax = sum(1 for v in freq.values() if v == 1)
    return hapax / len(tokens)


def compute_yule_k(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    n = len(tokens)
    spectrum = Counter(freq.values())
    m2 = sum(i * i * spectrum[i] for i in spectrum)
    if n <= 1:
        return 0.0
    return 10000 * (m2 - n) / (n * n)


def compute_mtld(tokens: list[str], threshold: float = 0.72) -> float:
    if len(tokens) < 10:
        return 0.0

    def _mtld_one_direction(tok_list: list[str]) -> float:
        factors = 0.0
        seen: set[str] = set()
        segment_len = 0
        for t in tok_list:
            seen.add(t)
            segment_len += 1
            ttr_val = len(seen) / segment_len
            if ttr_val <= threshold:
                factors += 1.0
                seen = set()
                segment_len = 0
        if segment_len > 0:
            current_ttr = len(seen) / segment_len
            factors += (1.0 - current_ttr) / (1.0 - threshold) if threshold < 1.0 else 0.0
        return _safe_div(len(tok_list), factors)

    forward = _mtld_one_direction(tokens)
    backward = _mtld_one_direction(tokens[::-1])
    return (forward + backward) / 2.0


def extract_legacy_hc_row(text: str) -> dict[str, float]:
    """One row of 68 features (same semantics as v1 HC block)."""
    from nltk import pos_tag
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import sent_tokenize, word_tokenize

    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text
    try:
        stops = set(stopwords.words("english"))
    except LookupError:
        stops = set()

    try:
        wnl = WordNetLemmatizer()
    except LookupError:
        wnl = None

    toks = word_tokenize(text)
    n_tokens = len(toks)
    tokens_alpha = [t.lower() for t in toks if t.isalpha()]
    n_alpha = len(tokens_alpha)
    n_chars = len(text)
    try:
        sents = sent_tokenize(text)
    except Exception:
        sents = [text] if text else []
    n_sents = max(len(sents), 1)

    feat: dict[str, float] = {}

    feat["n_chars"] = float(n_chars)
    feat["n_tokens"] = float(n_tokens)
    feat["n_alpha_tokens"] = float(n_alpha)
    feat["n_sentences"] = float(len(sents)) if sents else 0.0
    feat["avg_word_len"] = _safe_div(sum(len(t) for t in tokens_alpha), max(n_alpha, 1))
    feat["avg_sentence_len_words"] = _safe_div(n_tokens, n_sents)
    feat["avg_sentence_len_chars"] = _safe_div(n_chars, n_sents)

    word_lengths = [len(t) for t in tokens_alpha]
    feat["std_word_len"] = float(np.std(word_lengths)) if word_lengths else 0.0
    feat["max_word_len"] = float(max(word_lengths)) if word_lengths else 0.0

    sent_lens = [len(word_tokenize(s)) for s in sents] if sents else [0]
    feat["std_sentence_len"] = float(np.std(sent_lens)) if sent_lens else 0.0

    feat["uppercase_ratio"] = _safe_div(sum(1 for c in text if c.isupper()), max(n_chars, 1))
    feat["digit_ratio"] = _safe_div(sum(1 for c in text if c.isdigit()), max(n_chars, 1))

    pos_counts: Counter[str] = Counter()
    if n_tokens > 0:
        try:
            tagged = pos_tag(toks, lang="eng")
        except TypeError:
            tagged = pos_tag(toks)
        for w, tag in tagged:
            u = _penn_to_univ(tag, w)
            if u in POS_TAGS:
                pos_counts[u] += 1
    for pos in POS_TAGS:
        feat[f"pos_{pos.lower()}_ratio"] = _safe_div(pos_counts.get(pos, 0), max(n_tokens, 1))

    lower_tokens = [t.lower() for t in toks]
    token_freq = Counter(lower_tokens)
    for fw in FUNCTION_WORDS:
        feat[f"fw_{fw}"] = _safe_div(token_freq.get(fw, 0), max(n_tokens, 1))

    char_freq = Counter(text)
    for pc in PUNCT_CHARS:
        safe_name = PUNCT_SAFE_NAMES[pc]
        feat[f"punct_{safe_name}_ratio"] = _safe_div(char_freq.get(pc, 0), max(n_chars, 1))

    feat["ttr"] = compute_ttr(tokens_alpha)
    feat["corrected_ttr"] = compute_corrected_ttr(tokens_alpha)
    feat["hapax_ratio"] = compute_hapax_ratio(tokens_alpha)
    feat["yule_k"] = compute_yule_k(tokens_alpha)
    feat["mtld"] = compute_mtld(tokens_alpha)

    lemmas: list[str] = []
    if wnl and tokens_alpha:
        try:
            tagged_alpha = pos_tag(tokens_alpha, lang="eng")
        except TypeError:
            tagged_alpha = pos_tag(tokens_alpha)

        def _wn_tag(t: str) -> str:
            if t.startswith("NN"):
                return "n"
            if t.startswith("VB") or t == "MD":
                return "v"
            if t.startswith("JJ"):
                return "a"
            if t.startswith("RB"):
                return "r"
            return "n"

        for w, t in tagged_alpha:
            try:
                lemmas.append(wnl.lemmatize(w, pos=_wn_tag(t)))
            except Exception:
                lemmas.append(w)
    else:
        lemmas = list(tokens_alpha)

    feat["avg_lemma_len"] = _safe_div(sum(len(lem) for lem in lemmas), max(len(lemmas), 1))
    feat["lemma_ttr"] = compute_ttr(lemmas)

    n_stop = sum(1 for t in lower_tokens if t in stops)
    feat["stopword_ratio"] = _safe_div(n_stop, max(n_tokens, 1))
    feat["short_word_ratio"] = _safe_div(sum(1 for t in tokens_alpha if len(t) <= 3), max(n_alpha, 1))

    return feat


LEGACY_HC_NAMES: tuple[str, ...] = tuple(
    [
        "n_chars",
        "n_tokens",
        "n_alpha_tokens",
        "n_sentences",
        "avg_word_len",
        "avg_sentence_len_words",
        "avg_sentence_len_chars",
        "std_word_len",
        "max_word_len",
        "std_sentence_len",
        "uppercase_ratio",
        "digit_ratio",
    ]
    + [f"pos_{p.lower()}_ratio" for p in POS_TAGS]
    + [f"fw_{fw}" for fw in FUNCTION_WORDS]
    + [f"punct_{PUNCT_SAFE_NAMES[pc]}_ratio" for pc in PUNCT_CHARS]
    + [
        "ttr",
        "corrected_ttr",
        "hapax_ratio",
        "yule_k",
        "mtld",
        "avg_lemma_len",
        "lemma_ttr",
        "stopword_ratio",
        "short_word_ratio",
    ]
)

assert len(LEGACY_HC_NAMES) == 68


def legacy_hc_row_to_ordered(row: dict[str, float]) -> list[float]:
    return [float(row.get(k, 0.0)) for k in LEGACY_HC_NAMES]


def add_hc_prefix(name: str) -> str:
    return f"hc_{name}"
