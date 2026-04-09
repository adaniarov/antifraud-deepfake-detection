"""
Hand-crafted baseline features for Core v2 (HC60).

Independent implementation: no `hc_*` legacy module, no TF-IDF.
Exactly 60 numeric features; stable for short/empty text (zeros / defined limits).

Spec: `v2/docs/hc60_feature_spec.md`
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

try:
    import textstat
except ImportError:  # pragma: no cover
    textstat = None

try:
    import nltk
    from nltk.corpus import stopwords

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:  # pragma: no cover
        nltk.download("stopwords", quiet=True)
    _stop = frozenset(stopwords.words("english"))
except Exception:  # pragma: no cover
    _stop = frozenset()

# --- 60 names (order fixed) -------------------------------------------------
HC60_FEATURE_NAMES: tuple[str, ...] = (
    # 1–10 length & structure
    "hc60_char_len",
    "hc60_word_count",
    "hc60_sentence_count",
    "hc60_avg_sentence_len_words",
    "hc60_std_sentence_len_words",
    "hc60_avg_word_len_chars",
    "hc60_std_word_len_chars",
    "hc60_long_word_ratio",
    "hc60_short_word_ratio",
    "hc60_chars_per_word",
    # 11–17 surface
    "hc60_digit_ratio",
    "hc60_uppercase_ratio",
    "hc60_whitespace_ratio",
    "hc60_punctuation_ratio",
    "hc60_char_entropy",
    "hc60_alnum_ratio",
    "hc60_non_alnum_symbol_ratio",
    # 18–26 punctuation profile
    "hc60_period_ratio",
    "hc60_comma_ratio",
    "hc60_colon_ratio",
    "hc60_semicolon_ratio",
    "hc60_dash_ratio",
    "hc60_quote_ratio",
    "hc60_paren_ratio",
    "hc60_question_ratio",
    "hc60_exclamation_ratio",
    # 27–36 lexical diversity & repetition
    "hc60_ttr",
    "hc60_corrected_ttr",
    "hc60_root_ttr",
    "hc60_hapax_ratio",
    "hc60_dislegomena_ratio",
    "hc60_yule_k",
    "hc60_repetition_rate",
    "hc60_top_token_share",
    "hc60_entropy_token_unigram",
    "hc60_avg_token_length_chars",
    # 37–51 functional / grammatical proxies (regex lexicons)
    "hc60_pronoun_ratio",
    "hc60_modal_ratio",
    "hc60_auxiliary_ratio",
    "hc60_article_ratio",
    "hc60_preposition_ratio",
    "hc60_conjunction_ratio",
    "hc60_det_like_ratio",
    "hc60_numeral_token_ratio",
    "hc60_verb_like_ratio",
    "hc60_noun_like_ratio",
    "hc60_adjective_like_ratio",
    "hc60_adverb_like_ratio",
    "hc60_negation_ratio",
    "hc60_question_word_ratio",
    "hc60_imperative_like_ratio",
    # 52–56 readability (textstat; zeros if unavailable)
    "hc60_flesch_reading_ease",
    "hc60_flesch_kincaid_grade",
    "hc60_smog_index",
    "hc60_coleman_liau_index",
    "hc60_automated_readability_index",
    # 57–60 discourse / burstiness proxies
    "hc60_stopword_ratio",
    "hc60_sentence_opener_repeat_rate",
    "hc60_lexical_burstiness",
    "hc60_discourse_marker_ratio",
)

assert len(HC60_FEATURE_NAMES) == 60

# Column index ranges (for ablations); end is exclusive.
HC60_GROUP_INDICES: dict[str, tuple[int, int]] = {
    "length_structure": (0, 10),
    "surface": (10, 17),
    "punctuation": (17, 26),
    "lexical_diversity": (26, 36),
    "functional_proxies": (36, 51),
    "readability_discourse": (51, 60),
}

_WORD_RE = re.compile(r"(?u)\b[a-zA-Z][a-zA-Z']*\b")
_SENT_SPLIT = re.compile(r"[.!?]+")
_NUM_TOKEN_RE = re.compile(r"(?u)\b\d+(?:[.,]\d+)*\b")

_PRON = re.compile(
    r"\b(i|me|my|mine|myself|you|your|yours|yourself|he|him|his|himself|she|her|hers|herself|"
    r"we|us|our|ours|ourselves|they|them|their|theirs|themselves|it|its|itself)\b",
    re.I,
)
_MODAL = re.compile(
    r"\b(can|could|may|might|must|shall|should|will|would|ought)\b", re.I
)
_AUX = re.compile(
    r"\b(am|is|are|was|were|be|been|being|have|has|had|do|does|did|get|got|getting)\b", re.I
)
_ART = re.compile(r"\b(a|an|the)\b", re.I)
_PREP = re.compile(
    r"\b(in|on|at|by|for|with|from|to|of|about|into|through|after|before|between|under|over)\b",
    re.I,
)
_CONJ = re.compile(r"\b(and|or|but|nor|so|yet|because|although|though|if|unless)\b", re.I)
_DET = re.compile(
    r"\b(this|that|these|those|some|any|each|every|no|another|such|what|which)\b", re.I
)
_VERB_LIKE = re.compile(
    r"\b(is|are|was|were|be|been|being|have|has|had|do|does|did|go|goes|went|gone|make|made|"
    r"take|took|taken|come|came|see|saw|seen|know|knew|known|think|thought|get|got|give|gave|"
    r"find|found|tell|told|ask|asked|work|worked|seem|seemed|feel|felt|try|tried|leave|left|"
    r"call|called)\b",
    re.I,
)
_NOUN_LIKE = re.compile(
    r"\b(time|year|people|way|day|man|woman|child|world|life|hand|part|place|case|week|company|"
    r"system|program|question|work|number|point|home|water|room|fact|story|result|change|"
    r"government|money|issue|service|market|problem|study|lot|right|book|business|issue)\b",
    re.I,
)
_ADJ_LIKE = re.compile(
    r"\b(new|good|high|old|great|big|small|large|long|little|own|other|last|first|next|"
    r"early|young|important|few|public|bad|same|able)\b",
    re.I,
)
_ADV_LIKE = re.compile(
    r"\b(not|very|also|just|only|well|even|back|there|however|therefore|still|already|"
    r"never|always|often|sometimes|today|now|here|away|again|once|too|quite|rather|almost)\b",
    re.I,
)
_NEG = re.compile(r"\b(no|not|never|neither|nor|n't)\b", re.I)
_QW = re.compile(r"\b(who|whom|whose|what|which|where|when|why|how)\b", re.I)
_IMP = re.compile(r"^(please\s+)?[a-z]{3,}\b", re.I)

_DISCOURSE = re.compile(
    r"\b(therefore|however|moreover|furthermore|nevertheless|meanwhile|otherwise|"
    r"consequently|thus|hence|additionally|specifically|finally|first|second|third|"
    r"in conclusion|for example|in particular|on the other hand)\b",
    re.I,
)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _char_entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    h = 0.0
    for ct in c.values():
        p = ct / n
        h -= p * math.log2(p)
    return h


def _yule_k(words: list[str]) -> float:
    if len(words) < 2:
        return 0.0
    freq = Counter(words)
    vi = Counter()
    for v in freq.values():
        vi[v] += 1
    m1 = sum(f * n for f, n in vi.items())
    m2 = sum((f**2) * n for f, n in vi.items())
    if m1 <= 0 or m1 * m1 == 0:
        return 0.0
    return 10000.0 * (m2 - m1) / (m1 * m1)


def _readability_feats(text: str) -> tuple[float, float, float, float, float]:
    if textstat is None or not (text or "").strip():
        return 0.0, 0.0, 0.0, 0.0, 0.0
    t = text
    try:
        fre = float(textstat.flesch_reading_ease(t))
        fk = float(textstat.flesch_kincaid_grade(t))
        sm = float(textstat.smog_index(t))
        cl = float(textstat.coleman_liau_index(t))
        ari = float(textstat.automated_readability_index(t))
        return fre, fk, sm, cl, ari
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0


def extract_hc60(text: str | None) -> dict[str, float]:
    """Return dict of exactly 60 floats (keys in HC60_FEATURE_NAMES)."""
    t = text if isinstance(text, str) else ""
    n_char = len(t)
    words = _WORD_RE.findall(t.lower())
    n_w = len(words)
    sents = [x.strip() for x in _SENT_SPLIT.split(t) if x.strip()]
    n_s = max(len(sents), 1)
    sent_word_counts = [len(_WORD_RE.findall(x.lower())) for x in sents]
    sw_arr = np.array(sent_word_counts, dtype=float) if sent_word_counts else np.array([0.0])

    wl = [len(w) for w in words] if words else [0]
    wl_arr = np.array(wl, dtype=float)

    long_r = _safe_div(sum(1 for L in wl if L >= 7), n_w)
    short_r = _safe_div(sum(1 for L in wl if L <= 3), n_w)

    if n_char > 0:
        digit_r = sum(c.isdigit() for c in t) / n_char
        upper_r = sum(c.isupper() for c in t) / n_char
        space_r = sum(c.isspace() for c in t) / n_char
        punct_r = sum(1 for c in t if not c.isalnum() and not c.isspace()) / n_char
        alnum_r = sum(c.isalnum() for c in t) / n_char
        sym_r = sum(1 for c in t if not c.isalnum() and not c.isspace()) / n_char
    else:
        digit_r = upper_r = space_r = punct_r = alnum_r = sym_r = 0.0

    ent_c = _char_entropy(t)

    def pr(pat: str) -> float:
        return _safe_div(len(re.findall(pat, t)), n_char)

    period_r = pr(r"\.")
    comma_r = pr(r",")
    colon_r = pr(r":")
    semi_r = pr(r";")
    dash_r = pr(r"[-–—]")
    quote_r = pr(r"['\"“”‘’]")
    paren_r = pr(r"[\(\)\[\]\{\}]")
    qmark_r = pr(r"\?")
    excl_r = pr(r"!")

    types = len(set(words))
    ttr = _safe_div(types, n_w)
    if n_w > 1:
        cttr = math.log(types + 1) / math.log(n_w + 1)
    else:
        cttr = 0.0
    roots = [w[:4] for w in words if w]
    root_ttr = _safe_div(len(set(roots)), len(roots)) if roots else 0.0
    fc = Counter(words)
    hapax = sum(1 for v in fc.values() if v == 1)
    disl = sum(1 for v in fc.values() if v == 2)
    hapax_r = _safe_div(hapax, n_w)
    disl_r = _safe_div(disl, n_w)
    yk = _yule_k(words)
    if n_w:
        top_share = max(fc.values()) / n_w
        rep = 1.0 - hapax_r
        probs = np.array(list(fc.values()), dtype=float) / n_w
        h_tok = float(-(probs * np.log2(probs + 1e-12)).sum())
    else:
        top_share = rep = h_tok = 0.0

    avg_tok_len = float(wl_arr.mean()) if n_w else 0.0

    def wrx(rx: re.Pattern[str]) -> float:
        return _safe_div(len(rx.findall(t)), n_w)

    out: dict[str, float] = {
        "hc60_char_len": float(n_char),
        "hc60_word_count": float(n_w),
        "hc60_sentence_count": float(len(sents) if sents else 0),
        "hc60_avg_sentence_len_words": float(sw_arr.mean()),
        "hc60_std_sentence_len_words": float(sw_arr.std(ddof=0)),
        "hc60_avg_word_len_chars": float(wl_arr.mean()) if n_w else 0.0,
        "hc60_std_word_len_chars": float(wl_arr.std(ddof=0)) if n_w else 0.0,
        "hc60_long_word_ratio": long_r,
        "hc60_short_word_ratio": short_r,
        "hc60_chars_per_word": _safe_div(n_char, n_w),
        "hc60_digit_ratio": digit_r,
        "hc60_uppercase_ratio": upper_r,
        "hc60_whitespace_ratio": space_r,
        "hc60_punctuation_ratio": punct_r,
        "hc60_char_entropy": ent_c,
        "hc60_alnum_ratio": alnum_r,
        "hc60_non_alnum_symbol_ratio": sym_r,
        "hc60_period_ratio": period_r,
        "hc60_comma_ratio": comma_r,
        "hc60_colon_ratio": colon_r,
        "hc60_semicolon_ratio": semi_r,
        "hc60_dash_ratio": dash_r,
        "hc60_quote_ratio": quote_r,
        "hc60_paren_ratio": paren_r,
        "hc60_question_ratio": qmark_r,
        "hc60_exclamation_ratio": excl_r,
        "hc60_ttr": ttr,
        "hc60_corrected_ttr": cttr,
        "hc60_root_ttr": root_ttr,
        "hc60_hapax_ratio": hapax_r,
        "hc60_dislegomena_ratio": disl_r,
        "hc60_yule_k": float(yk),
        "hc60_repetition_rate": rep,
        "hc60_top_token_share": top_share,
        "hc60_entropy_token_unigram": h_tok,
        "hc60_avg_token_length_chars": avg_tok_len,
        "hc60_pronoun_ratio": wrx(_PRON),
        "hc60_modal_ratio": wrx(_MODAL),
        "hc60_auxiliary_ratio": wrx(_AUX),
        "hc60_article_ratio": wrx(_ART),
        "hc60_preposition_ratio": wrx(_PREP),
        "hc60_conjunction_ratio": wrx(_CONJ),
        "hc60_det_like_ratio": wrx(_DET),
        "hc60_numeral_token_ratio": _safe_div(len(_NUM_TOKEN_RE.findall(t)), n_w),
        "hc60_verb_like_ratio": wrx(_VERB_LIKE),
        "hc60_noun_like_ratio": wrx(_NOUN_LIKE),
        "hc60_adjective_like_ratio": wrx(_ADJ_LIKE),
        "hc60_adverb_like_ratio": wrx(_ADV_LIKE),
        "hc60_negation_ratio": wrx(_NEG),
        "hc60_question_word_ratio": wrx(_QW),
        "hc60_imperative_like_ratio": _safe_div(
            sum(1 for s in sents if s and _IMP.match(s.strip())), n_s
        ),
        "hc60_stopword_ratio": _safe_div(sum(1 for w in words if w in _stop), n_w),
        "hc60_discourse_marker_ratio": wrx(_DISCOURSE),
    }

    fre, fk, sm, cl, ari = _readability_feats(t)
    out["hc60_flesch_reading_ease"] = fre
    out["hc60_flesch_kincaid_grade"] = fk
    out["hc60_smog_index"] = sm
    out["hc60_coleman_liau_index"] = cl
    out["hc60_automated_readability_index"] = ari

    if len(sent_word_counts) >= 2:
        mu = float(np.mean(sent_word_counts))
        burst = float(np.std(sent_word_counts, ddof=0) / mu) if mu > 0 else 0.0
    else:
        burst = 0.0
    out["hc60_lexical_burstiness"] = burst

    if sents:
        first2 = []
        for s in sents:
            fw = _WORD_RE.findall(s.lower())
            first2.append(fw[0] if fw else "")
        rep_f = 1.0 - (len(set(first2)) / len(first2)) if first2 else 0.0
    else:
        rep_f = 0.0
    out["hc60_sentence_opener_repeat_rate"] = rep_f

    assert set(out.keys()) == set(HC60_FEATURE_NAMES)
    return {k: float(out[k]) for k in HC60_FEATURE_NAMES}


def row_to_hc60_vector(row: dict[str, Any]) -> dict[str, float]:
    return extract_hc60(str(row.get("text", "") or ""))
