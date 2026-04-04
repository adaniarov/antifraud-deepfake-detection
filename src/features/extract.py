"""
extract.py — Stage 6: hand-crafted feature extraction for classical ML.

Feature groups (directly from Chapter 1 of the thesis):

  A. STYLOMETRIC (§1.1 Лингвистические подходы)
     - Basic statistics: char/word/sentence counts, avg lengths
     - Function word density: conjunctions, prepositions, pronouns  [§1.1.1]
     - POS tag distributions: unigrams + selected bigrams           [§1.1.1, §1.1.3]
     - Imperative verb frequency                                     [§1.1.1]
     - Clause density (subordinate clause ratio)                     [§1.1.3]
     - Punctuation variability: !, ?, —, …, commas, colons          [§1.1.3]
     - LLM-marker patterns: 3-item enumerations, formal connectives  [§1.1.3]

  B. LEXICAL DIVERSITY (§1.1.3)
     - TTR  (Type-Token Ratio)
     - MTLD (Measure of Textual Lexical Diversity)
     - Yule's K

  C. PERPLEXITY + BURSTINESS (§1.2.2)
     - Mean perplexity over full text (GPT-2 as reference model)
     - Burstiness = Var( NLL per sentence )                         [§1.2.2]
     - Mean / std / min / max of per-sentence NLL

  D. TF-IDF VECTORS (§1.2.1 n-gram analysis)
     - Word n-grams (1–2), top 10 000 features
     - Char n-grams (2–4), top 10 000 features
     NOTE: TF-IDF matrices are saved separately (sparse), not in the
     dense feature matrix.

Outputs:
  data/features/
    train_features.parquet   — dense hand-crafted features (A+B+C), train
    val_features.parquet     — same for val
    test_features.parquet    — same for test
    feature_names.json       — ordered list of feature names
    tfidf_word/              — sklearn TF-IDF word model + matrices (.npz)
    tfidf_char/              — sklearn TF-IDF char model + matrices (.npz)

Usage:
    uv run python -m src.features.extract
    uv run python -m src.features.extract --no-perplexity   # skip slow GPU step
    uv run python -m src.features.extract --split train     # one split only
"""

import argparse
import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FINAL_DIR   = Path("data/final")
FEATURE_DIR = Path("data/features")

SPLITS = ["train", "val", "test"]

# ---------------------------------------------------------------------------
# Lazy model loading (avoid loading heavy models if not needed)
# ---------------------------------------------------------------------------

_spacy_nlp = None
_ppl_model = None
_ppl_tokenizer = None


def get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy  # type: ignore
        try:
            _spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except OSError:
            logger.error(
                "spaCy model 'en_core_web_sm' not found.\n"
                "Install it with:  uv run python -m spacy download en_core_web_sm"
            )
            raise
    return _spacy_nlp


def get_ppl_model():
    global _ppl_model, _ppl_tokenizer
    if _ppl_model is None:
        import torch  # type: ignore
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # type: ignore
        logger.info("Loading GPT-2 for perplexity calculation ...")
        _ppl_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        _ppl_model     = GPT2LMHeadModel.from_pretrained("gpt2")
        _ppl_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _ppl_model = _ppl_model.to(device)
        logger.info(f"  GPT-2 loaded on {device}")
    return _ppl_model, _ppl_tokenizer

# ---------------------------------------------------------------------------
# A. STYLOMETRIC FEATURES
# ---------------------------------------------------------------------------

# Function word lists (English)
CONJUNCTIONS  = {"and","but","or","nor","for","yet","so","although","because",
                 "since","while","whereas","if","unless","until","when","though"}
PREPOSITIONS  = {"in","on","at","to","for","of","with","by","from","about",
                 "as","into","through","during","before","after","above","below",
                 "between","out","off","over","under","again","further"}
PRONOUNS      = {"i","me","my","we","us","our","you","your","he","she","it",
                 "they","them","their","this","that","these","those","who","which"}

# LLM-typical formal connectives (§1.1.3 "шаблонные конструкции")
LLM_CONNECTIVES = {"furthermore","moreover","additionally","consequently",
                   "nevertheless","nonetheless","therefore","thus","hence",
                   "in conclusion","in summary","it is important","please note",
                   "kindly","herewith","hereby","pursuant","as per"}

PUNCT_CHARS = set("!?—…,:;")


def stylometric_features(text: str, nlp) -> dict:
    """
    Extract all stylometric features from a single text string.
    Requires a spaCy Doc for POS/syntax analysis.
    """
    f: dict = {}
    doc = nlp(text)

    tokens  = [t for t in doc if not t.is_space]
    words   = [t for t in tokens if t.is_alpha]
    sents   = list(doc.sents)

    n_tokens   = max(len(tokens), 1)
    n_words    = max(len(words), 1)
    n_sents    = max(len(sents), 1)
    n_chars    = len(text)

    # ── Basic statistics ──────────────────────────────────────────────────
    f["n_chars"]           = n_chars
    f["n_words"]           = n_words
    f["n_sentences"]       = n_sents
    f["avg_word_len"]      = np.mean([len(w.text) for w in words]) if words else 0.0
    f["avg_sent_len_words"]= n_words / n_sents
    f["avg_sent_len_chars"]= n_chars / n_sents
    f["std_sent_len_words"]= float(np.std([len([t for t in s if t.is_alpha])
                                           for s in sents]))

    # ── Function word densities ───────────────────────────────────────────
    lower_words = [w.lower_ for w in words]
    f["conj_density"]  = sum(1 for w in lower_words if w in CONJUNCTIONS) / n_words
    f["prep_density"]  = sum(1 for w in lower_words if w in PREPOSITIONS) / n_words
    f["pron_density"]  = sum(1 for w in lower_words if w in PRONOUNS) / n_words
    f["func_density"]  = f["conj_density"] + f["prep_density"] + f["pron_density"]

    # ── POS tag distribution ──────────────────────────────────────────────
    pos_tags = [t.pos_ for t in tokens]
    for tag in ["NOUN","VERB","ADJ","ADV","PRON","DET","ADP","CCONJ","SCONJ","NUM","PUNCT"]:
        f[f"pos_{tag.lower()}"] = pos_tags.count(tag) / n_tokens

    # ── POS bigrams (selected discriminative pairs from [4]) ─────────────
    pos_bigrams = list(zip(pos_tags[:-1], pos_tags[1:]))
    n_bigrams   = max(len(pos_bigrams), 1)
    for pair in [("VERB","NOUN"), ("NOUN","VERB"), ("ADJ","NOUN"),
                 ("ADP","NOUN"), ("SCONJ","VERB"), ("VERB","ADP")]:
        key = f"posbigram_{'_'.join(pair).lower()}"
        f[key] = pos_bigrams.count(pair) / n_bigrams

    # ── Imperative verb frequency (§1.1.1) ───────────────────────────────
    # Heuristic: VB (base form) at start of sentence or after punctuation
    imperative_count = 0
    for sent in sents:
        sent_tokens = [t for t in sent if not t.is_space and t.is_alpha]
        if sent_tokens and sent_tokens[0].tag_ == "VB":
            imperative_count += 1
    f["imperative_density"] = imperative_count / n_sents

    # ── Clause density / subordination (§1.1.3) ──────────────────────────
    # Proxy: SCONJ tokens per sentence
    sconj_count = pos_tags.count("SCONJ")
    f["clause_density"] = sconj_count / n_sents

    # ── Punctuation variability (§1.1.3) ─────────────────────────────────
    f["exclamation_rate"] = text.count("!") / n_sents
    f["question_rate"]    = text.count("?") / n_sents
    f["ellipsis_rate"]    = text.count("…") / n_sents + text.count("...") / n_sents
    f["dash_rate"]        = (text.count("—") + text.count("--")) / n_sents
    f["comma_rate"]       = text.count(",") / n_sents
    f["colon_rate"]       = text.count(":") / n_sents
    # Normalised punct diversity: unique punct chars used / total punct chars
    punct_chars_used = [c for c in text if c in PUNCT_CHARS]
    f["punct_diversity"]  = (len(set(punct_chars_used)) / len(PUNCT_CHARS)
                             if punct_chars_used else 0.0)

    # ── LLM-marker patterns (§1.1.3 "шаблонные конструкции") ────────────
    lower_text = text.lower()
    f["llm_connective_count"] = sum(1 for c in LLM_CONNECTIVES if c in lower_text)
    # 3-item enumerations proxy: "X, Y, and Z" / "X, Y, or Z"
    three_enum = len(re.findall(r'\b\w+,\s+\w+,?\s+(?:and|or)\s+\w+', lower_text))
    f["three_item_enum"] = three_enum / n_sents

    # ── Uppercase ratio (ALL CAPS urgency — phishing signal) ─────────────
    alpha_chars = [c for c in text if c.isalpha()]
    f["uppercase_ratio"] = (sum(1 for c in alpha_chars if c.isupper()) /
                            max(len(alpha_chars), 1))

    # ── URL marker presence (after preprocessing URLs become [URL]) ───────
    f["url_token_count"] = text.count("[URL]") / n_sents

    return f

# ---------------------------------------------------------------------------
# B. LEXICAL DIVERSITY
# ---------------------------------------------------------------------------

def ttr(words: list[str]) -> float:
    """Type-Token Ratio. Sensitive to text length — use on equal-length windows
    for comparison; here used as-is since texts are length-filtered."""
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def yule_k(words: list[str]) -> float:
    """
    Yule's K statistic — less length-sensitive than TTR.
    K = 10^4 * (Σ V(m,N) * m^2 - N) / N^2
    where V(m,N) = number of words appearing exactly m times.
    """
    if len(words) < 2:
        return 0.0
    from collections import Counter
    freq    = Counter(words)
    n       = len(words)
    freq_of_freq = Counter(freq.values())
    sumterm = sum(v * m * m for m, v in freq_of_freq.items())
    denom   = n * n
    if denom == 0:
        return 0.0
    return 1e4 * (sumterm - n) / denom


def mtld(words: list[str], threshold: float = 0.72) -> float:
    """
    MTLD — Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).
    Robust to text length; standard threshold = 0.72.
    """
    if len(words) < 10:
        return 0.0

    def _mtld_pass(word_list):
        types, tokens = set(), 0
        factor_count  = 0.0
        for w in word_list:
            types.add(w)
            tokens += 1
            current_ttr = len(types) / tokens
            if current_ttr <= threshold:
                factor_count += 1
                types, tokens = set(), 0
        # partial factor
        if tokens > 0:
            factor_count += (1.0 - current_ttr) / (1.0 - threshold)
        return len(word_list) / max(factor_count, 1e-9)

    forward  = _mtld_pass(words)
    backward = _mtld_pass(list(reversed(words)))
    return (forward + backward) / 2.0


def lexical_diversity_features(text: str) -> dict:
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return {
        "ttr":    ttr(words),
        "yule_k": yule_k(words),
        "mtld":   mtld(words),
    }

# ---------------------------------------------------------------------------
# C. PERPLEXITY + BURSTINESS
# ---------------------------------------------------------------------------

def perplexity_features(text: str, max_length: int = 512) -> dict:
    """
    Compute GPT-2 perplexity and burstiness features.

    Perplexity: PPL(x) = exp(-1/N Σ log p(xi | x<i))   [§1.2.2]
    Burstiness: Var({NLL(s_k)})                          [§1.2.2]

    Sentences shorter than 5 tokens are skipped for NLL calculation
    (perplexity is unreliable on very short sequences [23]).
    """
    import torch  # type: ignore
    model, tokenizer = get_ppl_model()
    device = next(model.parameters()).device

    f = {
        "ppl_mean":      -1.0,
        "ppl_sent_mean": -1.0,
        "ppl_sent_std":  -1.0,
        "ppl_sent_min":  -1.0,
        "ppl_sent_max":  -1.0,
        "burstiness":    -1.0,
    }

    # ── Full-text perplexity ──────────────────────────────────────────────
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_length).to(device)
        with torch.no_grad():
            loss = model(**enc, labels=enc["input_ids"]).loss
        f["ppl_mean"] = float(torch.exp(loss).cpu())
    except Exception as e:
        logger.debug(f"PPL full-text failed: {e}")

    # ── Per-sentence NLL (for burstiness) ────────────────────────────────
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
    if not sentences:
        return f

    nll_values = []
    for sent in sentences:
        try:
            enc = tokenizer(sent, return_tensors="pt", truncation=True,
                            max_length=128).to(device)
            if enc["input_ids"].shape[1] < 5:
                continue
            with torch.no_grad():
                loss = model(**enc, labels=enc["input_ids"]).loss
            nll_values.append(float(loss.cpu()))
        except Exception:
            continue

    if len(nll_values) >= 2:
        arr = np.array(nll_values)
        f["ppl_sent_mean"] = float(np.mean(arr))
        f["ppl_sent_std"]  = float(np.std(arr))
        f["ppl_sent_min"]  = float(np.min(arr))
        f["ppl_sent_max"]  = float(np.max(arr))
        f["burstiness"]    = float(np.var(arr))   # Var({NLL(s_k)}) as in §1.2.2

    return f

# ---------------------------------------------------------------------------
# Feature extraction pipeline (one record)
# ---------------------------------------------------------------------------

def extract_one(text: str, nlp, compute_ppl: bool = True) -> dict:
    features = {}
    features.update(stylometric_features(text, nlp))
    features.update(lexical_diversity_features(text))
    if compute_ppl:
        features.update(perplexity_features(text))
    return features


def extract_split(
    split: str,
    compute_ppl: bool = True,
    batch_log_every: int = 500,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load one split, extract all hand-crafted features.
    Returns (feature_df, label_series).
    """
    path = FINAL_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run assemble.py first.")

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(f"[{split}] {len(records):,} records — extracting features ...")

    nlp = get_spacy()
    all_features = []
    labels       = []

    for i, rec in enumerate(records):
        text = rec.get("text_clean") or rec.get("text", "")
        feats = extract_one(text, nlp, compute_ppl=compute_ppl)
        all_features.append(feats)
        labels.append(rec["label"])

        if (i + 1) % batch_log_every == 0:
            logger.info(f"  {split}: {i+1:,}/{len(records):,}")

    feature_df = pd.DataFrame(all_features)

    # Attach metadata columns for downstream use (not used as ML features)
    feature_df["label"]          = labels
    feature_df["content_type"]   = [r.get("content_type", "")  for r in records]
    feature_df["origin_model"]   = [r.get("origin_model") or "" for r in records]
    feature_df["dataset_source"] = [r.get("dataset_source", "") for r in records]
    feature_df["split"]          = split
    # Preserve _companion flag for partition-based test evaluation
    if split == "test":
        feature_df["_companion"] = [
            r.get("_companion", False) for r in records
        ]

    logger.success(f"[{split}] done — {len(feature_df):,} records, "
                   f"{len(feature_df.columns):,} columns")
    return feature_df, pd.Series(labels, name="label")

# ---------------------------------------------------------------------------
# TF-IDF matrices (D)
# ---------------------------------------------------------------------------

def build_tfidf(
    train_texts: list[str],
    val_texts:   list[str],
    test_texts:  list[str],
    mode: str = "word",
    max_features: int = 10_000,
    out_dir: Path = FEATURE_DIR,
) -> None:
    """
    Fit TF-IDF on train texts, transform all splits.
    mode = "word"  → word n-grams (1,2)
    mode = "char"  → char n-grams (2,4)
    """
    import scipy.sparse as sp  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    import joblib  # type: ignore

    if mode == "word":
        vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
            min_df=3,
            token_pattern=r"(?u)\b\w+\b",
        )
        subdir = out_dir / "tfidf_word"
    else:
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=max_features,
            sublinear_tf=True,
            min_df=3,
        )
        subdir = out_dir / "tfidf_char"

    subdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  TF-IDF [{mode}] — fitting on {len(train_texts):,} train texts ...")
    X_train = vec.fit_transform(train_texts)
    X_val   = vec.transform(val_texts)
    X_test  = vec.transform(test_texts)

    sp.save_npz(subdir / "train.npz", X_train)
    sp.save_npz(subdir / "val.npz",   X_val)
    sp.save_npz(subdir / "test.npz",  X_test)
    joblib.dump(vec, subdir / "vectorizer.joblib")

    logger.success(f"  TF-IDF [{mode}]: matrix shape train={X_train.shape}, "
                   f"val={X_val.shape}, test={X_test.shape}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(splits_to_run: list[str], compute_ppl: bool) -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}

    for split in splits_to_run:
        df, _ = extract_split(split, compute_ppl=compute_ppl)
        out_path = FEATURE_DIR / f"{split}_features.parquet"
        df.to_parquet(out_path, index=False)
        logger.success(f"Saved {out_path}  ({len(df):,} rows × {len(df.columns)} cols)")
        dfs[split] = df

    # Save feature names (columns that are actual features, not metadata)
    meta_cols = {"label", "content_type", "origin_model", "dataset_source", "split"}
    if dfs:
        first_df = next(iter(dfs.values()))
        feature_names = [c for c in first_df.columns if c not in meta_cols]
        feat_path = FEATURE_DIR / "feature_names.json"
        feat_path.write_text(json.dumps(feature_names, indent=2))
        logger.info(f"Feature names saved → {feat_path}  ({len(feature_names)} features)")

    # Build TF-IDF matrices if all three splits are available
    if set(splits_to_run) == {"train", "val", "test"}:
        logger.info("\nBuilding TF-IDF matrices ...")

        def _texts(split):
            path = FINAL_DIR / f"{split}.jsonl"
            texts = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        texts.append(r.get("text_clean") or r.get("text", ""))
            return texts

        train_t = _texts("train")
        val_t   = _texts("val")
        test_t  = _texts("test")

        build_tfidf(train_t, val_t, test_t, mode="word", out_dir=FEATURE_DIR)
        build_tfidf(train_t, val_t, test_t, mode="char", out_dir=FEATURE_DIR)
    else:
        logger.warning("TF-IDF skipped — need all three splits to build matrices.")

    print("\n" + "=" * 56)
    print("  FEATURE EXTRACTION COMPLETE")
    print("=" * 56)
    print(f"  Output directory : {FEATURE_DIR}")
    if dfs:
        first_df = next(iter(dfs.values()))
        feature_names = [c for c in first_df.columns if c not in meta_cols]
        print(f"  Hand-crafted features : {len(feature_names)}")
        ppl_features = [f for f in feature_names if "ppl" in f or "burst" in f]
        print(f"    Stylometric + lexical : {len(feature_names) - len(ppl_features)}")
        print(f"    Perplexity/burstiness : {len(ppl_features)}")
    print("=" * 56)
    print("\nNext: uv run python -m src.models.train_classical\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hand-crafted features (Stage 6)")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split(s) to process (default: all)",
    )
    parser.add_argument(
        "--no-perplexity",
        action="store_true",
        help="Skip GPT-2 perplexity/burstiness features (much faster, no GPU needed)",
    )
    args = parser.parse_args()

    splits = SPLITS if args.split == "all" else [args.split]
    
    main(splits_to_run=splits, compute_ppl=not args.no_perplexity)