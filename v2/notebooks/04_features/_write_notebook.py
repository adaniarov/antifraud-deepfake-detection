"""One-shot generator for 01_core_feature_extraction_and_eda.ipynb — run from v2/: uv run python notebooks/04_features/_write_notebook.py"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "01_core_feature_extraction_and_eda.ipynb"


def cell_md(text: str) -> dict:
    lines = text.strip().split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else ["\n"])
    return {"cell_type": "markdown", "id": f"md-{hash(text) % 10**8}", "metadata": {}, "source": src}


def cell_code(text: str, cid: str) -> dict:
    src = []
    for line in text.split("\n"):
        src.append(line + "\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


cells: list[dict] = []

cells.append(
    cell_md(
        r"""# Core v2 — извлечение признаков и глубокий EDA

**Цель:** построить классические и продвинутые текстовые признаки по замороженному Core, сохранить артефакты и провести диагностический анализ (не финальный production-baseline).

**Данные:** `v2/data/interim/assembled/core_*.jsonl` (см. `core_manifest.json`, `core_dataset_description.md`).

**Протокол отчётности (метрики моделей — отдельно):** для интерпретации признаков полезно смотреть срезы **val**, **test_non_claude** (`test_seen`), **test_claude_only**; `test_full` — только как агрегат.

**Тексты:** использовать поле `text` как в Core; не вводить новую нормализацию, ломающую сравнимость human/LLM.

**Зависимости:** базовый v2 env; **NLTK** (punkt, POS-tagger, wordnet, stopwords) для **68 legacy HC-признаков** в духе v1; для LM-perplexity: `uv sync --extra lm_scoring` (torch + transformers)."""
    )
)

cells.append(
    cell_code(
        r'''from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["figure.dpi"] = 120

SEED = 42
np.random.seed(SEED)


def resolve_v2_base() -> Path:
    cur = Path.cwd().resolve()
    for p in [cur, *cur.parents]:
        cand = p / "v2" / "pyproject.toml"
        if cand.is_file():
            return p / "v2"
        if p.name == "v2" and (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError("Запустите из репозитория (нужен v2/pyproject.toml).")


BASE = resolve_v2_base()
ASSEMBLED = BASE / "data" / "interim" / "assembled"
FEATURE_DIR = BASE / "data" / "interim" / "features"
FIG_DIR = BASE / "outputs" / "figures" / "features"
TABLE_DIR = BASE / "outputs" / "tables" / "features"

for d in (FEATURE_DIR, FIG_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

print("BASE:", BASE)
print("FEATURE_DIR:", FEATURE_DIR)''',
        "cfg",
    )
)

cells.append(cell_md("## Загрузка Core"))

cells.append(
    cell_code(
        r'''def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


paths = {
    "train": ASSEMBLED / "core_train.jsonl",
    "val": ASSEMBLED / "core_val.jsonl",
    "test_non_claude": ASSEMBLED / "core_test_non_claude.jsonl",
    "test_claude_only": ASSEMBLED / "core_test_claude_only.jsonl",
    "test_full": ASSEMBLED / "core_test_full.jsonl",
}
for k, p in paths.items():
    if not p.is_file():
        raise FileNotFoundError(p)

parts = []
for name, p in paths.items():
    recs = load_jsonl(p)
    df = pd.DataFrame(recs)
    df["_load_split"] = name
    parts.append(df)

df_all = pd.concat(parts, ignore_index=True)
# одна строка могла бы продублироваться только если пересечение файлов — Core сплиты не пересекаются
df_all = df_all.drop_duplicates(subset=["core_row_id"], keep="first")

need = {"text", "label", "split", "core_eval_slice", "scenario_family", "channel", "fraudness"}
miss = need - set(df_all.columns)
if miss:
    raise ValueError(f"Missing columns: {miss}")

print("rows:", len(df_all))
print(df_all.groupby(["_load_split", "core_eval_slice"]).size().head(20))''',
        "load",
    )
)

cells.append(
    cell_md(
        """## Классические dense-признаки

Лексика поверхности, readability (`textstat`), энтропия символов, простые TTR; без spaCy (соответствие v2 lockfile)."""
    )
)

cells.append(
    cell_code(
        r'''import re
import string
import textstat
import tiktoken

try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import word_tokenize
except Exception:
    word_tokenize = None

_enc = tiktoken.get_encoding("cl100k_base")


def shannon_entropy_chars(s: str) -> float:
    s = s.lower()
    if not s:
        return np.nan
    from collections import Counter

    c = Counter(s)
    n = len(s)
    h = 0.0
    for v in c.values():
        p = v / n
        h -= p * np.log2(p)
    return float(h)


def extract_dense_row(text: str) -> dict:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    t = text
    nch = max(len(t), 1)
    words = t.split()
    nw = len(words)
    feats = {
        "char_len": len(t),
        "word_len_ws": nw,
        "tiktoken_len": len(_enc.encode(t)),
        "digit_ratio": sum(c.isdigit() for c in t) / nch,
        "upper_ratio": sum(c.isupper() for c in t) / nch,
        "punct_ratio": sum(c in string.punctuation for c in t) / nch,
        "space_ratio": sum(c.isspace() for c in t) / nch,
        "mean_word_len": float(np.mean([len(w) for w in words])) if words else np.nan,
        "char_entropy": shannon_entropy_chars(t),
    }
    if word_tokenize:
        try:
            toks = word_tokenize(t.lower())
            ul = len(set(toks))
            feats["ttr"] = ul / len(toks) if toks else np.nan
        except Exception:
            feats["ttr"] = np.nan
    else:
        feats["ttr"] = len(set(words)) / nw if nw else np.nan

    for name, fn in [
        ("flesch_reading_ease", textstat.flesch_reading_ease),
        ("flesch_kincaid_grade", textstat.flesch_kincaid_grade),
        ("smog_index", textstat.smog_index),
        ("coleman_liau_index", textstat.coleman_liau_index),
    ]:
        try:
            if nw < 3:
                feats[name] = np.nan
            else:
                feats[name] = float(fn(t))
        except Exception:
            feats[name] = np.nan
    return feats


def add_dense_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = [extract_dense_row(x) for x in df["text"].tolist()]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


dense_cols = [
    "char_len",
    "word_len_ws",
    "tiktoken_len",
    "digit_ratio",
    "upper_ratio",
    "punct_ratio",
    "space_ratio",
    "mean_word_len",
    "char_entropy",
    "ttr",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "smog_index",
    "coleman_liau_index",
]

df_feat = add_dense_features(df_all)
print("Dense shape (без legacy HC):", df_feat.shape)''',
        "dense",
    )
)

cells.append(
    cell_md(
        r"""## Наследие v1: 68 hand-crafted стилометрических признаков (HC)

Соответствует группам из `notebooks/02_features/03_feature_engineering.ipynb` (**68** признаков: лексика, POS-доли, функциональные слова, пунктуация, MTLD / Yule / hapax / TTR, леммы через WordNet). Реализация на **NLTK** (без spaCy). Колонки с префиксом `hc_` добавляются к таблице и пишутся в общий `core_dense_features.parquet`.

Извлечение по всем строкам может занять несколько минут (`joblib` параллельно)."""
    )
)

cells.append(
    cell_code(
        r'''import sys

sys.path.insert(0, str(BASE))

import nltk
from joblib import Parallel, delayed

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
try:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
except Exception:
    nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from src.core_legacy_hc_features import LEGACY_HC_NAMES, add_hc_prefix, extract_legacy_hc_row

legacy_hc_cols = [add_hc_prefix(n) for n in LEGACY_HC_NAMES]
assert len(legacy_hc_cols) == 68

texts = df_feat["text"].astype(str).tolist()

def _hc_row(t: str) -> dict:
    r = extract_legacy_hc_row(t)
    return {add_hc_prefix(k): float(v) for k, v in r.items()}

hc_part = Parallel(n_jobs=-1, verbose=5)(delayed(_hc_row)(t) for t in texts)
hc_df = pd.DataFrame(hc_part).reindex(columns=legacy_hc_cols, fill_value=0.0)

df_feat = pd.concat([df_feat.reset_index(drop=True), hc_df], axis=1)

feat_path = FEATURE_DIR / "core_dense_features.parquet"
df_feat.to_parquet(feat_path, index=False)
print("Wrote", feat_path, "shape", df_feat.shape, "hc columns:", len(legacy_hc_cols))''',
        "legacy-hc",
    )
)

cells.append(cell_md("## TF-IDF (fit только на train)"))

cells.append(
    cell_code(
        r'''from sklearn.feature_extraction.text import TfidfVectorizer

train_mask = df_feat["_load_split"] == "train"
text_train = df_feat.loc[train_mask, "text"].astype(str)

vec_word = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
)
vec_char = TfidfVectorizer(
    analyzer="char_wb",
    max_features=4000,
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
)

Xw_train = vec_word.fit_transform(text_train)
Xc_train = vec_char.fit_transform(text_train)

def tfidf_for_subset(sub_df: pd.DataFrame):
    """Пустой подфрейм: после drop_duplicates по core_row_id у test_full часто 0 строк."""
    if len(sub_df) == 0:
        return (
            sparse.csr_matrix((0, Xw_train.shape[1])),
            sparse.csr_matrix((0, Xc_train.shape[1])),
        )
    t = sub_df["text"].astype(str)
    return vec_word.transform(t), vec_char.transform(t)

X_word_parts = {}
X_char_parts = {}
for name in paths:
    m = df_feat["_load_split"] == name
    sub = df_feat.loc[m]
    w, c = tfidf_for_subset(sub)
    if len(sub) == 0:
        print(f"TF-IDF: {name} — 0 строк (часто test_full дублирует test_non_claude/test_claude_only)")
    X_word_parts[name] = w
    X_char_parts[name] = c

sparse.save_npz(FEATURE_DIR / "core_tfidf_word_train.npz", Xw_train)
sparse.save_npz(FEATURE_DIR / "core_tfidf_char_train.npz", Xc_train)
with (FEATURE_DIR / "core_tfidf_word_vectorizer.pkl").open("wb") as f:
    pickle.dump(vec_word, f)
with (FEATURE_DIR / "core_tfidf_char_vectorizer.pkl").open("wb") as f:
    pickle.dump(vec_char, f)

print("word tfidf train", Xw_train.shape, "char", Xc_train.shape)''',
        "tfidf",
    )
)

cells.append(
    cell_md(
        r"""## LM-based scoring: perplexity / mean NLL (отдельная секция)

**Требуется:** `uv sync --extra lm_scoring`.

**Важно:** scoring-модель (например DistilGPT-2) **не** совпадает с генераторами Core; метрика отражает «насколько текст предсказуем для этого LM», а не ground-truth вероятность генерации.

Кэш: `data/interim/features/core_lm_distilgpt2_scores.parquet` — при повторном запуске подгружается, если существует и число строк совпадает."""
    )
)

cells.append(
    cell_code(
        r'''LM_CACHE = FEATURE_DIR / "core_lm_distilgpt2_scores.parquet"
LM_MODEL_NAME = "distilgpt2"
MAX_LEN = 512
BATCH_SIZE = 8

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _LM_OK = True
except ImportError:
    _LM_OK = False
    print("Skip LM: install with uv sync --extra lm_scoring")


def per_sample_mean_nll(logits, labels, attention_mask):
    """Mean CE over non-padding positions (shifted for causal LM)."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_labels = shift_labels.reshape(-1)
    flat_mask = shift_mask.reshape(-1).float()
    loss = loss_fct(flat_logits, flat_labels).reshape(shift_labels.shape)
    loss = loss * shift_mask.float()
    ntok = shift_mask.sum(dim=1).clamp(min=1)
    return (loss.sum(dim=1) / ntok).detach().cpu().numpy()


def compute_lm_scores(df: pd.DataFrame, model, tokenizer, device) -> np.ndarray:
    texts = df["text"].astype(str).tolist()
    out = np.full(len(texts), np.nan, dtype=np.float64)
    model.eval()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        mnl = per_sample_mean_nll(logits, enc["input_ids"], enc["attention_mask"])
        out[i : i + len(batch)] = mnl
    return out


if _LM_OK:
    run_lm = True
    if LM_CACHE.is_file():
        cached = pd.read_parquet(LM_CACHE)
        if {"core_row_id", "lm_mean_nll"}.issubset(cached.columns):
            sub = cached[["core_row_id", "lm_mean_nll"]].drop_duplicates("core_row_id")
            m = df_feat[["core_row_id"]].merge(sub, on="core_row_id", how="left")
            if m["lm_mean_nll"].notna().all() and len(m) == len(df_feat):
                df_feat["lm_mean_nll"] = m["lm_mean_nll"].values
                df_feat["lm_perplexity"] = np.exp(df_feat["lm_mean_nll"])
                print("Loaded LM from cache", LM_CACHE)
                run_lm = False
    if run_lm:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(LM_MODEL_NAME)
        mod = AutoModelForCausalLM.from_pretrained(LM_MODEL_NAME).to(device)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        scores = compute_lm_scores(df_feat, mod, tok, device)
        df_feat["lm_mean_nll"] = scores
        df_feat["lm_perplexity"] = np.exp(scores)
        df_feat[["core_row_id", "lm_mean_nll", "lm_perplexity"]].to_parquet(LM_CACHE, index=False)
        print("Wrote", LM_CACHE)
else:
    df_feat["lm_mean_nll"] = np.nan
    df_feat["lm_perplexity"] = np.nan''',
        "lm",
    )
)

cells.append(
    cell_md(
        """## Глубокий EDA (диагностика)

Не финальный baseline: описательная статистика, тесты различий (с учётом множественных сравнений), PCA, MI, логистическая регрессия train→val и SHAP на подвыборке.

**Срезы отчётности:** `core_eval_slice` ∈ {`val`, `test_seen` ≈ non-Claude test, `test_claude_holdout`}; `test_full` — только при явной пометке."""
    )
)

cells.append(
    cell_code(
        r'''from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

train_mask = df_feat["_load_split"] == "train"
val_mask = df_feat["_load_split"] == "val"
report_slices = {"val", "test_seen", "test_claude_holdout"}
df_rep = df_feat[df_feat["core_eval_slice"].isin(report_slices)].copy()

lm_ok = "lm_mean_nll" in df_rep.columns and df_rep["lm_mean_nll"].notna().any()
legacy_hc_cols = sorted([c for c in df_feat.columns if str(c).startswith("hc_")])
num_for_model = list(dense_cols) + legacy_hc_cols + (["lm_mean_nll"] if lm_ok else [])
print("Features for model block: dense", len(dense_cols), "legacy_hc", len(legacy_hc_cols), "lm", int(lm_ok))

# Boxplot: human vs LLM по отчётным срезам (агрегат)
key_feat = "char_len"
fig, ax = plt.subplots(figsize=(8, 4))
sub = df_rep[[key_feat, "label", "core_eval_slice"]].dropna()
sns.boxplot(data=sub, x="core_eval_slice", y=key_feat, hue="label", ax=ax)
ax.set_title(f"{key_feat} по core_eval_slice (label 0=human, 1=LLM)")
fig.tight_layout()
fig.savefig(FIG_DIR / "box_char_len_by_eval_slice.png", dpi=150)
plt.close(fig)
print("Wrote", FIG_DIR / "box_char_len_by_eval_slice.png")

# По channel / scenario_family на train
for col in ["channel", "scenario_family"]:
    g = df_feat.loc[train_mask].groupby([col, "label"])[key_feat].median().unstack()
    g.to_csv(TABLE_DIR / f"median_{key_feat}_by_{col}_train.csv")
print("Wrote group medians", TABLE_DIR)''',
        "eda-desc",
    )
)

cells.append(
    cell_md(
        """### Mann–Whitney U, эффекты (Cohen's d), множественные сравнения

На **train** сравниваем распределения признаков для `label=0` vs `label=1`. p-values корректируем по **Bonferroni** (`α/m`). Для рангового теста Cohen's d — ориентир по величине эффекта (осторожно при сильных выбросах)."""
    )
)

cells.append(
    cell_code(
        r'''def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1))
    if sp == 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / sp)


rows = []
m = train_mask
for feat in num_for_model:
    x0 = df_feat.loc[m & (df_feat["label"] == 0), feat].astype(float).values
    x1 = df_feat.loc[m & (df_feat["label"] == 1), feat].astype(float).values
    x0, x1 = x0[np.isfinite(x0)], x1[np.isfinite(x1)]
    if len(x0) < 20 or len(x1) < 20:
        continue
    stat, p = mannwhitneyu(x0, x1, alternative="two-sided")
    rows.append(
        {
            "feature": feat,
            "p_mannwhitney": p,
            "cohen_d_human_minus_llm": cohen_d(x0, x1),
            "n0": len(x0),
            "n1": len(x1),
        }
    )

tab = pd.DataFrame(rows).sort_values("p_mannwhitney")
mtests = len(tab)
tab["p_bonferroni"] = np.minimum(tab["p_mannwhitney"] * mtests, 1.0)
tab.to_csv(TABLE_DIR / "mannwhitney_train_human_vs_llm.csv", index=False)
print("Tests:", mtests, "→ table", TABLE_DIR / "mannwhitney_train_human_vs_llm.csv")
tab.head(12)''',
        "eda-mw",
    )
)

cells.append(
    cell_md(
        """### Legacy HC (v1-style): распределения и связь с меткой (train)

Сводные mean/std по `label`, **point-biserial correlation** с бинарной меткой, KDE для топ-8 HC по |r|, heatmap Pearson среди **топ-24** HC по |point-biserial|."""
    )
)

cells.append(
    cell_code(
        r'''from scipy.stats import pointbiserialr

legacy_hc_cols = sorted([c for c in df_feat.columns if str(c).startswith("hc_")])
train_df = df_feat.loc[train_mask]
y_bin = train_df["label"].astype(int).values

summ = train_df.groupby("label")[legacy_hc_cols].agg(["mean", "std"])
summ.to_csv(TABLE_DIR / "hc_summary_mean_std_by_label_train.csv")
print("Wrote", TABLE_DIR / "hc_summary_mean_std_by_label_train.csv")

pbs = []
for c in legacy_hc_cols:
    x = train_df[c].astype(float).values
    m = np.isfinite(x)
    if m.sum() < 30:
        continue
    r, p = pointbiserialr(y_bin[m], x[m])
    if np.isfinite(r):
        pbs.append({"feature": c, "r_pointbiserial": float(r), "p": float(p)})

pb_df = pd.DataFrame(pbs)
pb_df["abs_r"] = pb_df["r_pointbiserial"].abs()
pb_df = pb_df.sort_values("abs_r", ascending=False)
pb_df.drop(columns=["abs_r"]).to_csv(TABLE_DIR / "hc_pointbiserial_train_label.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 7))
top20 = pb_df.head(20)
ax.barh(top20["feature"], top20["r_pointbiserial"], color="steelblue")
ax.axvline(0, color="k", lw=0.8)
ax.set_title("Top-20 legacy HC by |point-biserial| with label (train)")
fig.tight_layout()
fig.savefig(FIG_DIR / "hc_pointbiserial_top20_train.png", dpi=150)
plt.close(fig)

top8 = pb_df.head(8)["feature"].tolist()
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
axes = axes.ravel()
for ax, cname in zip(axes, top8):
    for lab in (0, 1):
        sub = train_df.loc[train_df["label"] == lab, cname].astype(float).dropna()
        if len(sub) > 5:
            try:
                sns.kdeplot(sub, ax=ax, label=f"label={lab}", warn_singular=False)
            except TypeError:
                sns.kdeplot(sub, ax=ax, label=f"label={lab}")
    ax.set_title(cname.replace("hc_", ""), fontsize=8)
    ax.legend(fontsize=7)
fig.suptitle("KDE: top-8 HC by |point-biserial| (train)")
fig.tight_layout()
fig.savefig(FIG_DIR / "hc_kde_top8_by_label_train.png", dpi=150)
plt.close(fig)

top24 = pb_df.head(24)["feature"].tolist()
C = train_df[top24].astype(float).corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(C, cmap="vlag", center=0, ax=ax, xticklabels=False, yticklabels=False)
ax.set_title("Pearson among top-24 HC by |point-biserial| (train)")
fig.tight_layout()
fig.savefig(FIG_DIR / "hc_correlation_top24_train.png", dpi=150)
plt.close(fig)
print("HC distribution figures →", FIG_DIR)''',
        "eda-hc-dist",
    )
)

cells.append(
    cell_md(
        """### Корреляции dense-признаков и PCA

Корреляции (Pearson) на **train** после median-impute. PCA на масштабированных признаках; точки окрашены по `label` и отдельно по `scenario_family` (train)."""
    )
)

cells.append(
    cell_code(
        r'''from scipy.sparse import csr_matrix, hstack

imp = SimpleImputer(strategy="median")
X_tr = imp.fit_transform(df_feat.loc[train_mask, dense_cols].astype(float))
X_tr_df = pd.DataFrame(X_tr, columns=dense_cols, index=df_feat.index[train_mask])
corr = X_tr_df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap="vlag", center=0, ax=ax)
ax.set_title("Pearson correlation (train, dense, imputed)")
fig.tight_layout()
fig.savefig(FIG_DIR / "dense_correlation_heatmap_train.png", dpi=150)
plt.close(fig)

sc = StandardScaler()
Z = sc.fit_transform(imp.transform(df_feat.loc[train_mask, dense_cols].astype(float)))
pca = PCA(n_components=2, random_state=SEED)
P = pca.fit_transform(Z)
var = pca.explained_variance_ratio_
pd.DataFrame({"PC": ["PC1", "PC2"], "explained_variance_ratio": var}).to_csv(
    TABLE_DIR / "pca_explained_variance_train.csv", index=False
)

idx_tr = df_feat.index[train_mask]
plot_df = pd.DataFrame({"PC1": P[:, 0], "PC2": P[:, 1], "label": df_feat.loc[idx_tr, "label"].values})
fig, ax = plt.subplots(figsize=(6, 5))
sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="label", alpha=0.35, ax=ax)
ax.set_title(f"PCA train (dense) — var={var[0]:.2f},{var[1]:.2f}")
fig.tight_layout()
fig.savefig(FIG_DIR / "pca_train_by_label.png", dpi=150)
plt.close(fig)

plot_df2 = pd.DataFrame(
    {
        "PC1": P[:, 0],
        "PC2": P[:, 1],
        "scenario_family": df_feat.loc[idx_tr, "scenario_family"].astype(str).values,
    }
)
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=plot_df2, x="PC1", y="PC2", hue="scenario_family", alpha=0.35, ax=ax, legend=False)
ax.set_title("PCA train colored by scenario_family")
fig.tight_layout()
fig.savefig(FIG_DIR / "pca_train_by_scenario.png", dpi=150)
plt.close(fig)
print("PCA tables/figures done")''',
        "eda-pca",
    )
)

cells.append(
    cell_md(
        """### Mutual information с `label` (train)

Нелинейная информативность (дискретизация внутри `mutual_info_classif`)."""
    )
)

cells.append(
    cell_code(
        r'''X_mi = imp.transform(df_feat.loc[train_mask, dense_cols].astype(float))
y_mi = df_feat.loc[train_mask, "label"].astype(int).values
mi = mutual_info_classif(X_mi, y_mi, random_state=SEED, discrete_features=False)
mi_tab = pd.DataFrame({"feature": dense_cols, "mi_label": mi}).sort_values("mi_label", ascending=False)
mi_tab.to_csv(TABLE_DIR / "mutual_info_train_label.csv", index=False)
mi_tab.head(15)''',
        "eda-mi",
    )
)

cells.append(
    cell_md(
        """### Диагностическая LogisticRegression (dense + TF-IDF, train → val)

`TfidfVectorizer` уже обучен **только на train** — утечки нет. Модель — sanity-check (сильная размерность, без тюнинга); метрики на **val** не позиционируются как итоговый результат."""
    )
)

cells.append(
    cell_code(
        r'''Xw_tr = X_word_parts["train"]
Xc_tr = X_char_parts["train"]
imp_lr = SimpleImputer(strategy="median")
sc_lr = StandardScaler()
Xd_tr = sc_lr.fit_transform(imp_lr.fit_transform(df_feat.loc[train_mask, num_for_model].astype(float)))

X_train_lr = hstack([csr_matrix(Xd_tr), Xw_tr, Xc_tr], format="csr")
y_tr = df_feat.loc[train_mask, "label"].astype(int).values

Xd_va = sc_lr.transform(imp_lr.transform(df_feat.loc[val_mask, num_for_model].astype(float)))
Xw_va = X_word_parts["val"]
Xc_va = X_char_parts["val"]
X_val_lr = hstack([csr_matrix(Xd_va), Xw_va, Xc_va], format="csr")
y_va = df_feat.loc[val_mask, "label"].astype(int).values

clf = LogisticRegression(
    max_iter=200,
    class_weight="balanced",
    solver="saga",
    n_jobs=-1,
    random_state=SEED,
)
clf.fit(X_train_lr, y_tr)
p_va = clf.predict_proba(X_val_lr)[:, 1]
auc = roc_auc_score(y_va, p_va)
ap = average_precision_score(y_va, p_va)
pd.DataFrame([{"split": "val", "roc_auc": auc, "avg_precision": ap}]).to_csv(
    TABLE_DIR / "diagnostic_logreg_val_metrics.csv", index=False
)
print("Diagnostic LogReg val ROC-AUC", auc, "AP", ap)''',
        "eda-lr",
    )
)

cells.append(
    cell_md(
        """### SHAP (LinearExplainer) на dense+LM (подвыборка train)

Полная разреженная матрица TF-IDF+LR слишком велика для стабильного `LinearExplainer` здесь; отдельная **логистическая модель только на dense** для интерпретации вкладов тех же инженерных признаков."""
    )
)

cells.append(
    cell_code(
        r'''import shap

# Отдельная линейная модель только на dense+LM: LinearExplainer на полном TF-IDF слишком тяжёл
clf_dense = LogisticRegression(
    max_iter=200,
    class_weight="balanced",
    solver="lbfgs",
    random_state=SEED,
)
clf_dense.fit(Xd_tr, y_tr)
n_shap = min(2000, Xd_tr.shape[0])
X_s = Xd_tr[:n_shap]

explainer = shap.LinearExplainer(clf_dense, shap.maskers.Independent(X_s))
sv = explainer.shap_values(X_s)
feat_names = num_for_model
imp_mean = np.abs(sv).mean(axis=0)
top_k = min(25, len(feat_names))
rank = np.argsort(-imp_mean)[:top_k]
shap_summary = pd.DataFrame(
    {"feature": [feat_names[i] for i in rank], "mean_abs_shap": imp_mean[rank]}
)
shap_summary.to_csv(TABLE_DIR / "shap_linear_dense_top_features.csv", index=False)
print(shap_summary)
print("Wrote", TABLE_DIR / "shap_linear_dense_top_features.csv")''',
        "eda-shap",
    )
)

cells.append(
    cell_md(
        """### Стабильность по генератору (только `label=1`)

Сравнение ключевых признаков: **test_seen** vs **test_claude_holdout** (и при наличии колонки `generator_lane`). Это диагностика сдвига распределений, не метрика классификатора."""
    )
)

cells.append(
    cell_code(
        r'''key_numeric = [c for c in ["char_len", "tiktoken_len", "ttr", "char_entropy", "lm_mean_nll"] if c in df_feat.columns]
stab_rows = []
for sl_a, sl_b in [("test_seen", "test_claude_holdout")]:
    m1 = (df_feat["core_eval_slice"] == sl_a) & (df_feat["label"] == 1)
    m2 = (df_feat["core_eval_slice"] == sl_b) & (df_feat["label"] == 1)
    for feat in key_numeric:
        a = df_feat.loc[m1, feat].astype(float).dropna().values
        b = df_feat.loc[m2, feat].astype(float).dropna().values
        if len(a) < 10 or len(b) < 10:
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        stab_rows.append(
            {
                "slice_a": sl_a,
                "slice_b": sl_b,
                "feature": feat,
                "median_a": float(np.median(a)),
                "median_b": float(np.median(b)),
                "p_mannwhitney": float(p),
                "n_a": len(a),
                "n_b": len(b),
            }
        )

if "generator_lane" in df_feat.columns:
    for lane in df_feat["generator_lane"].dropna().unique()[:12]:
        mL = (df_feat["label"] == 1) & (df_feat["generator_lane"].astype(str) == str(lane))
        if mL.sum() < 30:
            continue
        for feat in key_numeric:
            v = df_feat.loc[mL, feat].astype(float).dropna()
            stab_rows.append(
                {
                    "slice_a": "generator_lane",
                    "slice_b": str(lane),
                    "feature": feat,
                    "median_a": float(v.median()),
                    "median_b": np.nan,
                    "p_mannwhitney": np.nan,
                    "n_a": int(len(v)),
                    "n_b": np.nan,
                }
            )

stab_df = pd.DataFrame(stab_rows)
stab_df.to_csv(TABLE_DIR / "stability_llm_slices_and_lanes.csv", index=False)
print("Wrote", TABLE_DIR / "stability_llm_slices_and_lanes.csv")
stab_df.head(10)''',
        "eda-stab",
    )
)

cells.append(
    cell_md(
        r"""## Опционально: `test_full` (агрегат)

Сводная фигура только для обзора; **не** подменяет отчётные срезы `val` / `test_seen` / `test_claude_holdout`."""
    )
)

cells.append(
    cell_code(
        r'''tf = df_feat[df_feat["_load_split"] == "test_full"]
if len(tf) > 0:
    fig, ax = plt.subplots(figsize=(5, 4))
    vc = tf["label"].value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values)
    ax.set_title("test_full: label counts (aggregated only)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "test_full_label_counts.png", dpi=150)
    plt.close(fig)
    print("Wrote", FIG_DIR / "test_full_label_counts.png")
else:
    print("No test_full rows")''',
        "eda-tfull",
    )
)

cells.append(
    cell_md(
        """## Валидация и ограничения

**Проверки:** целостность строк, согласованность размерностей TF-IDF и матриц LR.

**Здесь намеренно не делали:** финальный тюнинг XGBoost/бустинга, калибровку прод-модели, отчётные test-метрики по протоколу Core (только диагностический val для LogReg). См. `v2/docs/next_tasks.md`."""
    )
)

cells.append(
    cell_code(
        r'''assert df_feat["label"].notna().all()
assert df_feat["core_row_id"].nunique() == len(df_feat)
assert X_word_parts["train"].shape[0] == train_mask.sum()
assert X_word_parts["val"].shape[0] == val_mask.sum()
assert X_train_lr.shape[1] == X_val_lr.shape[1]
print("Checks OK")''',
        "validate",
    )
)

if __name__ == "__main__":
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Wrote notebook", OUT)
