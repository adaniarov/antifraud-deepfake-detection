"""Generate 01_core_classical_baselines.ipynb — run: cd v2 && uv run python notebooks/05_classical_ml/_gen_baseline_nb.py"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "01_core_classical_baselines.ipynb"


def cell_md(text: str) -> dict:
    lines = text.strip().split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else ["\n"])
    return {"cell_type": "markdown", "id": f"md-{abs(hash(text)) % 10**8}", "metadata": {}, "source": src}


def cell_code(text: str, cid: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")],
    }


cells: list[dict] = []

cells.append(
    cell_md(
        r"""# Core v2 — классические baseline (train → отчётные срезы)

**Задача:** обучить воспроизводимые baseline-классификаторы human vs LLM на замороженном Core и оценить по протоколу v2.

**Метрики (раздельно):** **val** (`core_val.jsonl`), **test_seen** (`core_test_non_claude.jsonl`), **test_claude_holdout** (`core_test_claude_only.jsonl`). `test_full` — только дополнительно.

**Признаки:** dense + **68 legacy HC** (`hc_*`, как в v1 `03_feature_engineering`, через NLTK) в `core_dense_features.parquet`, опционально LM NLL; TF-IDF word/char — pickle из `04_features` (fit только на train).

**Фиксация:** `random_state=42`, `class_weight='balanced'` где применимо."""
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
from scipy.sparse import csr_matrix, hstack

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["figure.dpi"] = 120

SEED = 42
np.random.seed(SEED)


def resolve_v2_base() -> Path:
    cur = Path.cwd().resolve()
    for p in [cur, *cur.parents]:
        if (p / "v2" / "pyproject.toml").is_file():
            return p / "v2"
        if p.name == "v2" and (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError("Нужен v2/pyproject.toml (запуск из репозитория).")


BASE = resolve_v2_base()
FEATURE_DIR = BASE / "data" / "interim" / "features"
FIG_DIR = BASE / "outputs" / "figures" / "classical_ml"
TABLE_DIR = BASE / "outputs" / "tables" / "classical_ml"
for d in (FIG_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

print("BASE:", BASE)''',
        "setup",
    )
)

cells.append(
    cell_code(
        r'''DENSE_NUMERIC = [
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

df = pd.read_parquet(FEATURE_DIR / "core_dense_features.parquet")
lm_path = FEATURE_DIR / "core_lm_distilgpt2_scores.parquet"
if lm_path.is_file():
    lm = pd.read_parquet(lm_path)
    if "core_row_id" in lm.columns and "lm_mean_nll" in lm.columns:
        df = df.drop(columns=[c for c in ("lm_mean_nll", "lm_perplexity") if c in df.columns], errors="ignore")
        df = df.merge(lm[["core_row_id", "lm_mean_nll"]], on="core_row_id", how="left")
        print("Merged LM scores from", lm_path)
else:
    print("No LM cache — run 04_features with lm_scoring optional extra if needed")

USE_LM = "lm_mean_nll" in df.columns and df["lm_mean_nll"].notna().any()
LEGACY_HC_COLS = sorted([c for c in df.columns if str(c).startswith("hc_")])
num_cols = list(DENSE_NUMERIC) + LEGACY_HC_COLS + (["lm_mean_nll"] if USE_LM else [])
print("num_cols: dense", len(DENSE_NUMERIC), "hc", len(LEGACY_HC_COLS), "lm", int(USE_LM), "total", len(num_cols))

train_m = df["_load_split"] == "train"
val_m = df["_load_split"] == "val"
ts_m = df["_load_split"] == "test_non_claude"
tcl_m = df["_load_split"] == "test_claude_only"

split_masks = {
    "train": train_m,
    "val": val_m,
    "test_seen": ts_m,
    "test_claude_holdout": tcl_m,
}

for k, m in split_masks.items():
    print(k, m.sum(), "label balance", df.loc[m, "label"].value_counts().to_dict())

with (FEATURE_DIR / "core_tfidf_word_vectorizer.pkl").open("rb") as f:
    vec_w = pickle.load(f)
with (FEATURE_DIR / "core_tfidf_char_vectorizer.pkl").open("rb") as f:
    vec_c = pickle.load(f)

def tf_stack(sub: pd.DataFrame):
    t = sub["text"].astype(str)
    return vec_w.transform(t), vec_c.transform(t)

Xw = {}
Xc = {}
a0, b0 = tf_stack(df.loc[train_m])
nw, nc = a0.shape[1], b0.shape[1]
for name, m in split_masks.items():
    sub = df.loc[m]
    if len(sub) == 0:
        Xw[name] = sparse.csr_matrix((0, nw))
        Xc[name] = sparse.csr_matrix((0, nc))
    else:
        a, b = tf_stack(sub)
        Xw[name] = a
        Xc[name] = b

print("TF-IDF word dim", nw, "char", nc)''',
        "load",
    )
)

cells.append(
    cell_code(
        r'''from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.preprocessing import StandardScaler

imp = SimpleImputer(strategy="median")
sc = StandardScaler()
imp.fit(df.loc[train_m, num_cols].astype(float))
sc.fit(imp.transform(df.loc[train_m, num_cols].astype(float)))

Xd = {}
for name, m in split_masks.items():
    Xd[name] = sc.transform(imp.transform(df.loc[m, num_cols].astype(float)))

y = {k: df.loc[m, "label"].astype(int).values for k, m in split_masks.items()}

X_full = {
    k: hstack([csr_matrix(Xd[k]), Xw[k], Xc[k]], format="csr")
    for k in split_masks
}

print("X_full train", X_full["train"].shape)''',
        "features",
    )
)

cells.append(
    cell_code(
        r'''from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def eval_split(clf, X, y_true, need_proba=True):
    if need_proba and hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)[:, 1]
    else:
        p = clf.decision_function(X)
        p = (p - p.min()) / (p.max() - p.min() + 1e-12)
    pred = (p >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": float("nan"),
        "avg_precision": float("nan"),
        "f1": float(f1_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "scores": p,
        "pred": pred,
    }
    if np.unique(y_true).size >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, p))
        out["avg_precision"] = float(average_precision_score(y_true, p))
    else:
        # В Core v2 в test_claude_only сейчас только label=1 (LLM) — нет пары human для ROC/AP
        out["llm_predicted_rate"] = float(np.mean(pred == 1))
    return out


def plot_curves(model_slug, clf, X_dict, y_dict, splits_plot):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, spl in zip(axes, splits_plot):
        yt = y_dict[spl]
        r = eval_split(clf, X_dict[spl], yt)
        if np.unique(yt).size >= 2:
            RocCurveDisplay.from_predictions(yt, r["scores"], ax=ax, name=spl)
        else:
            ax.text(0.5, 0.5, "ROC N/A\n(один класс в y)", ha="center", va="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.set_title(f"ROC — {spl}")
    fig.suptitle(f"{model_slug} — ROC")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{model_slug}_roc.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, spl in zip(axes, splits_plot):
        yt = y_dict[spl]
        r = eval_split(clf, X_dict[spl], yt)
        if np.unique(yt).size >= 2:
            PrecisionRecallDisplay.from_predictions(yt, r["scores"], ax=ax, name=spl)
        else:
            ax.text(0.5, 0.5, "PR N/A\n(один класс в y)", ha="center", va="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.set_title(f"PR — {spl}")
    fig.suptitle(f"{model_slug} — PR")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{model_slug}_pr.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, spl in zip(axes, splits_plot):
        r = eval_split(clf, X_dict[spl], y_dict[spl])
        cm = confusion_matrix(y_dict[spl], r["pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(spl)
    fig.suptitle(f"{model_slug} — confusion")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{model_slug}_confusion.png", dpi=150)
    plt.close(fig)


EVAL_SPLITS = ["val", "test_seen", "test_claude_holdout"]
rows = []''',
        "helpers",
    )
)

cells.append(
    cell_code(
        r'''# 1) LogisticRegression: dense + TF-IDF (главный sparse baseline)
lr_full = LogisticRegression(
    max_iter=300,
    class_weight="balanced",
    solver="saga",
    n_jobs=-1,
    random_state=SEED,
)
lr_full.fit(X_full["train"], y["train"])

for spl in EVAL_SPLITS:
    r = eval_split(lr_full, X_full[spl], y[spl])
    row = {k: r[k] for k in r if k not in ("scores", "pred")}
    rows.append({"model": "lr_dense_tfidf", "split": spl, **row})

plot_curves("lr_dense_tfidf", lr_full, X_full, y, EVAL_SPLITS)
print("lr_dense_tfidf done")''',
        "m1",
    )
)

cells.append(
    cell_code(
        r'''# 2) LogisticRegression: только dense (+LM)
lr_d = LogisticRegression(
    max_iter=200,
    class_weight="balanced",
    solver="lbfgs",
    random_state=SEED,
)
lr_d.fit(Xd["train"], y["train"])

for spl in EVAL_SPLITS:
    r = eval_split(lr_d, Xd[spl], y[spl])
    rows.append({"model": "lr_dense_only", "split": spl, **{k: r[k] for k in r if k not in ("scores", "pred")}})

plot_curves("lr_dense_only", lr_d, Xd, y, EVAL_SPLITS)
print("lr_dense_only done")''',
        "m2",
    )
)

cells.append(
    cell_code(
        r'''# 3) RandomForest на dense (+LM)
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=24,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(Xd["train"], y["train"])

for spl in EVAL_SPLITS:
    r = eval_split(rf, Xd[spl], y[spl])
    rows.append({"model": "rf_dense", "split": spl, **{k: r[k] for k in r if k not in ("scores", "pred")}})

plot_curves("rf_dense", rf, Xd, y, EVAL_SPLITS)
print("rf_dense done")''',
        "m3",
    )
)

cells.append(
    cell_code(
        r'''# 4) XGBoost на dense (если доступен)
xgb_clf = None
if HAS_XGB:
    xgb_clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        eval_metric="logloss",
    )
    xgb_clf.fit(Xd["train"], y["train"], verbose=False)
    for spl in EVAL_SPLITS:
        r = eval_split(xgb_clf, Xd[spl], y[spl])
        rows.append({"model": "xgb_dense", "split": spl, **{k: r[k] for k in r if k not in ("scores", "pred")}})
    plot_curves("xgb_dense", xgb_clf, Xd, y, EVAL_SPLITS)
    print("xgb_dense done")
else:
    print("xgboost not available — skip")''',
        "m4",
    )
)

cells.append(
    cell_code(
        r'''# Сводная таблица + test_full (агрегат): объединение test_seen ∪ test_claude_holdout по строкам df
m_tf = df["_load_split"].isin(["test_non_claude", "test_claude_only"])
if m_tf.sum() > 0:
    imp_tf = SimpleImputer(strategy="median")
    sc_tf = StandardScaler()
    imp_tf.fit(df.loc[train_m, num_cols].astype(float))
    sc_tf.fit(imp_tf.transform(df.loc[train_m, num_cols].astype(float)))
    Xd_tf = sc_tf.transform(imp_tf.transform(df.loc[m_tf, num_cols].astype(float)))
    tw, tc = tf_stack(df.loc[m_tf])
    X_tf = hstack([csr_matrix(Xd_tf), tw, tc], format="csr")
    y_tf = df.loc[m_tf, "label"].astype(int).values
    for model_name, clf in [
        ("lr_dense_tfidf", lr_full),
        ("lr_dense_only", lr_d),
        ("rf_dense", rf),
    ] + ([("xgb_dense", xgb_clf)] if HAS_XGB and xgb_clf is not None else []):
        if model_name == "lr_dense_tfidf":
            r = eval_split(clf, X_tf, y_tf)
        else:
            r = eval_split(clf, Xd_tf, y_tf)
        rows.append({"model": model_name, "split": "test_full_aggregate", **{k: r[k] for k in r if k not in ("scores", "pred")}})

metrics_df = pd.DataFrame(rows)
metrics_path = TABLE_DIR / "baseline_metrics_by_split.csv"
metrics_df.to_csv(metrics_path, index=False)
print(metrics_df.to_string(index=False))
print("Wrote", metrics_path)''',
        "aggregate",
    )
)

cells.append(
    cell_md(
        """## Диагностика по `scenario_family` (test_seen)

Ориентировочные метрики **только** для среза `test_seen` — показывают неоднородность сценариев."""
    )
)

cells.append(
    cell_code(
        r'''rows_sf = []
sub = df.loc[ts_m].copy()
if len(sub) > 0:
    imp2 = SimpleImputer(strategy="median")
    sc2 = StandardScaler()
    imp2.fit(df.loc[train_m, num_cols].astype(float))
    sc2.fit(imp2.transform(df.loc[train_m, num_cols].astype(float)))
    for fam in sorted(sub["scenario_family"].astype(str).unique()):
        mloc = sub["scenario_family"].astype(str) == fam
        if mloc.sum() < 30:
            continue
        idx = sub.index[mloc]
        Xloc = hstack(
            [
                csr_matrix(
                    sc2.transform(imp2.transform(df.loc[idx, num_cols].astype(float)))
                ),
                vec_w.transform(df.loc[idx, "text"].astype(str)),
                vec_c.transform(df.loc[idx, "text"].astype(str)),
            ],
            format="csr",
        )
        ylab = df.loc[idx, "label"].astype(int).values
        if np.unique(ylab).size < 2:
            continue
        r = eval_split(lr_full, Xloc, ylab)
        rows_sf.append({"scenario_family": fam, "n": int(mloc.sum()), **{k: r[k] for k in r if k not in ("scores", "pred")}})
    if rows_sf:
        sf_df = pd.DataFrame(rows_sf).sort_values("roc_auc")
        sf_path = TABLE_DIR / "baseline_by_scenario_family_test_seen.csv"
        sf_df.to_csv(sf_path, index=False)
        print(sf_df.to_string(index=False))
        print("Wrote", sf_path)
    else:
        print("No per-family rows with both classes")
else:
    print("No test_seen rows")''',
        "by_family",
    )
)

cells.append(
    cell_md(
        """## Где лежат артефакты

- Таблицы: `v2/outputs/tables/classical_ml/`
- Фигуры: `v2/outputs/figures/classical_ml/` (`*_roc.png`, `*_pr.png`, `*_confusion.png`)"""
    )
)

if __name__ == "__main__":
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Wrote", OUT)
