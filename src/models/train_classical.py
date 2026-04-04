"""
train_classical.py — Stage 7: Classical ML baseline.

VERSION: no-perplexity
  Perplexity / burstiness features (ppl_*, burstiness) are excluded
  because GPT-2 inference has not been run yet.
  Re-run with --include-perplexity once extract.py has been run in full.

Feature sets used:
  A. Hand-crafted stylometric + lexical diversity  (from train_features.parquet)
     EXCLUDING: ppl_mean, ppl_sent_*, burstiness
  B. TF-IDF word n-grams (1,2)       (from tfidf_word/*.npz)
  C. TF-IDF char n-grams (2,4)       (from tfidf_char/*.npz)
  D. Combined: A + B (sparse concat)
  E. Combined: A + C (sparse concat)
  F. Combined: A + B + C

Models evaluated on each feature set:
  1. Logistic Regression
  2. Random Forest
  3. XGBoost

Evaluation:
  - Primary metrics: Accuracy, Precision, Recall, F1, ROC-AUC
  - Reported on val set during training
  - Final evaluation on test set ONLY for the best model
  - Partition-based test evaluation: Full / Claude / Non-Claude
  - Per-content-type F1 breakdown
  - Results saved to outputs/tables/classical_results.csv
  - Best model saved to outputs/checkpoints/best_classical/

Usage:
    uv run python -m src.models.train_classical
    uv run python -m src.models.train_classical --include-perplexity
    uv run python -m src.models.train_classical --feature-set handcrafted
    uv run python -m src.models.train_classical --models logreg xgb
"""

import argparse
import copy
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURE_DIR = Path("data/features")
FINAL_DIR = Path("data/final")
CHECKPOINT_DIR = Path("outputs/checkpoints/best_classical")
TABLE_DIR = Path("outputs/tables")

# ---------------------------------------------------------------------------
# Perplexity / burstiness feature names — excluded in no-perplexity mode
# ---------------------------------------------------------------------------
PPL_FEATURE_NAMES = {
    "ppl_mean",
    "ppl_sent_mean",
    "ppl_sent_std",
    "ppl_sent_min",
    "ppl_sent_max",
    "burstiness",
}


# ---------------------------------------------------------------------------
# Test partition helpers
# ---------------------------------------------------------------------------
def _load_test_metadata() -> pd.DataFrame:
    """Load test metadata for partition-based evaluation."""
    df = pd.read_parquet(FEATURE_DIR / "test_features.parquet")
    meta_cols = ["label", "content_type", "origin_model", "dataset_source"]
    if "_companion" in df.columns:
        meta_cols.append("_companion")
    return df[meta_cols].copy()


def _partition_masks(test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Build boolean masks for test sub-partitions.

    Claude partition   = Claude LLM records + human companion records.
    Non-Claude partition = everything else.
    """
    is_claude_llm = test_df["origin_model"].str.contains(
        "claude", case=False, na=False
    ).values

    if "_companion" in test_df.columns:
        is_companion = test_df["_companion"].fillna(False).astype(bool).values
    else:
        # Fallback: read _companion from JSONL
        recs = []
        with open(FINAL_DIR / "test.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        is_companion = np.array([r.get("_companion", False) for r in recs])
        logger.warning(
            "_companion not in parquet — loaded from JSONL. "
            "Re-run extract.py to fix."
        )

    claude_partition = is_claude_llm | is_companion
    non_claude_partition = ~claude_partition

    return {
        "full": np.ones(len(test_df), dtype=bool),
        "claude": claude_partition,
        "non_claude": non_claude_partition,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_handcrafted(
    split: str, feat_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    path = FEATURE_DIR / f"{split}_features.parquet"
    df = pd.read_parquet(path)
    cols = [c for c in feat_names if c in df.columns]
    X = df[cols].fillna(0.0).values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].values
    return X, y


def load_tfidf(split: str, kind: str) -> sp.csr_matrix:
    path = FEATURE_DIR / f"tfidf_{kind}" / f"{split}.npz"
    return sp.load_npz(path)


# ---------------------------------------------------------------------------
# Feature set assembly
# ---------------------------------------------------------------------------
def build_feature_sets(
    feat_names: list[str],
    include_ppl: bool,
) -> dict[str, dict]:
    """
    Returns a dict of feature_set_name → {X_train, X_val, X_test, y_*,
    scaler (or None), is_sparse}.
    """
    hc_names = [
        f for f in feat_names if include_ppl or f not in PPL_FEATURE_NAMES
    ]
    logger.info(
        f"Hand-crafted features: {len(hc_names)} "
        f"({'with' if include_ppl else 'WITHOUT'} perplexity)"
    )

    X_hc_tr, y_tr = load_handcrafted("train", hc_names)
    X_hc_va, y_va = load_handcrafted("val", hc_names)
    X_hc_te, y_te = load_handcrafted("test", hc_names)

    sc = StandardScaler()
    X_hc_tr_s = sc.fit_transform(X_hc_tr)
    X_hc_va_s = sc.transform(X_hc_va)
    X_hc_te_s = sc.transform(X_hc_te)

    X_word_tr = load_tfidf("train", "word")
    X_word_va = load_tfidf("val", "word")
    X_word_te = load_tfidf("test", "word")
    X_char_tr = load_tfidf("train", "char")
    X_char_va = load_tfidf("val", "char")
    X_char_te = load_tfidf("test", "char")

    def to_sparse(arr):
        return sp.csr_matrix(arr.astype(np.float32))

    sets = {
        "handcrafted": {
            "X_tr": X_hc_tr_s,
            "X_va": X_hc_va_s,
            "X_te": X_hc_te_s,
            "y_tr": y_tr,
            "y_va": y_va,
            "y_te": y_te,
            "scaler": sc,
            "is_sparse": False,
            "n_features": X_hc_tr_s.shape[1],
        },
        "tfidf_word": {
            "X_tr": X_word_tr,
            "X_va": X_word_va,
            "X_te": X_word_te,
            "y_tr": y_tr,
            "y_va": y_va,
            "y_te": y_te,
            "scaler": None,
            "is_sparse": True,
            "n_features": X_word_tr.shape[1],
        },
        "tfidf_char": {
            "X_tr": X_char_tr,
            "X_va": X_char_va,
            "X_te": X_char_te,
            "y_tr": y_tr,
            "y_va": y_va,
            "y_te": y_te,
            "scaler": None,
            "is_sparse": True,
            "n_features": X_char_tr.shape[1],
        },
        "hc_plus_word": {
            "X_tr": sp.hstack([to_sparse(X_hc_tr_s), X_word_tr]),
            "X_va": sp.hstack([to_sparse(X_hc_va_s), X_word_va]),
            "X_te": sp.hstack([to_sparse(X_hc_te_s), X_word_te]),
            "y_tr": y_tr,
            "y_va": y_va,
            "y_te": y_te,
            "scaler": sc,
            "is_sparse": True,
            "n_features": X_hc_tr_s.shape[1] + X_word_tr.shape[1],
        },
        "hc_plus_char": {
            "X_tr": sp.hstack([to_sparse(X_hc_tr_s), X_char_tr]),
            "X_va": sp.hstack([to_sparse(X_hc_va_s), X_char_va]),
            "X_te": sp.hstack([to_sparse(X_hc_te_s), X_char_te]),
            "y_tr": y_tr,
            "y_va": y_va,
            "y_te": y_te,
            "scaler": sc,
            "is_sparse": True,
            "n_features": X_hc_tr_s.shape[1] + X_char_tr.shape[1],
        },
        "hc_plus_word_char": {
            "X_tr": sp.hstack(
                [to_sparse(X_hc_tr_s), X_word_tr, X_char_tr]
            ),
            "X_va": sp.hstack(
                [to_sparse(X_hc_va_s), X_word_va, X_char_va]
            ),
            "X_te": sp.hstack(
                [to_sparse(X_hc_te_s), X_word_te, X_char_te]
            ),
            "y_tr": y_tr,
            "y_va": y_va,
            "y_te": y_te,
            "scaler": sc,
            "is_sparse": True,
            "n_features": (
                X_hc_tr_s.shape[1]
                + X_word_tr.shape[1]
                + X_char_tr.shape[1]
            ),
        },
    }
    return sets


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def get_models() -> dict:
    return {
        "logreg": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="saga",
            n_jobs=-1,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "xgboost": _make_xgb(),
    }


def _make_xgb():
    try:
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        logger.warning("xgboost not available — skipping XGBClassifier")
        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X, y_true, split_name: str) -> dict:
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return {
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
    }


# ---------------------------------------------------------------------------
# Single experiment: one model × one feature set
# ---------------------------------------------------------------------------
def run_experiment(
    model_name: str,
    model,
    fs_name: str,
    fs: dict,
    eval_test: bool = False,
) -> tuple[dict, object]:
    t0 = time.time()

    X_tr = fs["X_tr"]
    X_va = fs["X_va"]
    X_te = fs["X_te"]

    if model_name == "random_forest" and fs["is_sparse"]:
        X_tr = X_tr.toarray()
        X_va = X_va.toarray()
        X_te = X_te.toarray() if eval_test else X_te

    if model_name == "xgboost" and hasattr(model, "fit"):
        model.fit(
            X_tr, fs["y_tr"], eval_set=[(X_va, fs["y_va"])], verbose=False
        )
    else:
        model.fit(X_tr, fs["y_tr"])

    elapsed = time.time() - t0

    val_metrics = evaluate(model, X_va, fs["y_va"], "val")

    result = {
        "model": model_name,
        "feature_set": fs_name,
        "n_features": fs["n_features"],
        "train_time_s": round(elapsed, 1),
        **{
            f"val_{k}": round(v, 4)
            for k, v in val_metrics.items()
            if k != "split"
        },
    }

    if eval_test:
        if model_name == "random_forest" and fs["is_sparse"]:
            X_te = X_te if isinstance(X_te, np.ndarray) else X_te.toarray()
        test_metrics = evaluate(model, X_te, fs["y_te"], "test")
        result.update(
            {
                f"test_{k}": round(v, 4)
                for k, v in test_metrics.items()
                if k != "split"
            }
        )
    return result, model


# ---------------------------------------------------------------------------
# Full test evaluation + report
# ---------------------------------------------------------------------------
def full_test_report(
    model, fs: dict, model_name: str, fs_name: str
) -> None:
    """
    Detailed test evaluation with partition breakdown:
      - Full test set
      - Claude partition (Claude LLM + human companion)
      - Non-Claude partition (seen generators + their human counterparts)
      - Per-content-type F1
    """
    X_te = fs["X_te"]
    y_te = fs["y_te"]

    if model_name == "random_forest" and fs["is_sparse"]:
        X_te = X_te.toarray()

    y_pred = model.predict(X_te)
    try:
        y_prob = model.predict_proba(X_te)[:, 1]
    except Exception:
        y_prob = None

    test_meta = _load_test_metadata()
    masks = _partition_masks(test_meta)

    # --- Partition reports ---
    for part_name, mask in masks.items():
        n = mask.sum()
        if n == 0:
            continue

        y_t = y_te[mask]
        y_p = y_pred[mask]
        n_human = (y_t == 0).sum()
        n_llm = (y_t == 1).sum()

        print(f"\n{'=' * 64}")
        print(
            f"  TEST — {part_name.upper()} "
            f"(n={n:,}, human={n_human:,}, llm={n_llm:,})"
        )
        print(f"  Model: {model_name} | Features: {fs_name}")
        print("=" * 64)

        if n_human == 0 or n_llm == 0:
            acc = accuracy_score(y_t, y_p)
            print(f"  Only one class present — Accuracy: {acc:.4f}")
            continue

        print(
            classification_report(
                y_t,
                y_p,
                target_names=["Human (0)", "LLM (1)"],
                digits=4,
            )
        )

        if y_prob is not None:
            auc = roc_auc_score(y_t, y_prob[mask])
            print(f"  ROC-AUC: {auc:.4f}")

    # --- Per-content-type F1 ---
    print(f"\n{'=' * 64}")
    print("  PER-CONTENT-TYPE F1 (full test set)")
    print("=" * 64)
    print(f"  {'content_type':<30} {'n':>6} {'F1':>8} {'Acc':>8}")
    print(f"  {'-' * 30} {'-' * 6} {'-' * 8} {'-' * 8}")

    content_types = test_meta["content_type"].values
    for ct in sorted(set(content_types)):
        ct_mask = content_types == ct
        y_t_ct = y_te[ct_mask]
        y_p_ct = y_pred[ct_mask]
        n_ct = ct_mask.sum()

        if len(set(y_t_ct)) < 2:
            f1_ct = float("nan")
        else:
            f1_ct = f1_score(y_t_ct, y_p_ct, zero_division=0)
        acc_ct = accuracy_score(y_t_ct, y_p_ct)
        print(f"  {ct:<30} {n_ct:>6,} {f1_ct:>8.4f} {acc_ct:>8.4f}")

    # --- Save predictions for error analysis ---
    pred_dir = Path("outputs/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_df = test_meta.copy()
    pred_df["y_pred"] = y_pred
    if y_prob is not None:
        pred_df["y_prob"] = y_prob
    pred_df["correct"] = y_te == y_pred
    pred_path = pred_dir / f"test_preds_{model_name}_{fs_name}.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"  Predictions saved → {pred_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    feature_set_filter: list[str] | None,
    model_filter: list[str] | None,
    include_ppl: bool,
) -> None:
    logger.info("=" * 60)
    logger.info("Stage 7 — Classical ML Baseline")
    ppl_note = (
        "WITH perplexity"
        if include_ppl
        else "WITHOUT perplexity (no-ppl version)"
    )
    logger.info(f"Feature mode: {ppl_note}")
    logger.info("=" * 60)

    feat_path = FEATURE_DIR / "feature_names.json"
    if not feat_path.exists():
        logger.error("feature_names.json not found. Run extract.py first.")
        return
    feat_names = json.loads(feat_path.read_text())

    logger.info("\nBuilding feature sets ...")
    all_fs = build_feature_sets(feat_names, include_ppl=include_ppl)
    if feature_set_filter:
        all_fs = {k: v for k, v in all_fs.items() if k in feature_set_filter}
        logger.info(f"Feature sets selected: {list(all_fs.keys())}")

    all_models = get_models()
    all_models = {k: v for k, v in all_models.items() if v is not None}
    if model_filter:
        all_models = {
            k: v for k, v in all_models.items() if k in model_filter
        }

    logger.info(f"Models      : {list(all_models.keys())}")
    logger.info(f"Feature sets: {list(all_fs.keys())}")
    total = len(all_models) * len(all_fs)
    logger.info(f"Total experiments: {total}")

    results = []
    best_val_f1 = -1.0
    best_model = None
    best_fs_name = None
    best_model_name = None

    for fs_idx, (fs_name, fs) in enumerate(all_fs.items()):
        for m_idx, (model_name, model_proto) in enumerate(all_models.items()):
            model = copy.deepcopy(model_proto)
            exp_id = (
                f"[{fs_idx * len(all_models) + m_idx + 1}/{total}]"
            )
            logger.info(
                f"\n{exp_id} {model_name} | {fs_name} "
                f"({fs['n_features']:,} features)"
            )
            try:
                result, trained_model = run_experiment(
                    model_name, model, fs_name, fs, eval_test=False
                )
                results.append(result)
                logger.info(
                    f"  val  acc={result['val_accuracy']:.4f}  "
                    f"f1={result['val_f1']:.4f}  "
                    f"auc={result['val_roc_auc']:.4f}  "
                    f"({result['train_time_s']}s)"
                )
                if result["val_f1"] > best_val_f1:
                    best_val_f1 = result["val_f1"]
                    best_model = trained_model
                    best_fs_name = fs_name
                    best_model_name = model_name
            except Exception as e:
                logger.error(f"  Failed: {e}")
                import traceback

                traceback.print_exc()

    # Print comparison table
    results_df = pd.DataFrame(results)
    val_cols = [
        "model",
        "feature_set",
        "n_features",
        "val_accuracy",
        "val_f1",
        "val_roc_auc",
        "train_time_s",
    ]

    print("\n" + "=" * 80)
    print("  VAL RESULTS SUMMARY")
    print("=" * 80)
    print(
        results_df[val_cols]
        .sort_values("val_f1", ascending=False)
        .to_string(index=False)
    )

    # Best model → full test evaluation
    if best_model is not None:
        logger.info(
            f"\nBest model: {best_model_name} | {best_fs_name} "
            f"(val F1 = {best_val_f1:.4f})"
        )
        full_test_report(
            best_model, all_fs[best_fs_name], best_model_name, best_fs_name
        )

        # Re-run to get test metrics in results_df
        m_copy = copy.deepcopy(
            list(all_models.values())[
                list(all_models.keys()).index(best_model_name)
            ]
        )
        best_result, _ = run_experiment(
            best_model_name,
            m_copy,
            best_fs_name,
            all_fs[best_fs_name],
            eval_test=True,
        )
        logger.info(
            f"  test acc={best_result.get('test_accuracy', 'n/a'):.4f}  "
            f"f1={best_result.get('test_f1', 'n/a'):.4f}  "
            f"auc={best_result.get('test_roc_auc', 'n/a'):.4f}"
        )

    # Save results table
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_no_ppl" if not include_ppl else ""
    table_path = TABLE_DIR / f"classical_results{suffix}.csv"
    results_df.to_csv(table_path, index=False)
    logger.success(f"Results table → {table_path}")

    # Save best model
    if best_model is not None:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, CHECKPOINT_DIR / "model.joblib")
        meta = {
            "model_name": best_model_name,
            "feature_set": best_fs_name,
            "val_f1": best_val_f1,
            "ppl_included": include_ppl,
            "version": "no_ppl" if not include_ppl else "full",
        }
        (CHECKPOINT_DIR / "meta.json").write_text(
            json.dumps(meta, indent=2)
        )
        scaler = all_fs[best_fs_name].get("scaler")
        if scaler is not None:
            joblib.dump(scaler, CHECKPOINT_DIR / "scaler.joblib")
        logger.success(f"Best model saved → {CHECKPOINT_DIR}/")

    print("\n" + "=" * 60)
    print("  CLASSICAL ML BASELINE COMPLETE")
    print(f"  Version: {ppl_note}")
    print("=" * 60)
    print("Next: uv run python -m src.models.train_transformer\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classical ML baseline — Stage 7"
    )
    parser.add_argument(
        "--feature-set",
        nargs="*",
        choices=[
            "handcrafted",
            "tfidf_word",
            "tfidf_char",
            "hc_plus_word",
            "hc_plus_char",
            "hc_plus_word_char",
        ],
        default=None,
        metavar="FS",
        help="Feature set(s) to run (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=["logreg", "random_forest", "xgboost"],
        default=None,
        metavar="MODEL",
        help="Model(s) to run (default: all)",
    )
    parser.add_argument(
        "--include-perplexity",
        action="store_true",
        help="Include perplexity/burstiness features "
        "(requires full extract.py run)",
    )
    args = parser.parse_args()
    main(
        feature_set_filter=args.feature_set,
        model_filter=args.models,
        include_ppl=args.include_perplexity,
    )