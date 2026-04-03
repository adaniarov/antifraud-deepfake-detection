"""
visualize_classical.py — Stage 7b: visualisations for classical ML results.

Reads:
  outputs/tables/classical_results_no_ppl.csv  (or classical_results.csv)
  data/features/{train,val,test}_features.parquet
  outputs/checkpoints/best_classical/model.joblib

Produces (outputs/figures/):
  1. classical_val_comparison.png  — bar chart: all 18 experiments by val F1
  2. classical_confusion_matrix.png — confusion matrices: full / non-claude / claude
  3. classical_roc_curves.png      — ROC curves for top-5 models on val set
  4. classical_pr_curves.png       — Precision-Recall curves for top-5 on val set

Usage:
    uv run python -m src.features.visualize_classical
    uv run python -m src.features.visualize_classical --no-ppl
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FEATURE_DIR    = Path("data/features")
TABLE_DIR      = Path("outputs/tables")
CHECKPOINT_DIR = Path("outputs/checkpoints/best_classical")
FIG_DIR        = Path("outputs/figures")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_fig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.success(f"Saved → {path}")


def load_handcrafted(split, feat_names):
    df = pd.read_parquet(FEATURE_DIR / f"{split}_features.parquet")
    ppl_names = {"ppl_mean","ppl_sent_mean","ppl_sent_std",
                 "ppl_sent_min","ppl_sent_max","burstiness"}
    cols = [c for c in feat_names if c in df.columns and c not in ppl_names]
    X = df[cols].fillna(0).values.astype("float32")
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    return X, df["label"].values, df


def load_tfidf(split, kind):
    return sp.load_npz(FEATURE_DIR / f"tfidf_{kind}/{split}.npz")


def build_X(split, feat_names, scaler, fs_name):
    X_hc, y, df = load_handcrafted(split, feat_names)
    X_hc_s = scaler.transform(X_hc)

    def _sp(arr):
        return sp.csr_matrix(arr.astype("float32"))

    if fs_name == "handcrafted":
        return X_hc_s, y, df
    if fs_name == "tfidf_word":
        return load_tfidf(split,"word"), y, df
    if fs_name == "tfidf_char":
        return load_tfidf(split,"char"), y, df
    if fs_name == "hc_plus_word":
        return sp.hstack([_sp(X_hc_s), load_tfidf(split,"word")]), y, df
    if fs_name == "hc_plus_char":
        return sp.hstack([_sp(X_hc_s), load_tfidf(split,"char")]), y, df
    if fs_name == "hc_plus_word_char":
        return (sp.hstack([_sp(X_hc_s),
                           load_tfidf(split,"word"),
                           load_tfidf(split,"char")]), y, df)
    raise ValueError(fs_name)

# ---------------------------------------------------------------------------
# Figure 1: val comparison bar chart — all 18 experiments
# ---------------------------------------------------------------------------

def fig_val_comparison(results_df: pd.DataFrame) -> None:
    df = results_df.sort_values("val_f1", ascending=True).copy()

    labels  = df["model"] + "\n" + df["feature_set"].str.replace("_", "\n")
    f1      = df["val_f1"].values
    auc     = df["val_roc_auc"].values

    model_colours = {
        "logreg":         "#4C72B0",
        "random_forest":  "#55A868",
        "xgboost":        "#C44E52",
    }
    colours = [model_colours.get(m, "#888") for m in df["model"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # F1
    bars = ax1.barh(range(len(labels)), f1, color=colours,
                    edgecolor="white", linewidth=0.4)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=7.5)
    ax1.set_xlabel("Val F1-score", fontsize=11)
    ax1.set_title("Val F1 — all experiments", fontsize=12)
    ax1.set_xlim(0.83, 1.002)
    ax1.axvline(0.99, color="grey", lw=0.8, ls="--", alpha=0.6)
    for i, v in enumerate(f1):
        ax1.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=7)
    ax1.spines[["top","right"]].set_visible(False)

    # ROC-AUC
    bars2 = ax2.barh(range(len(labels)), auc, color=colours,
                     edgecolor="white", linewidth=0.4)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=7.5)
    ax2.set_xlabel("Val ROC-AUC", fontsize=11)
    ax2.set_title("Val ROC-AUC — all experiments", fontsize=12)
    ax2.set_xlim(0.89, 1.002)
    for i, v in enumerate(auc):
        ax2.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=7)
    ax2.spines[["top","right"]].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=m)
              for m, c in model_colours.items()]
    ax1.legend(handles=legend, fontsize=9, loc="lower right")

    fig.suptitle("Classical ML Baseline — Validation Results\n"
                 "(no perplexity features)", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "classical_val_comparison.png")

# ---------------------------------------------------------------------------
# Figure 2: confusion matrices — full / non-claude / claude
# ---------------------------------------------------------------------------

def fig_confusion_matrices(model, fs_name, feat_names, scaler) -> None:
    from sklearn.metrics import confusion_matrix  # type: ignore

    X_te, y_te, df_te = build_X("test", feat_names, scaler, fs_name)
    y_pred = model.predict(X_te)

    claude_mask     = df_te["origin_model"].str.contains("claude",
                                                          case=False, na=False).values
    non_claude_mask = ~claude_mask

    subsets = [
        ("Full test\n(n=3,494,  13%/87%)", y_te, y_pred),
        (f"Non-Claude\n(n={non_claude_mask.sum()},  50%/50%)",
         y_te[non_claude_mask], y_pred[non_claude_mask]),
        (f"Claude only\n(n={claude_mask.sum()},  held-out)",
         y_te[claude_mask],     y_pred[claude_mask]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (title, yt, yp) in zip(axes, subsets):
        if len(yt) == 0:
            ax.set_visible(False)
            continue
        cm = confusion_matrix(yt, yp)
        # Normalise by row (recall-normalised)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Human", "LLM"], fontsize=10)
        ax.set_yticklabels(["Human", "LLM"], fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(title, fontsize=10, pad=8)

        for i in range(2):
            for j in range(2):
                raw  = cm[i, j]
                norm = cm_norm[i, j]
                colour = "white" if norm > 0.6 else "black"
                ax.text(j, i, f"{norm:.2%}\n(n={raw:,})",
                        ha="center", va="center",
                        fontsize=9, color=colour, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Confusion Matrices — XGBoost | {fs_name}\n"
                 "(row-normalised, i.e. recall per class)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, "classical_confusion_matrix.png")

# ---------------------------------------------------------------------------
# Figure 3 & 4: ROC and PR curves — top-5 configs on val set
# ---------------------------------------------------------------------------

def fig_roc_pr_curves(results_df, feat_names) -> None:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve  # type: ignore
    from sklearn.preprocessing import StandardScaler

    # Top-5 by val F1
    top5 = results_df.nlargest(5, "val_f1")

    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr,  ax_pr  = plt.subplots(figsize=(7, 6))

    colours = plt.cm.tab10(np.linspace(0, 0.9, len(top5)))

    for (_, row), colour in zip(top5.iterrows(), colours):
        model_name = row["model"]
        fs_name    = row["feature_set"]

        # Load fresh data + retrain (we need probabilities on val set)
        X_hc_tr, y_tr, _  = load_handcrafted("train", feat_names)
        X_hc_va, y_va, _  = load_handcrafted("val",   feat_names)
        sc = StandardScaler()
        X_hc_tr_s = sc.fit_transform(X_hc_tr)
        X_hc_va_s = sc.transform(X_hc_va)

        def _sp(arr):
            return sp.csr_matrix(arr.astype("float32"))

        def _X(split, X_hc_s):
            if fs_name == "handcrafted":
                return X_hc_s
            elif fs_name == "tfidf_word":
                return load_tfidf(split,"word")
            elif fs_name == "tfidf_char":
                return load_tfidf(split,"char")
            elif fs_name == "hc_plus_word":
                return sp.hstack([_sp(X_hc_s), load_tfidf(split,"word")])
            elif fs_name == "hc_plus_char":
                return sp.hstack([_sp(X_hc_s), load_tfidf(split,"char")])
            else:
                return sp.hstack([_sp(X_hc_s), load_tfidf(split,"word"),
                                  load_tfidf(split,"char")])

        X_tr_final = _X("train", X_hc_tr_s)
        X_va_final = _X("val",   X_hc_va_s)

        import copy
        if model_name == "logreg":
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(C=1.0, max_iter=1000,
                                   solver="saga", n_jobs=-1, random_state=42)
        elif model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                       random_state=42)
        else:
            from xgboost import XGBClassifier
            m = XGBClassifier(n_estimators=200, max_depth=6,
                              learning_rate=0.1, verbosity=0,
                              random_state=42, eval_metric="logloss")

        if model_name == "random_forest" and sp.issparse(X_tr_final):
            X_tr_final = X_tr_final.toarray()
            X_va_final = X_va_final.toarray()

        if model_name == "xgboost":
            m.fit(X_tr_final, y_tr, verbose=False)
        else:
            m.fit(X_tr_final, y_tr)

        y_prob = m.predict_proba(X_va_final)[:, 1]
        label  = f"{model_name} | {fs_name}\n(F1={row['val_f1']:.4f})"

        # ROC
        fpr, tpr, _ = roc_curve(y_va, y_prob)
        roc_auc     = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=colour, lw=1.5,
                    label=f"{label}  AUC={roc_auc:.4f}")

        # PR
        prec, rec, _ = precision_recall_curve(y_va, y_prob)
        pr_auc       = auc(rec, prec)
        ax_pr.plot(rec, prec, color=colour, lw=1.5,
                   label=f"{label}  AUC={pr_auc:.4f}")

    # ROC
    ax_roc.plot([0,1],[0,1],"k--",lw=0.8,label="Random")
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate",  fontsize=11)
    ax_roc.set_title("ROC Curves — Top-5 models (val set)", fontsize=12)
    ax_roc.legend(fontsize=7.5, loc="lower right")
    ax_roc.spines[["top","right"]].set_visible(False)
    fig_roc.tight_layout()
    save_fig(fig_roc, "classical_roc_curves.png")

    # PR
    ax_pr.set_xlabel("Recall",    fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title("Precision-Recall Curves — Top-5 models (val set)",
                    fontsize=12)
    ax_pr.legend(fontsize=7.5, loc="lower left")
    ax_pr.spines[["top","right"]].set_visible(False)
    fig_pr.tight_layout()
    save_fig(fig_pr, "classical_pr_curves.png")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(no_ppl: bool) -> None:
    suffix = "_no_ppl" if no_ppl else ""
    table_path = TABLE_DIR / f"classical_results{suffix}.csv"

    if not table_path.exists():
        logger.error(f"{table_path} not found. Run train_classical.py first.")
        return

    results_df = pd.read_csv(table_path)
    logger.info(f"Loaded {len(results_df)} experiment results from {table_path}")

    # Load feature names
    feat_names = json.loads((FEATURE_DIR / "feature_names.json").read_text())
    ppl_names  = {"ppl_mean","ppl_sent_mean","ppl_sent_std",
                  "ppl_sent_min","ppl_sent_max","burstiness"}
    feat_names = [f for f in feat_names if f not in ppl_names]

    # Load best model + scaler
    model_path  = CHECKPOINT_DIR / "model.joblib"
    scaler_path = CHECKPOINT_DIR / "scaler.joblib"
    meta_path   = CHECKPOINT_DIR / "meta.json"

    if not model_path.exists():
        logger.error(f"Best model not found at {model_path}. Run train_classical.py first.")
        return

    best_model  = joblib.load(model_path)
    scaler      = joblib.load(scaler_path) if scaler_path.exists() else None
    meta        = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    best_fs     = meta.get("feature_set", "hc_plus_word_char")
    best_mname  = meta.get("model_name",  "xgboost")

    logger.info(f"Best model: {best_mname} | {best_fs}")

    # Figure 1 — val comparison
    logger.info("Generating val comparison chart ...")
    fig_val_comparison(results_df)

    # Figure 2 — confusion matrices
    if scaler is not None:
        logger.info("Generating confusion matrices ...")
        fig_confusion_matrices(best_model, best_fs, feat_names, scaler)
    else:
        logger.warning("No scaler found — skipping confusion matrices")

    # Figures 3 & 4 — ROC / PR curves (retrains top-5, ~2 min)
    logger.info("Generating ROC + PR curves (retraining top-5 models) ...")
    fig_roc_pr_curves(results_df, feat_names)

    print("\n" + "=" * 52)
    print("  CLASSICAL ML VISUALISATIONS COMPLETE")
    print("=" * 52)
    for f in sorted(FIG_DIR.glob("classical_*.png")):
        print(f"    {f.name:<45} {f.stat().st_size//1024:>4} KB")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise classical ML results (Stage 7b)"
    )
    parser.add_argument(
        "--no-ppl", action="store_true", default=True,
        help="Use no-perplexity results table (default: True)",
    )
    args = parser.parse_args()
    main(no_ppl=args.no_ppl)