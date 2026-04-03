"""
visualize_features.py — Stage 6b: exploratory visualisations for Chapter 2.

Produces four publication-ready figures saved to outputs/figures/:

  1. tsne_tfidf.png
     t-SNE projection of word TF-IDF (train set, 3 000 sample).
     Colour = label (human / LLM). Useful for showing separability.

  2. tsne_handcrafted.png
     t-SNE projection of hand-crafted features (scaled).
     Colour = label. Marker shape = content_type.

  3. feature_importance.png
     Top-30 hand-crafted features ranked by XGBoost feature importance
     (trained on train, evaluated on val). Horizontal bar chart.

  4. feature_distributions.png
     Box plots of the 12 most discriminative features split by label.
     Shows WHY certain features separate classes.

Usage:
    uv run python -m src.features.visualize_features
    uv run python -m src.features.visualize_features --no-tsne   # skip slow t-SNE
    uv run python -m src.features.visualize_features --sample 2000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FEATURE_DIR = Path("data/features")
FIG_DIR     = Path("outputs/figures")

TRAIN_FEATS = FEATURE_DIR / "train_features.parquet"
VAL_FEATS   = FEATURE_DIR / "val_features.parquet"
FEAT_NAMES  = FEATURE_DIR / "feature_names.json"
TFIDF_TRAIN = FEATURE_DIR / "tfidf_word" / "train.npz"

# ---------------------------------------------------------------------------
# Colour palette — consistent across all figures
# ---------------------------------------------------------------------------

PALETTE = {
    "human": "#4C72B0",   # blue
    "llm":   "#DD8452",   # orange
}

CONTENT_TYPE_COLOURS = {
    "phishing":           "#e41a1c",
    "smishing":           "#ff7f00",
    "social_engineering": "#984ea3",
    "scam_419":           "#a65628",
    "bank_notification":  "#4daf4a",
    "financial_review":   "#377eb8",
    "legitimate":         "#999999",
    "financial_qa":       "#f781bf",
    "review":             "#a6cee3",
    "spam":               "#b2df8a",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_features(path: Path, feat_names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (full_df, X_array) where X contains only the ML feature columns."""
    df = pd.read_parquet(path)
    # Keep only feature columns that actually exist in the file
    cols = [c for c in feat_names if c in df.columns]
    X = df[cols].fillna(0.0).values.astype(np.float32)
    return df, X


def scale(X_train: np.ndarray, X_other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """StandardScaler fitted on train, applied to other."""
    from sklearn.preprocessing import StandardScaler  # type: ignore
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_other_s = sc.transform(X_other)
    return X_train_s, X_other_s


def run_tsne(X: np.ndarray, n_components: int = 2,
             perplexity: float = 40, random_state: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE  # type: ignore
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=random_state, n_iter=1000, verbose=0)
    return tsne.fit_transform(X)


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.success(f"Saved → {path}")

# ---------------------------------------------------------------------------
# Figure 1: t-SNE of TF-IDF features
# ---------------------------------------------------------------------------

def fig_tsne_tfidf(sample_n: int) -> None:
    if not TFIDF_TRAIN.exists():
        logger.warning(f"TF-IDF matrix not found: {TFIDF_TRAIN}. Skipping.")
        return

    import scipy.sparse as sp  # type: ignore

    logger.info("t-SNE (TF-IDF) ...")
    X_sparse = sp.load_npz(TFIDF_TRAIN)
    df_train = pd.read_parquet(TRAIN_FEATS)

    # Sample
    n      = min(sample_n, X_sparse.shape[0])
    rng    = np.random.default_rng(42)
    idx    = rng.choice(X_sparse.shape[0], size=n, replace=False)
    X_sub  = X_sparse[idx].toarray()
    labels = df_train["label"].values[idx]

    # Dimensionality reduction to 50 via SVD before t-SNE (standard practice)
    from sklearn.decomposition import TruncatedSVD  # type: ignore
    svd   = TruncatedSVD(n_components=50, random_state=42)
    X_svd = svd.fit_transform(X_sub)

    logger.info(f"  Running t-SNE on {n} samples (SVD→50 dims) ...")
    emb = run_tsne(X_svd, perplexity=40)

    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl, name, colour in [(0, "Human", PALETTE["human"]),
                               (1, "LLM",   PALETTE["llm"])]:
        mask = labels == lbl
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=colour, alpha=0.45, s=14, linewidths=0,
                   label=f"{name} (n={mask.sum():,})")

    ax.set_title("t-SNE: Word TF-IDF features (train sample)", fontsize=13, pad=10)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(framealpha=0.9, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    save_fig(fig, "tsne_tfidf.png")

# ---------------------------------------------------------------------------
# Figure 2: t-SNE of hand-crafted features, coloured by content_type
# ---------------------------------------------------------------------------

def fig_tsne_handcrafted(feat_names: list[str], sample_n: int) -> None:
    if not TRAIN_FEATS.exists():
        logger.warning("Hand-crafted features not found. Skipping.")
        return

    logger.info("t-SNE (hand-crafted features) ...")
    df_train, X_train = load_features(TRAIN_FEATS, feat_names)
    X_train_s, _ = scale(X_train, X_train)

    n   = min(sample_n, len(df_train))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df_train), size=n, replace=False)

    X_sub  = X_train_s[idx]
    labels = df_train["label"].values[idx]
    ctypes = df_train["content_type"].values[idx]

    logger.info(f"  Running t-SNE on {n} samples ...")
    emb = run_tsne(X_sub, perplexity=40)

    # Two subplots: left = by label, right = by content_type
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: human vs LLM
    for lbl, name, colour in [(0, "Human", PALETTE["human"]),
                               (1, "LLM",   PALETTE["llm"])]:
        mask = labels == lbl
        ax1.scatter(emb[mask, 0], emb[mask, 1],
                    c=colour, alpha=0.45, s=12, linewidths=0,
                    label=f"{name} (n={mask.sum():,})")
    ax1.set_title("Coloured by label", fontsize=12)
    ax1.legend(framealpha=0.9, fontsize=9)
    ax1.set_xticks([]); ax1.set_yticks([])

    # Right: by content type
    unique_ct = sorted(set(ctypes))
    for ct in unique_ct:
        colour = CONTENT_TYPE_COLOURS.get(ct, "#888888")
        mask   = ctypes == ct
        ax2.scatter(emb[mask, 0], emb[mask, 1],
                    c=colour, alpha=0.50, s=12, linewidths=0,
                    label=ct.replace("_", " "))
    ax2.set_title("Coloured by content type", fontsize=12)
    ax2.legend(fontsize=7, ncol=2, framealpha=0.9,
               loc="lower right", markerscale=1.5)
    ax2.set_xticks([]); ax2.set_yticks([])

    fig.suptitle("t-SNE: Hand-crafted stylometric + perplexity features (train sample)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "tsne_handcrafted.png")

# ---------------------------------------------------------------------------
# Figure 3: Feature importance (XGBoost on hand-crafted features)
# ---------------------------------------------------------------------------

def fig_feature_importance(feat_names: list[str], top_n: int = 30) -> None:
    if not TRAIN_FEATS.exists() or not VAL_FEATS.exists():
        logger.warning("Features not found. Skipping feature importance.")
        return

    from xgboost import XGBClassifier  # type: ignore

    logger.info("Training XGBoost for feature importance ...")
    df_train, X_train = load_features(TRAIN_FEATS, feat_names)
    df_val,   X_val   = load_features(VAL_FEATS,   feat_names)

    y_train = df_train["label"].values
    y_val   = df_val["label"].values

    # Replace any remaining NaN/Inf with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val   = np.nan_to_num(X_val,   nan=0.0, posinf=0.0, neginf=0.0)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    val_acc = (model.predict(X_val) == y_val).mean()
    logger.info(f"  XGBoost val accuracy (hand-crafted only): {val_acc:.3f}")

    # Feature importances
    importances = model.feature_importances_
    cols        = [c for c in feat_names if c in pd.read_parquet(TRAIN_FEATS).columns]
    imp_df      = pd.DataFrame({"feature": cols, "importance": importances})
    imp_df      = imp_df.sort_values("importance", ascending=False).head(top_n)

    # Assign colours by feature group
    def _group_colour(name):
        if any(x in name for x in ["ppl", "burst", "nll"]):
            return "#2ca02c"   # green = perplexity/burstiness
        if any(x in name for x in ["ttr","yule","mtld"]):
            return "#9467bd"   # purple = lexical diversity
        if name.startswith("pos") or "bigram" in name:
            return "#d62728"   # red = POS
        if any(x in name for x in ["conj","prep","pron","func","imperative","clause"]):
            return "#ff7f0e"   # orange = function words / syntax
        if any(x in name for x in ["punct","exclamation","question","ellipsis","dash","comma"]):
            return "#8c564b"   # brown = punctuation
        return "#1f77b4"       # blue = basic stats / other

    colours = [_group_colour(f) for f in imp_df["feature"]]

    fig, ax = plt.subplots(figsize=(9, 8))
    bars = ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1],
                   color=list(reversed(colours)), edgecolor="white", linewidth=0.5)
    ax.set_xlabel("XGBoost feature importance (gain)", fontsize=11)
    ax.set_title(f"Top-{top_n} hand-crafted features\n"
                 f"(XGBoost, val accuracy = {val_acc:.3f})", fontsize=12)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top","right"]].set_visible(False)

    # Legend for groups
    legend_items = [
        mpatches.Patch(color="#2ca02c", label="Perplexity / burstiness"),
        mpatches.Patch(color="#9467bd", label="Lexical diversity (TTR/MTLD/Yule)"),
        mpatches.Patch(color="#d62728", label="POS tags / bigrams"),
        mpatches.Patch(color="#ff7f0e", label="Function words / syntax"),
        mpatches.Patch(color="#8c564b", label="Punctuation"),
        mpatches.Patch(color="#1f77b4", label="Basic statistics"),
    ]
    ax.legend(handles=legend_items, fontsize=8, loc="lower right",
              framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "feature_importance.png")

    # Save importance table for thesis appendix
    imp_df.to_csv(FIG_DIR / "feature_importance.csv", index=False)
    logger.info(f"  Importance table → {FIG_DIR / 'feature_importance.csv'}")

# ---------------------------------------------------------------------------
# Figure 4: Feature distributions by label (box plots)
# ---------------------------------------------------------------------------

def fig_feature_distributions(feat_names: list[str], top_n: int = 12) -> None:
    if not TRAIN_FEATS.exists():
        logger.warning("Features not found. Skipping distributions.")
        return

    logger.info("Feature distribution box plots ...")
    df = pd.read_parquet(TRAIN_FEATS)
    cols = [c for c in feat_names if c in df.columns]

    # Select top-N most discriminative features by absolute mean difference
    human_df = df[df["label"] == 0][cols].fillna(0)
    llm_df   = df[df["label"] == 1][cols].fillna(0)
    diff     = (llm_df.mean() - human_df.mean()).abs()
    # Normalise by std to get standardised effect size
    pooled_std = pd.concat([human_df, llm_df]).std().replace(0, 1)
    effect     = (diff / pooled_std).sort_values(ascending=False)
    top_cols   = effect.head(top_n).index.tolist()

    n_cols = 4
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.5, n_rows * 3.2))
    axes = axes.flatten()

    for i, col in enumerate(top_cols):
        ax = axes[i]
        h_vals = human_df[col].dropna().values
        l_vals = llm_df[col].dropna().values

        # Clip outliers to 1st–99th percentile for readability
        lo = np.percentile(np.concatenate([h_vals, l_vals]), 1)
        hi = np.percentile(np.concatenate([h_vals, l_vals]), 99)
        h_vals = np.clip(h_vals, lo, hi)
        l_vals = np.clip(l_vals, lo, hi)

        bp = ax.boxplot(
            [h_vals, l_vals],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
            notch=False,
        )
        bp["boxes"][0].set_facecolor(PALETTE["human"] + "99")  # add alpha
        bp["boxes"][1].set_facecolor(PALETTE["llm"]   + "99")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Human", "LLM"], fontsize=9)
        ax.set_title(col.replace("_", " "), fontsize=9, pad=4)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7)

    # Hide unused subplots
    for j in range(len(top_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Top-{top_n} most discriminative features: Human vs LLM (train set)\n"
        "(clipped to 1st–99th percentile)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "feature_distributions.png")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_tsne: bool, sample_n: int) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not FEAT_NAMES.exists():
        logger.error(
            f"{FEAT_NAMES} not found.\n"
            "Run extract.py first:  uv run python -m src.features.extract"
        )
        return

    feat_names = json.loads(FEAT_NAMES.read_text())
    logger.info(f"Loaded {len(feat_names)} feature names from {FEAT_NAMES}")

    # Figure 3 and 4 don't need t-SNE — always run them
    fig_feature_importance(feat_names, top_n=30)
    fig_feature_distributions(feat_names, top_n=12)

    if run_tsne:
        fig_tsne_handcrafted(feat_names, sample_n=sample_n)
        fig_tsne_tfidf(sample_n=sample_n)
    else:
        logger.info("t-SNE skipped (--no-tsne).")

    print("\n" + "=" * 52)
    print("  VISUALISATIONS COMPLETE")
    print("=" * 52)
    print(f"  Output directory : {FIG_DIR}")
    for f in sorted(FIG_DIR.glob("*.png")):
        size_kb = f.stat().st_size // 1024
        print(f"    {f.name:<40} {size_kb:>5} KB")
    print("=" * 52)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate exploratory visualisations (Stage 6b)"
    )
    parser.add_argument(
        "--no-tsne", action="store_true",
        help="Skip t-SNE plots (slow, ~5 min). Feature importance + distributions still run.",
    )
    parser.add_argument(
        "--sample", type=int, default=3000, metavar="N",
        help="Number of points to sample for t-SNE (default: 3000)",
    )
    args = parser.parse_args()
    main(run_tsne=not args.no_tsne, sample_n=args.sample)