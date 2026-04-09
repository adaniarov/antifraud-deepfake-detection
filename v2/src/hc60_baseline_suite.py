"""
19 classical baselines on HC60 numeric features only (no TF-IDF).
Reproducible: random_state=42 where applicable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from scipy.special import expit

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

from src.core_hc60_features import HC60_FEATURE_NAMES, HC60_GROUP_INDICES


def _pos_proba(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(np.float64)
    z = model.decision_function(X)
    if z.ndim > 1:
        z = z[:, 1]
    return expit(z).astype(np.float64)


def _scale_pipe(est: Any) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


@dataclass(frozen=True)
class ExperimentSpec:
    exp_id: int
    name: str
    description: str
    builder: Callable[[], Any]
    use_scaler: bool
    feature_slice: tuple[int, int] | None = None  # end exclusive, indices into HC60_FEATURE_NAMES


def _feat_range(spec: ExperimentSpec) -> list[str]:
    if spec.feature_slice is None:
        return list(HC60_FEATURE_NAMES)
    lo, hi = spec.feature_slice
    return list(HC60_FEATURE_NAMES[lo:hi])


def build_experiments() -> list[ExperimentSpec]:
    rs = 42
    exps: list[ExperimentSpec] = [
        ExperimentSpec(
            1,
            "lr_l2_c1",
            "LogisticRegression C=1.0",
            lambda: LogisticRegression(
                max_iter=4000, random_state=rs, class_weight="balanced", solver="lbfgs"
            ),
            True,
        ),
        ExperimentSpec(
            2,
            "lr_l2_c01",
            "LogisticRegression C=0.1 (stronger regularization)",
            lambda: LogisticRegression(
                max_iter=4000, random_state=rs, class_weight="balanced", C=0.1, solver="lbfgs"
            ),
            True,
        ),
        ExperimentSpec(
            3,
            "linearsvc_calibrated",
            "LinearSVC + sigmoid calibration",
            lambda: CalibratedClassifierCV(
                LinearSVC(class_weight="balanced", random_state=rs, max_iter=5000), cv=3
            ),
            True,
        ),
        ExperimentSpec(
            4,
            "ridge_classifier",
            "RidgeClassifier (linear margin)",
            lambda: RidgeClassifier(class_weight="balanced", random_state=rs),
            True,
        ),
        ExperimentSpec(
            5,
            "decision_tree",
            "DecisionTree depth 12",
            lambda: DecisionTreeClassifier(
                max_depth=12, min_samples_leaf=5, random_state=rs, class_weight="balanced"
            ),
            False,
        ),
        ExperimentSpec(
            6,
            "rf_constrained",
            "RandomForest n=200 depth=12",
            lambda: RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                random_state=rs,
                class_weight="balanced",
                n_jobs=-1,
            ),
            False,
        ),
        ExperimentSpec(
            7,
            "rf_less_constrained",
            "RandomForest n=400 unrestricted depth",
            lambda: RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                random_state=rs,
                class_weight="balanced",
                n_jobs=-1,
            ),
            False,
        ),
        ExperimentSpec(
            8,
            "extra_trees",
            "ExtraTrees n=300",
            lambda: ExtraTreesClassifier(
                n_estimators=300, random_state=rs, class_weight="balanced", n_jobs=-1
            ),
            False,
        ),
    ]
    if XGBClassifier is None:
        raise ImportError("xgboost required for HC60 baseline suite")
    exps.extend(
        [
            ExperimentSpec(
                9,
                "xgb_conservative",
                "XGBoost conservative",
                lambda: XGBClassifier(
                    max_depth=3,
                    n_estimators=200,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=rs,
                    n_jobs=-1,
                    eval_metric="logloss",
                ),
                False,
            ),
            ExperimentSpec(
                10,
                "xgb_richer",
                "XGBoost richer",
                lambda: XGBClassifier(
                    max_depth=6,
                    n_estimators=400,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=rs,
                    n_jobs=-1,
                    eval_metric="logloss",
                ),
                False,
            ),
        ]
    )
    exps.extend(
        [
            ExperimentSpec(
                11,
                "hist_gradient_boosting",
                "sklearn HistGradientBoosting",
                lambda: HistGradientBoostingClassifier(
                    max_depth=6, max_iter=250, random_state=rs, class_weight="balanced"
                ),
                False,
            ),
            ExperimentSpec(
                12,
                "gradient_boosting",
                "sklearn GradientBoostingClassifier",
                lambda: GradientBoostingClassifier(
                    random_state=rs, max_depth=3, n_estimators=150, learning_rate=0.1
                ),
                False,
            ),
            ExperimentSpec(
                13,
                "adaboost",
                "AdaBoost shallow trees",
                lambda: AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=2, random_state=rs),
                    n_estimators=150,
                    random_state=rs,
                ),
                False,
            ),
            ExperimentSpec(
                14,
                "sgd_log",
                "SGDClassifier log_loss",
                lambda: SGDClassifier(
                    loss="log_loss", max_iter=3000, random_state=rs, class_weight="balanced"
                ),
                True,
            ),
            ExperimentSpec(
                15,
                "bagging_dt",
                "Bagging shallow DecisionTrees",
                lambda: BaggingClassifier(
                    estimator=DecisionTreeClassifier(max_depth=8, random_state=rs),
                    n_estimators=40,
                    random_state=rs,
                    n_jobs=-1,
                ),
                False,
            ),
            ExperimentSpec(
                16,
                "knn_15",
                "KNeighbors k=15",
                lambda: KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
                True,
            ),
        ]
    )
    # Ablations (17–19)
    lo_s = HC60_GROUP_INDICES["length_structure"][0]
    hi_p = HC60_GROUP_INDICES["punctuation"][1]
    lo_l = HC60_GROUP_INDICES["lexical_diversity"][0]
    hi_l = HC60_GROUP_INDICES["lexical_diversity"][1]
    exps.append(
        ExperimentSpec(
            17,
            "lr_structural_subset",
            "LogisticRegression on length+surface+punctuation indices",
            lambda: LogisticRegression(
                max_iter=4000, random_state=rs, class_weight="balanced", solver="lbfgs"
            ),
            True,
            feature_slice=(lo_s, hi_p),
        )
    )
    exps.append(
        ExperimentSpec(
            18,
            "lr_lexical_diversity_only",
            "LogisticRegression on lexical_diversity block only",
            lambda: LogisticRegression(
                max_iter=4000, random_state=rs, class_weight="balanced", solver="lbfgs"
            ),
            True,
            feature_slice=(lo_l, hi_l),
        )
    )
    exps.append(
        ExperimentSpec(
            19,
            "xgb_full_ablation_capstone",
            "Same as xgb_richer (#10); separate row for ablation suite reporting",
            lambda: XGBClassifier(
                max_depth=6,
                n_estimators=400,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=rs,
                n_jobs=-1,
                eval_metric="logloss",
            ),
            False,
        )
    )
    assert len(exps) == 19
    return exps


def _split_xy(
    df: pd.DataFrame, cols: list[str], mask: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    sub = df.loc[mask]
    X = sub[cols].to_numpy(dtype=np.float64)
    y = sub["label"].to_numpy(dtype=np.int64)
    return X, y


def _metrics_binary(y_true: np.ndarray, proba: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    y_hat = (proba >= thr).astype(np.int64)
    out: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_true, proba))
        if len(np.unique(y_true)) > 1
        else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat)),
    }
    return out


def run_one(
    df: pd.DataFrame,
    spec: ExperimentSpec,
    train_mask: pd.Series,
    val_mask: pd.Series,
    test_seen_mask: pd.Series,
    test_cb_mask: pd.Series,
    test_claude_only_mask: pd.Series | None,
) -> dict[str, Any]:
    cols = _feat_range(spec)
    X_tr, y_tr = _split_xy(df, cols, train_mask)
    X_va, y_va = _split_xy(df, cols, val_mask)
    est = spec.builder()
    model = _scale_pipe(est) if spec.use_scaler else est

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_time_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    n_inf = min(500, max(len(X_va), 1))
    _ = _pos_proba(model, X_va[:n_inf])
    infer_time_s = (time.perf_counter() - t1) / n_inf

    p_val = _pos_proba(model, X_va)
    row: dict[str, Any] = {
        "exp_id": spec.exp_id,
        "name": spec.name,
        "description": spec.description,
        "n_features": len(cols),
        "train_time_s": train_time_s,
        "infer_time_per_sample_s_est": infer_time_s,
    }
    row.update({f"val__{k}": v for k, v in _metrics_binary(y_va, p_val).items()})

    X_ts, y_ts = _split_xy(df, cols, test_seen_mask)
    p_ts = _pos_proba(model, X_ts)
    row.update({f"test_seen__{k}": v for k, v in _metrics_binary(y_ts, p_ts).items()})

    X_cb, y_cb = _split_xy(df, cols, test_cb_mask)
    p_cb = _pos_proba(model, X_cb)
    row.update({f"test_claude_binary__{k}": v for k, v in _metrics_binary(y_cb, p_cb).items()})

    if test_claude_only_mask is not None and test_claude_only_mask.any():
        X_co, y_co = _split_xy(df, cols, test_claude_only_mask)
        p_co = _pos_proba(model, X_co)
        # one-class LLM: llm_predicted_rate @ 0.5
        row["test_claude_only__llm_predicted_rate"] = float((p_co >= 0.5).mean())
        row["test_claude_only__n"] = int(len(y_co))
    else:
        row["test_claude_only__llm_predicted_rate"] = float("nan")
        row["test_claude_only__n"] = 0

    # threshold tuned on val (maximize F1 coarse grid)
    thrs = np.linspace(0.1, 0.9, 17)
    best_f1, best_t = -1.0, 0.5
    for t in thrs:
        f1v = f1_score(y_va, (p_val >= t).astype(int), zero_division=0)
        if f1v > best_f1:
            best_f1, best_t = float(f1v), float(t)
    row["val_tuned_threshold_f1"] = best_t
    row["test_claude_binary__f1_at_val_threshold"] = float(
        f1_score(y_cb, (p_cb >= best_t).astype(int), zero_division=0)
    )
    return row


def run_all(
    df: pd.DataFrame,
    *,
    train_mask: pd.Series | None = None,
) -> pd.DataFrame:
    if train_mask is None:
        train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_seen_mask = df["core_eval_slice"] == "test_seen"
    test_cb_mask = df["core_eval_slice"] == "test_claude_binary"
    co_mask = df["core_eval_slice"] == "test_claude_holdout"
    rows = []
    for spec in build_experiments():
        rows.append(
            run_one(
                df,
                spec,
                train_mask,
                val_mask,
                test_seen_mask,
                test_cb_mask,
                co_mask,
            )
        )
    return pd.DataFrame(rows)
