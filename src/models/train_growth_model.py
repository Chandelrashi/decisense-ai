from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_business_workforce.csv"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

MODELS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

TARGET_COL = "growth_target_hit"
DROP_COLS = ["growth_target_hit", "high_attrition_risk", "expected_attrition_cost"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    return df


def build_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # Strong, fast baseline (works well on mixed features)
    model = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=300,
        random_state=42
    )

    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def main():
    df = load_data()

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=DROP_COLS)

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(cat_cols, num_cols)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_roc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_pr = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision")

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    test_roc = roc_auc_score(y_test, y_prob)
    test_pr = average_precision_score(y_test, y_prob)

    # Save model
    out_model = MODELS_DIR / "growth_model.joblib"
    dump(
        {
            "pipeline": pipe,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "target": TARGET_COL,
        },
        out_model
    )

    # Save metrics (growth only)
    metrics = {
        "model": "GrowthModel",
        "target": TARGET_COL,
        "cv_roc_auc_mean": float(np.mean(cv_roc)),
        "cv_roc_auc_std": float(np.std(cv_roc)),
        "cv_pr_auc_mean": float(np.mean(cv_pr)),
        "cv_pr_auc_std": float(np.std(cv_pr)),
        "test_roc_auc": float(test_roc),
        "test_pr_auc": float(test_pr),
    }

    with open(ASSETS_DIR / "growth_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:", out_model)
    print(
        f"GrowthModel: CV ROC-AUC {metrics['cv_roc_auc_mean']:.3f}±{metrics['cv_roc_auc_std']:.3f}, "
        f"Test ROC-AUC {metrics['test_roc_auc']:.3f}, Test PR-AUC {metrics['test_pr_auc']:.3f}"
    )


if __name__ == "__main__":
    main()
