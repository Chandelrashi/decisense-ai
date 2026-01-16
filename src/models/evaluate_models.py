from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_business_workforce.csv"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

ASSETS_DIR.mkdir(exist_ok=True)


def load_metrics(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_feature_names(pipe) -> list[str]:
    pre = pipe.named_steps["preprocess"]
    cat = pre.named_transformers_["cat"]
    cat_names = list(cat.get_feature_names_out())
    num_names = list(pre.transformers_[1][2])  # ("num", ..., num_cols)
    return cat_names + num_names


def export_feature_importance(model_dict: dict, out_csv: Path):
    pipe = model_dict["pipeline"]
    model = pipe.named_steps["model"]

    feat_names = get_feature_names(pipe)

    importances = None
    if hasattr(model, "coef_"):
        importances = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    if importances is None:
        return

    df_imp = pd.DataFrame({"feature": feat_names, "importance": importances})
    df_imp["abs_importance"] = np.abs(df_imp["importance"])
    df_imp = df_imp.sort_values("abs_importance", ascending=False).drop(columns=["abs_importance"])
    df_imp.to_csv(out_csv, index=False)


def main():
    df = pd.read_csv(DATA_PATH)

    # Prepare X only once
    drop_cols = ["growth_target_hit", "high_attrition_risk", "expected_attrition_cost"]
    X = df.drop(columns=drop_cols)

    # Load models
    growth = load(MODELS_DIR / "growth_model.joblib")
    attr = load(MODELS_DIR / "attrition_model.joblib")

    # Build test splits per target (same X, different y)
    y_growth = df["growth_target_hit"].astype(int)
    y_attr = df["high_attrition_risk"].astype(int)

    Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(X, y_growth, test_size=0.2, random_state=42, stratify=y_growth)
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X, y_attr, test_size=0.2, random_state=42, stratify=y_attr)

    # Plot ROC curves
    plt.figure()
    RocCurveDisplay.from_estimator(growth["pipeline"], Xg_te, yg_te, name="Growth Model")
    RocCurveDisplay.from_estimator(attr["pipeline"], Xa_te, ya_te, name="Attrition Model")
    out_png = ASSETS_DIR / "roc_curves.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    # Feature importance exports
    export_feature_importance(growth, ASSETS_DIR / "growth_feature_importance.csv")
    export_feature_importance(attr, ASSETS_DIR / "attrition_feature_importance.csv")

    # Combine metrics
    growth_metrics = load_metrics(ASSETS_DIR / "growth_metrics.json")
    attr_metrics = load_metrics(ASSETS_DIR / "attrition_metrics.json")

    summary = {
        "n_rows": int(len(df)),
        "growth_target_hit_rate": float(y_growth.mean()),
        "high_attrition_risk_rate": float(y_attr.mean()),
        "models": [growth_metrics, attr_metrics],
        "artifacts": {
            "roc_plot": "assets/roc_curves.png",
            "growth_feature_importance": "assets/growth_feature_importance.csv",
            "attrition_feature_importance": "assets/attrition_feature_importance.csv",
        }
    }

    with open(ASSETS_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", out_png)
    print("Saved: assets/results.json")
    print("Saved: feature importance CSVs")


if __name__ == "__main__":
    main()
