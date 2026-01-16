from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class Weights:
    wg: float  # growth weight
    wr: float  # attrition risk weight
    wc: float  # cost weight


def infer_weights(risk_tolerance: str, budget_flexibility: str) -> Weights:
    """
    Simple, transparent weight rules (MVP).
    You can later learn these weights from preference data, but rules are more defensible early on.
    """
    # Base weights
    wg, wr, wc = 1.0, 1.0, 0.3

    # Risk tolerance adjustment
    if risk_tolerance == "Low":
        wr += 0.6
        wg -= 0.1
    elif risk_tolerance == "High":
        wg += 0.4
        wr -= 0.1

    # Budget adjustment
    if budget_flexibility == "Very limited":
        wc += 0.6
        wg -= 0.05
    elif budget_flexibility == "Flexible":
        wc -= 0.1

    return Weights(wg=wg, wr=wr, wc=max(0.05, wc))


def estimate_action_cost_index(row: Dict[str, Any]) -> float:
    """
    Cost proxy (0..1-ish). Transparent, not "real money".
    Used only for ranking scenarios.
    """
    mkt_map = {"-10": 0.0, "0": 0.1, "+5": 0.25, "+10": 0.45, "+20": 0.7}
    ret_map = {"0": 0.1, "+3": 0.25, "+5": 0.45, "+10": 0.75}
    hire_map = {"Freeze": 0.05, "Replace only": 0.2, "Net new hires": 0.55}

    mkt = mkt_map.get(str(row.get("planned_marketing_change", "0")), 0.1)
    ret = ret_map.get(str(row.get("planned_retention_invest", "0")), 0.1)
    hire = hire_map.get(str(row.get("hiring_plan", "Replace only")), 0.2)

    # average as simple cost index
    return float(np.clip((mkt + ret + hire) / 3.0, 0.0, 1.0))


def utility_score(
    growth_prob: float,
    attrition_prob: float,
    cost_index: float,
    weights: Weights
) -> float:
    """
    Higher is better.
    """
    return (weights.wg * growth_prob) - (weights.wr * attrition_prob) - (weights.wc * cost_index)


def make_rationale(row: Dict[str, Any], growth_prob: float, attr_prob: float) -> str:
    """
    Human-readable explanation (MVP rules).
    """
    parts = []
    if growth_prob >= 0.6:
        parts.append("strong growth potential")
    elif growth_prob >= 0.45:
        parts.append("moderate growth potential")
    else:
        parts.append("limited short-term growth potential")

    if attr_prob >= 0.5:
        parts.append("high workforce risk")
    elif attr_prob >= 0.3:
        parts.append("moderate workforce risk")
    else:
        parts.append("low workforce risk")

    # Highlight major levers
    if row.get("planned_marketing_change") in ["+10", "+20"]:
        parts.append("marketing-driven push")
    if row.get("planned_retention_invest") in ["+5", "+10"]:
        parts.append("retention stabiliser")
    if row.get("hiring_plan") == "Net new hires":
        parts.append("capacity expansion")
    if row.get("hiring_plan") == "Freeze":
        parts.append("cost containment")

    return ", ".join(parts).capitalize() + "."


def rank_scenarios(
    scenario_rows: List[Dict[str, Any]],
    growth_probs: List[float],
    attrition_probs: List[float]
) -> pd.DataFrame:
    """
    Returns a ranked table with utility and rationale.
    """
    if not (len(scenario_rows) == len(growth_probs) == len(attrition_probs)):
        raise ValueError("Scenario rows and probability lists must match lengths.")

    base = scenario_rows[0]  # baseline first by design
    weights = infer_weights(
        risk_tolerance=str(base.get("risk_tolerance", "Medium")),
        budget_flexibility=str(base.get("budget_flexibility", "Moderate")),
    )

    records = []
    for row, gp, ap in zip(scenario_rows, growth_probs, attrition_probs):
        cost_idx = estimate_action_cost_index(row)
        util = utility_score(gp, ap, cost_idx, weights)

        records.append({
            "scenario": row.get("_scenario_name", "Scenario"),
            "description": row.get("_scenario_description", ""),
            "growth_probability": float(gp),
            "attrition_risk": float(ap),
            "cost_index": float(cost_idx),
            "utility": float(util),
            "rationale": make_rationale(row, gp, ap),
            "planned_marketing_change": row.get("planned_marketing_change"),
            "planned_retention_invest": row.get("planned_retention_invest"),
            "hiring_plan": row.get("hiring_plan"),
        })

    df = pd.DataFrame(records)

    # Add deltas vs baseline (baseline assumed first)
    base_g = df.loc[0, "growth_probability"]
    base_a = df.loc[0, "attrition_risk"]
    df["delta_growth"] = df["growth_probability"] - base_g
    df["delta_attrition_risk"] = df["attrition_risk"] - base_a

    df = df.sort_values("utility", ascending=False).reset_index(drop=True)
    return df
