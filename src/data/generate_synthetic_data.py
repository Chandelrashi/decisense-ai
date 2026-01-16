from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42

# Output paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

OUT_PATH = DATA_DIR / "synthetic_business_workforce.csv"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_synthetic_data(n: int = 3000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Categorical features (match your questionnaire) ---
    business_size = rng.choice(["1-10", "11-50", "51-200", "200+"], size=n, p=[0.25, 0.35, 0.25, 0.15])
    business_stage = rng.choice(["Early-stage", "Growth", "Scaling", "Mature"], size=n, p=[0.25, 0.35, 0.25, 0.15])
    industry = rng.choice(["Tech", "Services", "Retail", "Finance", "Other"], size=n, p=[0.30, 0.25, 0.15, 0.15, 0.15])

    revenue_trend = rng.choice(
        ["Declining", "Flat", "Growing_slowly", "Growing_fast"],
        size=n,
        p=[0.15, 0.30, 0.35, 0.20]
    )

    marketing_spend_pct = rng.choice(["<5", "5-10", "10-20", ">20"], size=n, p=[0.25, 0.35, 0.25, 0.15])
    acquisition_trend = rng.choice(["Declining", "Stable", "Increasing"], size=n, p=[0.20, 0.45, 0.35])
    pricing_strategy = rng.choice(["Low-cost", "Market-average", "Premium"], size=n, p=[0.25, 0.55, 0.20])
    capacity_utilisation = rng.choice(["Under-utilised", "Optimal", "Over-stretched"], size=n, p=[0.20, 0.55, 0.25])

    attrition_rate_bucket = rng.choice(["<5", "5-10", "10-20", ">20"], size=n, p=[0.25, 0.35, 0.25, 0.15])
    skill_gap_level = rng.choice(["Fully covered", "Some skill gaps", "Significant gaps"], size=n, p=[0.45, 0.40, 0.15])
    salary_competitiveness = rng.choice(["Below market", "Market-aligned", "Above market"], size=n, p=[0.30, 0.55, 0.15])
    workload_level = rng.choice(["Sustainable", "High", "Unsustainable"], size=n, p=[0.40, 0.45, 0.15])

    risk_tolerance = rng.choice(["Low", "Medium", "High"], size=n, p=[0.30, 0.50, 0.20])
    budget_flexibility = rng.choice(["Very limited", "Moderate", "Flexible"], size=n, p=[0.25, 0.55, 0.20])
    hiring_feasibility = rng.choice(["Easy", "Moderate", "Difficult"], size=n, p=[0.25, 0.50, 0.25])

    # --- Numeric-ish features ---
    # Workforce size correlates with business size
    size_to_mean = {"1-10": 8, "11-50": 35, "51-200": 120, "200+": 450}
    workforce_size = np.array([max(3, int(rng.normal(size_to_mean[s], size_to_mean[s] * 0.15))) for s in business_size])

    # --- Scenario levers (these become knobs in the simulator later) ---
    planned_marketing_change = rng.choice(["-10", "0", "+5", "+10", "+20"], size=n, p=[0.10, 0.35, 0.25, 0.20, 0.10])
    planned_retention_invest = rng.choice(["0", "+3", "+5", "+10"], size=n, p=[0.40, 0.25, 0.25, 0.10])
    hiring_plan = rng.choice(["Freeze", "Replace only", "Net new hires"], size=n, p=[0.20, 0.55, 0.25])

    # --- Map categories to numeric signals for target generation ---
    # (These are transparent rules + noise = defensible synthetic dataset)
    stage_score = pd.Series(business_stage).map({"Early-stage": 0.2, "Growth": 0.6, "Scaling": 0.5, "Mature": 0.3}).to_numpy()
    trend_score = pd.Series(revenue_trend).map({"Declining": -0.6, "Flat": -0.1, "Growing_slowly": 0.3, "Growing_fast": 0.7}).to_numpy()
    mkt_score = pd.Series(marketing_spend_pct).map({"<5": -0.3, "5-10": 0.1, "10-20": 0.4, ">20": 0.6}).to_numpy()
    acq_score = pd.Series(acquisition_trend).map({"Declining": -0.4, "Stable": 0.0, "Increasing": 0.4}).to_numpy()
    cap_score = pd.Series(capacity_utilisation).map({"Under-utilised": 0.2, "Optimal": 0.4, "Over-stretched": -0.4}).to_numpy()
    price_score = pd.Series(pricing_strategy).map({"Low-cost": 0.1, "Market-average": 0.2, "Premium": 0.15}).to_numpy()

    attr_bucket_score = pd.Series(attrition_rate_bucket).map({"<5": -0.4, "5-10": -0.1, "10-20": 0.3, ">20": 0.7}).to_numpy()
    gap_score = pd.Series(skill_gap_level).map({"Fully covered": -0.2, "Some skill gaps": 0.2, "Significant gaps": 0.6}).to_numpy()
    salary_score = pd.Series(salary_competitiveness).map({"Below market": 0.5, "Market-aligned": 0.0, "Above market": -0.2}).to_numpy()
    workload_score = pd.Series(workload_level).map({"Sustainable": -0.2, "High": 0.3, "Unsustainable": 0.8}).to_numpy()

    risk_score = pd.Series(risk_tolerance).map({"Low": -0.2, "Medium": 0.0, "High": 0.2}).to_numpy()
    budget_score = pd.Series(budget_flexibility).map({"Very limited": -0.3, "Moderate": 0.1, "Flexible": 0.3}).to_numpy()
    hire_feas_score = pd.Series(hiring_feasibility).map({"Easy": 0.2, "Moderate": 0.0, "Difficult": -0.2}).to_numpy()

    # Scenario levers as numeric deltas
    mkt_delta = pd.Series(planned_marketing_change).map({"-10": -0.3, "0": 0.0, "+5": 0.15, "+10": 0.25, "+20": 0.4}).to_numpy()
    retention_delta = pd.Series(planned_retention_invest).map({"0": 0.0, "+3": 0.10, "+5": 0.18, "+10": 0.35}).to_numpy()
    hire_plan_delta = pd.Series(hiring_plan).map({"Freeze": -0.25, "Replace only": 0.0, "Net new hires": 0.2}).to_numpy()

    # --- Target 1: Growth target hit (>=10% growth in 90 days) ---
    # Growth is helped by positive trend, acquisition, capacity (not overstretched), marketing, budget,
    # and hiring; harmed by high attrition risk factors (workload, skill gaps).
    growth_logit = (
        0.8 * trend_score
        + 0.6 * acq_score
        + 0.4 * cap_score
        + 0.35 * mkt_score
        + 0.25 * price_score
        + 0.25 * budget_score
        + 0.20 * stage_score
        + 0.35 * mkt_delta
        + 0.20 * hire_plan_delta
        - 0.35 * workload_score
        - 0.25 * gap_score
        - 0.20 * attr_bucket_score
        + 0.10 * risk_score
        + rng.normal(0, 0.35, size=n)  # noise
    )
    p_growth = sigmoid(growth_logit)
    growth_target_hit = rng.binomial(1, p_growth)

    # --- Target 2: High attrition risk ---
    # Attrition rises with workload, low salary competitiveness, skill gaps, prior attrition bucket;
    # reduced by retention investment and better hiring feasibility (less strain).
    attrition_logit = (
        0.85 * workload_score
        + 0.55 * salary_score
        + 0.45 * gap_score
        + 0.50 * attr_bucket_score
        - 0.55 * retention_delta
        - 0.15 * hire_feas_score
        + 0.15 * (cap_score < 0).astype(float)  # overstretched adds risk
        + rng.normal(0, 0.35, size=n)
    )
    p_attr = sigmoid(attrition_logit)
    high_attrition_risk = rng.binomial(1, p_attr)

    # Optional: expected attrition cost proxy (simple, for decision ranking)
    # Assumes cost grows with workforce size and high attrition risk probability
    expected_attrition_cost = (workforce_size * 1200) * p_attr  # arbitrary but consistent scaling

    df = pd.DataFrame({
        "business_size": business_size,
        "business_stage": business_stage,
        "industry": industry,
        "revenue_trend": revenue_trend,
        "marketing_spend_pct": marketing_spend_pct,
        "acquisition_trend": acquisition_trend,
        "pricing_strategy": pricing_strategy,
        "capacity_utilisation": capacity_utilisation,

        "workforce_size": workforce_size,
        "attrition_rate_bucket": attrition_rate_bucket,
        "skill_gap_level": skill_gap_level,
        "salary_competitiveness": salary_competitiveness,
        "workload_level": workload_level,

        "risk_tolerance": risk_tolerance,
        "budget_flexibility": budget_flexibility,
        "hiring_feasibility": hiring_feasibility,

        "planned_marketing_change": planned_marketing_change,
        "planned_retention_invest": planned_retention_invest,
        "hiring_plan": hiring_plan,

        "growth_target_hit": growth_target_hit,
        "high_attrition_risk": high_attrition_risk,
        "expected_attrition_cost": expected_attrition_cost.round(2),
    })

    return df


def main():
    df = generate_synthetic_data(n=3000, seed=RANDOM_SEED)
    df.to_csv(OUT_PATH, index=False)

    # Quick sanity checks (prints for dev, not required in app)
    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Growth target hit rate: {df['growth_target_hit'].mean():.3f}")
    print(f"High attrition risk rate: {df['high_attrition_risk'].mean():.3f}")


if __name__ == "__main__":
    main()
