from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import streamlit as st
from joblib import load

# Ensure repo root is on the Python path (important for Streamlit Cloud)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.decision.scenario_simulator import default_scenarios, generate_scenario_inputs
from src.decision.ranker import rank_scenarios


MODELS_DIR = PROJECT_ROOT / "models"
GROWTH_MODEL_PATH = MODELS_DIR / "growth_model.joblib"
ATTRITION_MODEL_PATH = MODELS_DIR / "attrition_model.joblib"


@st.cache_resource
def load_models():
    if not GROWTH_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {GROWTH_MODEL_PATH}")
    if not ATTRITION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {ATTRITION_MODEL_PATH}")

    growth = load(GROWTH_MODEL_PATH)
    attr = load(ATTRITION_MODEL_PATH)

    return growth["pipeline"], attr["pipeline"]


def build_input_form() -> Dict[str, Any]:
    st.header("Input: Business & Workforce Signals")

    # Business context
    business_size = st.selectbox("Business size", ["1-10", "11-50", "51-200", "200+"])
    business_stage = st.selectbox("Business stage", ["Early-stage", "Growth", "Scaling", "Mature"])
    industry = st.selectbox("Industry", ["Tech", "Services", "Retail", "Finance", "Other"])
    revenue_trend = st.selectbox("Monthly revenue trend", ["Declining", "Flat", "Growing_slowly", "Growing_fast"])

    # Growth signals
    marketing_spend_pct = st.selectbox("Marketing spend (% of revenue)", ["<5", "5-10", "10-20", ">20"])
    acquisition_trend = st.selectbox("Customer acquisition trend", ["Declining", "Stable", "Increasing"])
    pricing_strategy = st.selectbox("Pricing strategy", ["Low-cost", "Market-average", "Premium"])
    capacity_utilisation = st.selectbox("Capacity utilisation", ["Under-utilised", "Optimal", "Over-stretched"])

    # Workforce signals
    workforce_size = st.number_input("Workforce size", min_value=3, max_value=5000, value=35, step=1)
    attrition_rate_bucket = st.selectbox("Attrition rate (last 12 months)", ["<5", "5-10", "10-20", ">20"])
    skill_gap_level = st.selectbox("Critical role coverage", ["Fully covered", "Some skill gaps", "Significant gaps"])
    salary_competitiveness = st.selectbox("Salary competitiveness", ["Below market", "Market-aligned", "Above market"])
    workload_level = st.selectbox("Average workload", ["Sustainable", "High", "Unsustainable"])

    # Risk & constraints
    risk_tolerance = st.selectbox("Risk tolerance", ["Low", "Medium", "High"])
    budget_flexibility = st.selectbox("Budget flexibility (next 90 days)", ["Very limited", "Moderate", "Flexible"])
    hiring_feasibility = st.selectbox("Hiring feasibility", ["Easy", "Moderate", "Difficult"])

    # Baseline scenario levers (user's current plan)
    st.subheader("Current Plan (Baseline Levers)")
    planned_marketing_change = st.selectbox("Planned marketing change", ["-10", "0", "+5", "+10", "+20"], index=1)
    planned_retention_invest = st.selectbox("Planned retention investment", ["0", "+3", "+5", "+10"], index=0)
    hiring_plan = st.selectbox("Hiring plan", ["Freeze", "Replace only", "Net new hires"], index=1)

    return {
        "business_size": business_size,
        "business_stage": business_stage,
        "industry": industry,
        "revenue_trend": revenue_trend,
        "marketing_spend_pct": marketing_spend_pct,
        "acquisition_trend": acquisition_trend,
        "pricing_strategy": pricing_strategy,
        "capacity_utilisation": capacity_utilisation,
        "workforce_size": int(workforce_size),
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
    }


def predict_probs(pipe, rows: List[Dict[str, Any]]) -> List[float]:
    df = pd.DataFrame(rows).drop(columns=["_scenario_name", "_scenario_description"], errors="ignore")
    return pipe.predict_proba(df)[:, 1].tolist()


def main():
    st.set_page_config(page_title="DeciSense AI", layout="wide")
    st.title("DeciSense AI — Growth & Workforce Decision Intelligence (MVP)")
    st.caption(
        "Decision-support tool: compares scenarios using Growth Probability, Attrition Risk, "
        "utility-based ranking, and clear rationale."
    )

    if not GROWTH_MODEL_PATH.exists() or not ATTRITION_MODEL_PATH.exists():
        st.error("Models not found. Ensure models/*.joblib exist in the repo.")
        st.stop()

    base_input = build_input_form()

    if st.button("Generate recommendations"):
        # Lazy-load only when needed (faster cold-start on Streamlit Cloud)
        growth_pipe, attr_pipe = load_models()

        scenarios = default_scenarios()
        scenario_rows = generate_scenario_inputs(base_input, scenarios)

        growth_probs = predict_probs(growth_pipe, scenario_rows)
        attr_probs = predict_probs(attr_pipe, scenario_rows)

        ranked = rank_scenarios(scenario_rows, growth_probs, attr_probs)

        st.subheader("Top Recommendation")
        top = ranked.iloc[0]
        st.success(f"**{top['scenario']}** — {top['rationale']}")
        st.write(top["description"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Growth Probability", f"{top['growth_probability']:.2f}", f"{top['delta_growth']:+.2f} vs baseline")
        c2.metric("Attrition Risk", f"{top['attrition_risk']:.2f}", f"{top['delta_attrition_risk']:+.2f} vs baseline")
        c3.metric("Utility Score", f"{top['utility']:.3f}")

        st.divider()
        st.subheader("Ranked Scenarios")
        st.dataframe(
            ranked[
                [
                    "scenario",
                    "growth_probability",
                    "attrition_risk",
                    "delta_growth",
                    "delta_attrition_risk",
                    "cost_index",
                    "utility",
                    "rationale",
                ]
            ],
            use_container_width=True,
        )

        st.subheader("Scenario Details (levers)")
        st.dataframe(
            ranked[
                [
                    "scenario",
                    "planned_marketing_change",
                    "planned_retention_invest",
                    "hiring_plan",
                ]
            ],
            use_container_width=True,
        )


if __name__ == "__main__":
    try:
        st.write("App initialised successfully.")
        main()
    except Exception as e:
        st.error("Application failed to start.")
        st.exception(e)


