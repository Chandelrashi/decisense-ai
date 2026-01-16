from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import streamlit as st
from joblib import load

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# Ensure repo root is on the Python path (important for Streamlit Cloud)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.decision.scenario_simulator import default_scenarios, generate_scenario_inputs
from src.decision.ranker import rank_scenarios


# ----------------------------
# Paths
# ----------------------------
MODELS_DIR = PROJECT_ROOT / "models"
GROWTH_MODEL_PATH = MODELS_DIR / "growth_model.joblib"
ATTRITION_MODEL_PATH = MODELS_DIR / "attrition_model.joblib"


# ----------------------------
# Metric explanations (for UI + PDF)
# ----------------------------
METRIC_EXPLANATIONS = {
    "Growth Probability": "Estimated likelihood of achieving a positive growth outcome under the scenario (higher is better).",
    "Attrition Risk": "Estimated likelihood of elevated employee attrition under the scenario (lower is better).",
    "Utility Score": "Overall ranking score combining growth benefit, attrition trade-off, and cost/effort. Higher means better recommendation.",
    "Δ Growth": "Change in growth probability compared to the baseline (current plan). Positive is good.",
    "Δ Attrition": "Change in attrition risk compared to the baseline. Negative is good (lower attrition).",
    "Cost Index": "Relative cost/effort estimate for implementing the scenario (higher means more cost/effort).",
}


# ----------------------------
# UI helpers
# ----------------------------
def pretty_label(x: str) -> str:
    """Turn snake_case / kebab-case into human-friendly Title Case."""
    return (
        str(x)
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .title()
    )


def fmt_pct(x) -> str:
    """Format 0-1 as percent; safely handles NaNs/None."""
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "-"
        return f"{float(x) * 100:.0f}%"
    except Exception:
        return "-"


def inject_css():
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1100px; }
          .section-card { background: #fff; padding: 18px; border-radius: 18px; border: 1px solid #eee; }
          .muted { color: #6b7280; font-size: 0.95rem; }
          div[data-testid="stMetric"] { background: #ffffff; padding: 14px; border-radius: 16px; border: 1px solid #eee; }
          div[data-testid="stMetricLabel"] { font-size: 0.9rem; }
          div[data-testid="stMetricValue"] { font-size: 1.6rem; }
          hr { margin: 1.25rem 0; }
        </style>
        """,
        unsafe_allow_html=True
    )


# ----------------------------
# Model loading
# ----------------------------
@st.cache_resource
def load_models():
    """Load trained pipelines once and cache them.

    Supports:
    - joblib file contains a Pipeline directly
    - joblib file contains dict {"pipeline": Pipeline, ...}
    """
    if not GROWTH_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {GROWTH_MODEL_PATH}")
    if not ATTRITION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {ATTRITION_MODEL_PATH}")

    growth_obj = load(GROWTH_MODEL_PATH)
    attr_obj = load(ATTRITION_MODEL_PATH)

    growth_pipe = growth_obj.get("pipeline") if isinstance(growth_obj, dict) else growth_obj
    attr_pipe = attr_obj.get("pipeline") if isinstance(attr_obj, dict) else attr_obj

    if not hasattr(growth_pipe, "predict_proba"):
        raise TypeError(f"Growth model is not a classifier pipeline. Type={type(growth_pipe)}")
    if not hasattr(attr_pipe, "predict_proba"):
        raise TypeError(f"Attrition model is not a classifier pipeline. Type={type(attr_pipe)}")

    return growth_pipe, attr_pipe


def predict_probs(pipe, rows: List[Dict[str, Any]]) -> List[float]:
    df = pd.DataFrame(rows).drop(columns=["_scenario_name", "_scenario_description"], errors="ignore")
    return pipe.predict_proba(df)[:, 1].tolist()


# ----------------------------
# Inputs UI (polished)
# ----------------------------
def build_input_form() -> Dict[str, Any]:
    st.subheader("Inputs")
    tab1, tab2, tab3 = st.tabs(["Business", "Workforce", "Constraints & Plan"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            business_size = st.selectbox("Business size", ["1-10", "11-50", "51-200", "200+"])
            business_stage = st.selectbox("Business stage", ["Early-stage", "Growth", "Scaling", "Mature"])
            industry = st.selectbox("Industry", ["Tech", "Services", "Retail", "Finance", "Other"])
        with c2:
            revenue_trend = st.selectbox("Monthly revenue trend", ["Declining", "Flat", "Growing slowly", "Growing fast"])
            marketing_spend_pct = st.selectbox("Marketing spend (% of revenue)", ["<5", "5-10", "10-20", ">20"])
            acquisition_trend = st.selectbox("Customer acquisition trend", ["Declining", "Stable", "Increasing"])

        c3, c4 = st.columns(2)
        with c3:
            pricing_strategy = st.selectbox("Pricing strategy", ["Low-cost", "Market-average", "Premium"])
        with c4:
            capacity_utilisation = st.selectbox("Capacity utilisation", ["Under-utilised", "Optimal", "Over-stretched"])

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            workforce_size = st.number_input("Workforce size", min_value=3, max_value=5000, value=35, step=1)
            attrition_rate_bucket = st.selectbox("Attrition rate (last 12 months)", ["<5", "5-10", "10-20", ">20"])
        with c2:
            skill_gap_level = st.selectbox("Critical role coverage", ["Fully covered", "Some skill gaps", "Significant gaps"])
            salary_competitiveness = st.selectbox("Salary competitiveness", ["Below market", "Market-aligned", "Above market"])
            workload_level = st.selectbox("Average workload", ["Sustainable", "High", "Unsustainable"])

    with tab3:
        c1, c2, c3 = st.columns(3)
        with c1:
            risk_tolerance = st.selectbox("Risk tolerance", ["Low", "Medium", "High"])
        with c2:
            budget_flexibility = st.selectbox("Budget flexibility (next 90 days)", ["Very limited", "Moderate", "Flexible"])
        with c3:
            hiring_feasibility = st.selectbox("Hiring feasibility", ["Easy", "Moderate", "Difficult"])

        st.markdown("#### Current plan (baseline levers)")
        c4, c5, c6 = st.columns(3)
        with c4:
            planned_marketing_change = st.selectbox("Planned marketing change (%)", ["-10", "0", "+5", "+10", "+20"], index=1)
        with c5:
            planned_retention_invest = st.selectbox("Planned retention investment (%)", ["0", "+3", "+5", "+10"], index=0)
        with c6:
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


def clean_ranked_df(ranked: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and format values for nicer display."""
    df = ranked.copy()

    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].apply(pretty_label)

    rename_map = {
        "scenario": "Scenario",
        "growth_probability": "Growth Probability",
        "attrition_risk": "Attrition Risk",
        "delta_growth": "Δ Growth",
        "delta_attrition_risk": "Δ Attrition",
        "cost_index": "Cost Index",
        "utility": "Utility Score",
        "rationale": "Why this is recommended",
        "description": "Description",
        "planned_marketing_change": "Planned Marketing Change (%)",
        "planned_retention_invest": "Planned Retention Investment (%)",
        "hiring_plan": "Hiring Plan",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["Growth Probability", "Attrition Risk", "Utility Score", "Cost Index", "Δ Growth", "Δ Attrition"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def render_top_card(top_row: pd.Series):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Top recommendation")
    st.markdown(f"**{top_row['Scenario']}**")
    if "Why this is recommended" in top_row and pd.notna(top_row["Why this is recommended"]):
        st.markdown(f"<span class='muted'>{top_row['Why this is recommended']}</span>", unsafe_allow_html=True)
    if "Description" in top_row and pd.notna(top_row["Description"]):
        st.markdown("---")
        st.markdown(top_row["Description"])
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# PDF report generator
# ----------------------------
def generate_pdf_report(
    user_inputs: dict,
    top_row: dict,
    ranked_df: pd.DataFrame,
    explanations: dict,
) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x = 2 * cm
    y = height - 2 * cm

    def line(text: str, dy: int = 14):
        nonlocal y
        c.drawString(x, y, text)
        y -= dy
        if y < 2 * cm:
            c.showPage()
            y = height - 2 * cm

    # Title
    c.setFont("Helvetica-Bold", 16)
    line("DeciSense AI — Decision Report", dy=22)

    c.setFont("Helvetica", 10)
    line("This report summarises your inputs, the recommended scenario, and how to interpret the outputs.")
    line("")

    # Inputs
    c.setFont("Helvetica-Bold", 12)
    line("1) Inputs")
    c.setFont("Helvetica", 10)
    for k, v in user_inputs.items():
        line(f"- {k}: {v}")
    line("")

    # Top recommendation
    c.setFont("Helvetica-Bold", 12)
    line("2) Top Recommendation")
    c.setFont("Helvetica", 10)
    for k, v in top_row.items():
        line(f"- {k}: {v}")
    line("")

    # Explanations
    c.setFont("Helvetica-Bold", 12)
    line("3) How to interpret the metrics")
    c.setFont("Helvetica", 10)
    for k, v in explanations.items():
        line(f"- {k}: {v}")
    line("")

    # Ranked scenarios (top 10)
    c.setFont("Helvetica-Bold", 12)
    line("4) Ranked scenarios (top 10)")
    c.setFont("Helvetica", 9)

    preview = ranked_df.head(10).copy()
    cols = [c for c in ["Scenario", "Growth Probability", "Attrition Risk", "Utility Score", "Cost Index"] if c in preview.columns]
    line("Columns: " + ", ".join(cols))
    line("")

    for _, r in preview.iterrows():
        parts = []
        for col in cols:
            val = r[col]
            if col in ["Growth Probability", "Attrition Risk"] and pd.notna(val):
                parts.append(f"{col}: {fmt_pct(val)}")
            elif pd.isna(val):
                parts.append(f"{col}: -")
            else:
                parts.append(f"{col}: {val}")
        line(" | ".join(parts), dy=12)

    c.showPage()
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


def main():
    st.set_page_config(page_title="DeciSense AI", page_icon="🧠", layout="wide")
    inject_css()

    st.title("DeciSense AI — Growth & Workforce Decision Intelligence")
    st.markdown(
        "<div class='muted'>A decision-support MVP that compares scenarios using Growth Probability, Attrition Risk, "
        "utility-based ranking, and clear rationale.</div>",
        unsafe_allow_html=True
    )

    if not GROWTH_MODEL_PATH.exists() or not ATTRITION_MODEL_PATH.exists():
        st.error("Models not found. Ensure models/*.joblib exist in the repo.")
        st.stop()

    with st.sidebar:
        st.header("Controls")
        top_n = st.slider("How many scenarios to show?", min_value=5, max_value=30, value=10, step=1)
        show_details = st.checkbox("Show scenario lever details", value=True)
        st.caption("Tip: Use your baseline plan, then compare recommended scenarios.")

    base_input = build_input_form()

    st.markdown("---")
    run = st.button("Generate recommendations", type="primary")

    if run:
        with st.spinner("Evaluating scenarios..."):
            growth_pipe, attr_pipe = load_models()

            scenarios = default_scenarios()
            scenario_rows = generate_scenario_inputs(base_input, scenarios)

            growth_probs = predict_probs(growth_pipe, scenario_rows)
            attr_probs = predict_probs(attr_pipe, scenario_rows)

            ranked = rank_scenarios(scenario_rows, growth_probs, attr_probs)

        display_df = clean_ranked_df(ranked)

        top = display_df.iloc[0]
        render_top_card(top)

        c1, c2, c3 = st.columns(3)
        c1.metric("Growth Probability", fmt_pct(top.get("Growth Probability")))
        c2.metric("Attrition Risk", fmt_pct(top.get("Attrition Risk")))
        c3.metric("Utility Score", f"{float(top.get('Utility Score', 0.0)):.3f}")

        # Explanation expander
        with st.expander("What do these numbers mean?"):
            for k, v in METRIC_EXPLANATIONS.items():
                st.markdown(f"**{k}:** {v}")

        st.markdown("---")

        # Charts
        st.subheader("Scenario comparison")

        if "Scenario" in display_df.columns and "Utility Score" in display_df.columns:
            top_chart = display_df.head(top_n)[["Scenario", "Utility Score"]].set_index("Scenario")
            st.bar_chart(top_chart)

        if "Growth Probability" in display_df.columns and "Attrition Risk" in display_df.columns:
            scatter_df = display_df.head(top_n)[["Scenario", "Growth Probability", "Attrition Risk"]].copy()
            scatter_df = scatter_df.rename(columns={
                "Growth Probability": "growth",
                "Attrition Risk": "attrition",
                "Scenario": "scenario",
            })
            st.markdown("##### Trade-off view (Growth vs Attrition)")
            st.scatter_chart(scatter_df, x="attrition", y="growth")

        # Table
        st.subheader("Ranked scenarios")

        show_cols = [c for c in [
            "Scenario",
            "Growth Probability",
            "Attrition Risk",
            "Δ Growth",
            "Δ Attrition",
            "Cost Index",
            "Utility Score",
            "Why this is recommended",
        ] if c in display_df.columns]

        table_df = display_df.copy()

        # Show Growth/Attrition as percentages in table
        if "Growth Probability" in table_df.columns:
            table_df["Growth Probability"] = table_df["Growth Probability"].apply(fmt_pct)
        if "Attrition Risk" in table_df.columns:
            table_df["Attrition Risk"] = table_df["Attrition Risk"].apply(fmt_pct)

        st.dataframe(table_df.head(top_n)[show_cols], use_container_width=True, hide_index=True)

        # Scenario lever details
        if show_details:
            detail_cols = [c for c in [
                "Scenario",
                "Planned Marketing Change (%)",
                "Planned Retention Investment (%)",
                "Hiring Plan",
            ] if c in display_df.columns]
            if detail_cols:
                with st.expander("Scenario lever details"):
                    st.dataframe(display_df.head(top_n)[detail_cols], use_container_width=True, hide_index=True)

        # Downloads (CSV + PDF)
        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="decisense_recommendations.csv",
            mime="text/csv",
        )

        friendly_inputs = {pretty_label(k): v for k, v in base_input.items()}

        pdf_top = {
            "Scenario": str(top.get("Scenario", "")),
            "Growth Probability": fmt_pct(top.get("Growth Probability")),
            "Attrition Risk": fmt_pct(top.get("Attrition Risk")),
            "Utility Score": f"{float(top.get('Utility Score', 0.0)):.3f}",
            "Why this is recommended": str(top.get("Why this is recommended", "")),
        }

        pdf_bytes = generate_pdf_report(
            user_inputs=friendly_inputs,
            top_row=pdf_top,
            ranked_df=display_df,
            explanations=METRIC_EXPLANATIONS,
        )

        st.download_button(
            "Download PDF decision report",
            data=pdf_bytes,
            file_name="decisense_decision_report.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Application failed to start.")
        st.exception(e)
