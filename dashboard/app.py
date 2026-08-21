"""
SaaS Churn Intelligence Dashboard
Streamlit app — run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.clv import (
    add_unit_economics,
    calculate_clv,
    churn_reduction_scenario,
    plot_clv_by_contract,
    plot_clv_distribution,
    plot_ltv_cac_scatter,
    plot_payback_distribution,
    segment_by_clv,
    unit_economics_kpis,
)
from src.cohort import (
    build_cohort_table, pivot_retention, plot_cohort_heatmap, plot_retention_curves,
    compute_nrr_by_cohort_month, nrr_by_segment, overall_nrr_summary,
    plot_nrr_trend, plot_nrr_waterfall,
)
from src.mrr import (
    aggregate_mrr,
    build_monthly_mrr,
    build_mrr_decomposition,
    plot_churn_rate_trend,
    plot_mrr_decomposition,
    plot_mrr_trend,
)
from src.preprocessing import preprocess, get_model_features
from src.drift import compute_feature_drift, compute_performance_drift, should_retrain
from src.experiment import (
    simulate_retention_experiment, analyze_experiment,
    required_sample_size, summarize_result,
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SaaS Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px 20px;
        border-left: 4px solid #4C78A8;
    }
    .metric-card.red  { border-left-color: #e74c3c; }
    .metric-card.green{ border-left-color: #2ecc71; }
    .metric-card.orange{ border-left-color: #f39c12; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── data loading (cached so it only runs once) ────────────────────────────────
@st.cache_data
def load_data():
    data_path = ROOT / "data" / "raw" / "telco_churn.csv"
    if not data_path.exists():
        import subprocess
        subprocess.run(
            ["python", str(ROOT / "data" / "generate_data.py")], check=True
        )
    df = preprocess(data_path)
    return df


@st.cache_resource
def load_model():
    model_path = ROOT / "models" / "churn_model.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data
def get_scored_df(_df):
    """Add churn probability + CLV to the dataframe."""
    model = load_model()
    if model is None:
        # Fallback: use contract-level average churn rate as proxy
        churn_rates = _df.groupby("Contract")["Churn"].transform("mean")
        _df = _df.copy()
        _df["churn_probability"] = churn_rates
    else:
        X, _ = get_model_features(_df)
        _df = _df.copy()
        _df["churn_probability"] = model.predict_proba(X)[:, 1]

    _df = calculate_clv(_df)
    _df = add_unit_economics(_df)
    _df = segment_by_clv(_df)
    return _df


@st.cache_data
def get_mrr_data(_df):
    long = build_monthly_mrr(_df)
    monthly = aggregate_mrr(long)
    decomp = build_mrr_decomposition(monthly)
    return monthly, decomp


@st.cache_data
def get_cohort_data(_df):
    cohort_df = build_cohort_table(_df)
    matrix = pivot_retention(cohort_df, max_offset=24)
    return cohort_df, matrix


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 SaaS Churn Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "Cohort Analysis", "MRR & Revenue", "Churn Model", "Unit Economics",
         "Revenue Retention (NRR)", "Drift Monitor", "Retention Experiments"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        "Dataset: Synthetic Telco Churn (~7,000 customers)\n\n"
        "Models: Logistic Regression + XGBoost\n\n"
        "Built with Python · DuckDB · Streamlit"
    )

# ── load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    df = load_data()
    scored_df = get_scored_df(df)

# =============================================================================
# PAGE: OVERVIEW
# =============================================================================
if page == "Overview":
    st.title("Overview — Subscription Health")
    st.markdown(
        "High-level snapshot of the subscription portfolio. "
        "Use the tabs below to explore deeper."
    )

    # KPI row
    total = len(df)
    active = (df["Churn"] == 0).sum()
    churned = (df["Churn"] == 1).sum()
    churn_rate = churned / total
    total_mrr = df[df["Churn"] == 0]["MonthlyCharges"].sum()
    arpu = df[df["Churn"] == 0]["MonthlyCharges"].mean()
    kpis = unit_economics_kpis(scored_df)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Active Customers", f"{active:,}")
    col3.metric("Overall Churn Rate", f"{churn_rate:.1%}")
    col4.metric("Total MRR", f"${total_mrr:,.0f}")
    col5.metric("Avg CLV", f"${kpis['avg_clv']:,.0f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        # Churn rate by contract
        contract_churn = (
            df.groupby("Contract")["Churn"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "churn_rate", "count": "customers"})
        )
        fig = px.bar(
            contract_churn,
            x="Contract",
            y="churn_rate",
            color="Contract",
            text=contract_churn["churn_rate"].apply(lambda x: f"{x:.1%}"),
            title="Churn Rate by Contract Type",
            color_discrete_sequence=["#e74c3c", "#f39c12", "#2ecc71"],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            yaxis_tickformat=".0%", showlegend=False, height=380,
            yaxis_title="Churn Rate", xaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Monthly charges distribution by churn
        fig = px.histogram(
            df,
            x="MonthlyCharges",
            color=df["Churn"].map({0: "Active", 1: "Churned"}),
            nbins=50,
            barmode="overlay",
            opacity=0.7,
            title="Monthly Charges Distribution by Churn Status",
            labels={"MonthlyCharges": "Monthly Charges ($)", "color": "Status"},
            color_discrete_map={"Active": "#4C78A8", "Churned": "#e74c3c"},
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Churn by tenure band
    tenure_churn = (
        df.groupby("tenure_band", observed=True)["Churn"]
        .mean()
        .reset_index()
        .rename(columns={"Churn": "churn_rate"})
    )
    fig = px.bar(
        tenure_churn,
        x="tenure_band",
        y="churn_rate",
        title="Churn Rate by Customer Tenure",
        labels={"tenure_band": "Tenure Band", "churn_rate": "Churn Rate"},
        color="churn_rate",
        color_continuous_scale="RdYlGn_r",
        text=tenure_churn["churn_rate"].apply(lambda x: f"{x:.1%}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis_tickformat=".0%", showlegend=False,
        height=350, coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: COHORT ANALYSIS
# =============================================================================
elif page == "Cohort Analysis":
    st.title("Cohort Retention Analysis")
    st.markdown(
        "Each row is a group of customers who signed up in the same month. "
        "The values show what percentage of that group was still active N months later. "
        "**Dark green = high retention. Red = high churn.**"
    )

    with st.spinner("Building cohort table..."):
        cohort_df, matrix = get_cohort_data(df)

    st.plotly_chart(plot_cohort_heatmap(matrix), use_container_width=True)
    st.plotly_chart(plot_retention_curves(matrix, n_cohorts=6), use_container_width=True)

    # Retention benchmarks
    st.markdown("### Retention Benchmarks")
    avg_retention = matrix.mean()
    bench_months = [1, 3, 6, 12]
    cols = st.columns(len(bench_months))
    for col, m in zip(cols, bench_months):
        if m in avg_retention.index:
            col.metric(f"Month {m} Retention", f"{avg_retention[m]:.1%}")


# =============================================================================
# PAGE: MRR & REVENUE
# =============================================================================
elif page == "MRR & Revenue":
    st.title("MRR & Revenue Analysis")
    st.markdown(
        "Monthly Recurring Revenue trend and decomposition. "
        "**New MRR** = revenue from new customers. "
        "**Churned MRR** = revenue lost from cancelled subscriptions."
    )

    with st.spinner("Building MRR time series (this takes a moment)..."):
        monthly, decomp = get_mrr_data(df)

    # KPIs
    latest = monthly.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Month MRR", f"${latest['total_mrr']:,.0f}")
    col2.metric("Active Customers", f"{int(latest['active_customers']):,}")
    col3.metric("Monthly Churn Rate", f"{latest['churn_rate']:.1%}")

    st.plotly_chart(plot_mrr_trend(monthly), use_container_width=True)
    st.plotly_chart(plot_mrr_decomposition(decomp), use_container_width=True)
    st.plotly_chart(plot_churn_rate_trend(monthly), use_container_width=True)


# =============================================================================
# PAGE: CHURN MODEL
# =============================================================================
elif page == "Churn Model":
    st.title("Churn Prediction Model")
    st.markdown(
        "XGBoost classifier trained to predict which customers will churn. "
        "Use the filters to explore predicted churn risk across customer segments."
    )

    # Risk distribution
    fig = px.histogram(
        scored_df,
        x="churn_probability",
        color=scored_df["Churn"].map({0: "Stayed", 1: "Churned"}),
        nbins=50,
        barmode="overlay",
        opacity=0.7,
        title="Predicted Churn Probability Distribution",
        labels={"churn_probability": "Churn Probability", "color": "Actual Outcome"},
        color_discrete_map={"Stayed": "#4C78A8", "Churned": "#e74c3c"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # High-risk customers table
    st.markdown("### High-Risk Customer Segments")
    threshold = st.slider("Churn probability threshold", 0.3, 0.9, 0.6, 0.05)

    high_risk = scored_df[scored_df["churn_probability"] >= threshold][
        ["customerID", "Contract", "tenure", "MonthlyCharges",
         "churn_probability", "clv", "ltv_cac_ratio"]
    ].sort_values("churn_probability", ascending=False)

    col1, col2 = st.columns(2)
    col1.metric("High-Risk Customers", f"{len(high_risk):,}")
    col2.metric(
        "MRR at Risk",
        f"${high_risk['MonthlyCharges'].sum():,.0f}",
        help="Monthly revenue from high-risk customers",
    )

    st.dataframe(
        high_risk.head(50).style.format(
            {
                "churn_probability": "{:.1%}",
                "MonthlyCharges": "${:.2f}",
                "clv": "${:,.0f}",
                "ltv_cac_ratio": "{:.1f}x",
            }
        ),
        use_container_width=True,
    )

    # Churn risk by contract
    st.markdown("### Average Churn Risk by Segment")
    seg_col = st.selectbox("Segment by", ["Contract", "InternetService", "PaymentMethod"])

    # Map back original column names
    col_map = {
        "Contract": "Contract",
        "InternetService": "InternetService",
        "PaymentMethod": "PaymentMethod",
    }
    # These columns may have been one-hot encoded — use the original df
    orig_df = df.copy()
    orig_df["churn_probability"] = scored_df["churn_probability"].values

    seg_risk = (
        orig_df.groupby(seg_col)["churn_probability"]
        .mean()
        .reset_index()
        .sort_values("churn_probability", ascending=False)
    )
    fig = px.bar(
        seg_risk,
        x=seg_col,
        y="churn_probability",
        color="churn_probability",
        color_continuous_scale="RdYlGn_r",
        text=seg_risk["churn_probability"].apply(lambda x: f"{x:.1%}"),
        title=f"Average Predicted Churn Risk by {seg_col}",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis_tickformat=".0%", coloraxis_showscale=False,
        height=380, showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: UNIT ECONOMICS
# =============================================================================
elif page == "Unit Economics":
    st.title("Unit Economics — CLV & CAC Analysis")
    st.markdown(
        "Customer Lifetime Value (CLV) estimates the total revenue a customer "
        "will generate over their relationship with the company. "
        "Comparing CLV to the cost of acquiring that customer (CAC = $250 assumed) "
        "tells us which customer segments are profitable."
    )

    kpis = unit_economics_kpis(scored_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg CLV", f"${kpis['avg_clv']:,.0f}")
    col2.metric("Avg LTV:CAC", f"{kpis['avg_ltv_cac']:.1f}x")
    col3.metric("Avg Payback Period", f"{kpis['avg_payback_months']:.1f} mo")
    col4.metric("% with Healthy LTV:CAC (≥3x)", f"{kpis['pct_healthy_ltv_cac']:.1%}")

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(plot_clv_by_contract(scored_df), use_container_width=True)
    with col_right:
        st.plotly_chart(plot_clv_distribution(scored_df), use_container_width=True)

    st.plotly_chart(plot_ltv_cac_scatter(scored_df), use_container_width=True)
    st.plotly_chart(plot_payback_distribution(scored_df), use_container_width=True)

    # Scenario analysis
    st.markdown("### Scenario: What if we reduce churn?")
    st.markdown(
        "If we invested in a retention program for **month-to-month customers**, "
        "how much incremental CLV would we unlock? "
        "(Uses the Kaplan-Meier survival-based CLV method — see `src/clv.py` — "
        "not a naive `1 / churn_rate` estimate.)"
    )
    churn_reduction = st.slider(
        "Churn incidence reduction (percentage points)", 1, 20, 5
    )

    scenario = churn_reduction_scenario(
        scored_df, segment="Month-to-month", pct_point_reduction=churn_reduction / 100
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Avg CLV (M2M)", f"${scenario['clv_per_customer_old']:,.0f}")
    col2.metric(
        "New Avg CLV (M2M)",
        f"${scenario['clv_per_customer_new']:,.0f}",
        f"+${scenario['clv_per_customer_new'] - scenario['clv_per_customer_old']:,.0f}",
    )
    col3.metric("Total Portfolio Gain", f"${scenario['incremental_portfolio_clv']:,.0f}")

    st.caption(
        f"Expected lifetime: {scenario['expected_lifetime_old']:.1f} months → "
        f"{scenario['expected_lifetime_new']:.1f} months "
        f"({scenario['n_customers']:,} month-to-month customers)"
    )

# =============================================================================
# PAGE: REVENUE RETENTION (NRR)
# =============================================================================
elif page == "Revenue Retention (NRR)":
    st.title("Net Revenue Retention")
    st.markdown(
        "Customer-count retention (see **Cohort Analysis**) answers *what % of customers stayed*. "
        "NRR answers a different question: of the **revenue** a cohort started with, how much do "
        "we still have N months later — decomposed into expansion, contraction, and churn."
    )

    panel_path = ROOT / "data" / "raw" / "monthly_revenue_panel.csv"
    if not panel_path.exists():
        st.warning(
            "Monthly revenue panel not found. Run `python data/generate_monthly_panel.py` "
            "first, then reload this page."
        )
    else:
        panel = pd.read_csv(panel_path)

        nrr_df = compute_nrr_by_cohort_month(panel)
        summary = overall_nrr_summary(nrr_df, window=12)

        col1, col2, col3 = st.columns(3)
        col1.metric("Trailing 12mo Avg NRR", f"{summary['avg_nrr']:.1%}")
        col2.metric("Total Expansion MRR", f"${summary['total_expansion_mrr']:,.0f}")
        col3.metric("Total Churned MRR", f"${summary['total_churned_mrr']:,.0f}")

        st.plotly_chart(plot_nrr_trend(nrr_df), use_container_width=True)

        st.markdown("### NRR by Contract Segment")
        seg_nrr = nrr_by_segment(panel, df, segment_col="Contract")
        seg_rows = []
        for seg in seg_nrr["Contract"].unique():
            s = overall_nrr_summary(seg_nrr[seg_nrr["Contract"] == seg], window=12)
            seg_rows.append({"Contract": seg, "Avg NRR (trailing 12mo)": s["avg_nrr"]})
        seg_df = pd.DataFrame(seg_rows)
        st.dataframe(
            seg_df.style.format({"Avg NRR (trailing 12mo)": "{:.1%}"}),
            use_container_width=True, hide_index=True,
        )

        st.markdown("### MRR Bridge for a Selected Month")
        valid_months = nrr_df.dropna(subset=["nrr"])["month"].tolist()
        if valid_months:
            month_choice = st.slider("Month (since signup)", min(valid_months), max(valid_months), value=int(np.median(valid_months)))
            st.plotly_chart(plot_nrr_waterfall(nrr_df, month=month_choice), use_container_width=True)

# =============================================================================
# PAGE: DRIFT MONITOR
# =============================================================================
elif page == "Drift Monitor":
    st.title("Data Drift Monitor")
    st.markdown(
        "A model trained once goes stale silently as the real customer population shifts. "
        "This page checks whether a slice of \"new\" data still looks like the data the model "
        "was trained on — using Population Stability Index (PSI), the standard metric for this "
        "in credit-risk and churn-model monitoring."
    )

    st.markdown("### Simulate a shift to see the monitor in action")
    st.caption(
        "Since there's no real live feed here, apply a synthetic shift below to see PSI "
        "correctly flag it — the same validation approach used in notebook 10."
    )

    shift_pct = st.slider("Simulated MonthlyCharges shift (%)", -30, 50, 25)
    split_frac = st.slider("Baseline / new-data split", 0.4, 0.9, 0.7)

    split_point = int(len(df) * split_frac)
    baseline = df.iloc[:split_point].copy()
    new_arrivals = df.iloc[split_point:].copy()
    new_arrivals["MonthlyCharges"] = new_arrivals["MonthlyCharges"] * (1 + shift_pct / 100)

    drift_report = compute_feature_drift(
        baseline, new_arrivals,
        features=["MonthlyCharges", "tenure", "Contract", "InternetService", "SeniorCitizen"],
    )

    def _flag_color(flag):
        return {"stable": "🟢", "watch": "🟡", "alert": "🔴"}.get(flag, "")

    drift_report_display = drift_report.copy()
    drift_report_display["status"] = drift_report_display["flag"].apply(_flag_color)
    st.dataframe(
        drift_report_display[["feature", "psi", "flag", "status"]].style.format({"psi": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )

    decision = should_retrain(drift_report, performance_drift=None)
    if decision["should_retrain"]:
        st.error(f"**Retrain recommended.** Reasons: {'; '.join(decision['reasons'])}")
    else:
        st.success("No retrain triggered — all monitored features within stable/watch range.")

    fig = px.bar(
        drift_report, x="feature", y="psi", color="flag",
        color_discrete_map={"stable": "#2ecc71", "watch": "#f39c12", "alert": "#e74c3c"},
        title="Population Stability Index by Feature",
    )
    fig.add_hline(y=0.10, line_dash="dash", line_color="orange", annotation_text="watch threshold (0.10)")
    fig.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="alert threshold (0.25)")
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: RETENTION EXPERIMENTS
# =============================================================================
elif page == "Retention Experiments":
    st.title("Retention Intervention Experiments")
    st.markdown(
        "The churn model tells you **who** is likely to leave. This page tests whether a "
        "retention intervention (discount, proactive outreach, etc.) actually **works** — "
        "with proper experiment sizing and statistical analysis, not just a before/after guess."
    )

    st.markdown("### Step 1 — Size the experiment")
    high_risk = scored_df[scored_df["churn_probability"] >= scored_df["churn_probability"].quantile(0.70)]
    baseline_rate = high_risk["churn_probability"].mean()
    st.caption(f"Targeting the top 30% highest-risk customers (n={len(high_risk):,}, avg predicted churn probability {baseline_rate:.1%})")

    mde = st.slider("Minimum detectable effect (percentage points)", 2, 15, 8) / 100
    n_needed = required_sample_size(baseline_rate=baseline_rate, minimum_detectable_effect=mde)
    st.info(f"To reliably detect a **{mde:.0%}-point** reduction: **{n_needed:,} customers per arm** ({n_needed*2:,} total) needed.")

    st.markdown("### Step 2 — Simulate and analyze the experiment")
    st.caption("Since no real intervention data exists, this simulates a known true effect so the statistical test can be validated against ground truth.")

    true_effect = st.slider("True effect used in simulation (percentage points)", 2, 15, 8) / 100
    sample_size_choice = st.select_slider(
        "Experiment sample size (per arm, approx.)",
        options=[30, 100, 250, 500, 1000, "all available"],
        value="all available",
    )

    exp_population = high_risk if sample_size_choice == "all available" else high_risk.sample(
        n=min(sample_size_choice * 2, len(high_risk)), random_state=1
    )
    experiment_df = simulate_retention_experiment(exp_population, true_treatment_effect=true_effect, seed=42)
    result = analyze_experiment(experiment_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Control churn rate", f"{result.churn_rate_control:.1%}", f"n={result.n_control}")
    col2.metric("Treatment churn rate", f"{result.churn_rate_treatment:.1%}", f"n={result.n_treatment}")
    col3.metric(
        "Effect",
        f"{result.absolute_effect:+.1%} pts",
        "significant" if result.significant else "not significant",
        delta_color="normal" if result.significant else "off",
    )

    st.write(summarize_result(result))
    if n_needed > (result.n_control + result.n_treatment) / 2 and not result.significant:
        st.warning(
            f"This experiment's sample size ({result.n_control + result.n_treatment} total) is below "
            f"the {n_needed*2:,} needed to reliably detect a {mde:.0%}-point effect — a non-significant "
            f"result here may mean the test was underpowered, not that the intervention doesn't work."
        )

