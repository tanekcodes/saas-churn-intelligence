"""
Customer Lifetime Value (CLV) and unit economics calculations.

CLV formula used here:
    CLV = Monthly Revenue x Gross Margin x Expected Lifetime (months)

Expected lifetime is estimated with a Kaplan-Meier survival curve fit on
each contract segment's observed `tenure` (duration) and `Churn` (event)
columns, rather than inverting the churn-classification model's predicted
probability. The classifier predicts a *cross-sectional* "has this customer
churned as of today" label, not a *monthly hazard rate* -- treating it as
1/churn_prob understates real customer lifetimes by roughly 15-20x, since a
0.6-0.8 "will this customer ever churn" probability is not the same thing
as a 60-80% chance of churning in any given month.

Unit economics metrics:
  - CAC  : Customer Acquisition Cost (assumed constant for illustration)
  - LTV  : same as CLV
  - LTV:CAC ratio  -- healthy SaaS = 3:1 or better
  - Payback period -- months until CAC is recovered
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter

# Assumed gross margin for a SaaS business (revenue - hosting/support costs)
GROSS_MARGIN = 0.75

# Assumed customer acquisition cost per customer (marketing + sales spend / new customers)
# In a real project this would come from finance data; here we use a reasonable assumption
CAC = 250.0

# Observation window cap (months) for restricted mean survival time
MAX_TENURE_MONTHS = 72


# ── survival-based expected lifetime ────────────────────────────────────────

def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


def fit_survival_curve(
    durations: pd.Series, events: pd.Series, tmax: int = MAX_TENURE_MONTHS
) -> pd.Series:
    """Fit a Kaplan-Meier survival curve. `events`=1 means churn observed,
    0 means censored (still active at last observation)."""
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events, timeline=np.arange(0, tmax + 1))
    return kmf.survival_function_.iloc[:, 0]


def expected_lifetime_months(
    durations: pd.Series, events: pd.Series, tmax: int = MAX_TENURE_MONTHS
) -> float:
    """Restricted mean survival time (RMST): area under the KM curve up to
    `tmax` months. This is the expected customer lifetime in months."""
    sf = fit_survival_curve(durations, events, tmax)
    return _trapz(sf.values, sf.index.values)


def lifetime_by_segment(
    df: pd.DataFrame,
    segment_col: str = "Contract",
    duration_col: str = "tenure",
    churn_col: str = "Churn",
    tmax: int = MAX_TENURE_MONTHS,
) -> dict[str, float]:
    """Expected lifetime (months) per segment, via Kaplan-Meier."""
    events = (df[churn_col] == "Yes").astype(int)
    out = {}
    for seg, grp_idx in df.groupby(segment_col).groups.items():
        grp = df.loc[grp_idx]
        out[seg] = expected_lifetime_months(grp[duration_col], events.loc[grp_idx], tmax)
    return out


# ── CLV calculation ───────────────────────────────────────────────────────────

def calculate_clv(
    df: pd.DataFrame,
    segment_col: str = "Contract",
    duration_col: str = "tenure",
    churn_col: str = "Churn",
    monthly_charge_col: str = "MonthlyCharges",
    gross_margin: float = GROSS_MARGIN,
    tmax: int = MAX_TENURE_MONTHS,
) -> pd.DataFrame:
    """
    Add expected_lifetime_months and clv columns, using segment-level
    Kaplan-Meier expected lifetime (not the classifier's churn probability).
    """
    df = df.copy()
    lifetimes = lifetime_by_segment(df, segment_col, duration_col, churn_col, tmax)
    df["expected_lifetime_months"] = df[segment_col].map(lifetimes)
    df["clv"] = df[monthly_charge_col] * gross_margin * df["expected_lifetime_months"]
    return df


def add_unit_economics(df: pd.DataFrame, cac: float = CAC) -> pd.DataFrame:
    """Add CAC, LTV:CAC ratio, and payback period columns."""
    df = df.copy()
    df["cac"] = cac
    df["ltv_cac_ratio"] = df["clv"] / cac
    df["payback_months"] = cac / (df["MonthlyCharges"] * GROSS_MARGIN)
    return df


def segment_by_clv(df: pd.DataFrame) -> pd.DataFrame:
    """Label customers as High / Medium / Low value based on CLV quartiles."""
    df = df.copy()
    q33 = df["clv"].quantile(0.33)
    q66 = df["clv"].quantile(0.66)
    df["clv_segment"] = pd.cut(
        df["clv"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["Low Value", "Medium Value", "High Value"],
    )
    return df


# ── scenario analysis ─────────────────────────────────────────────────────────

def churn_reduction_scenario(
    df: pd.DataFrame,
    segment: str,
    pct_point_reduction: float,
    segment_col: str = "Contract",
    duration_col: str = "tenure",
    churn_col: str = "Churn",
    monthly_charge_col: str = "MonthlyCharges",
    gross_margin: float = GROSS_MARGIN,
    tmax: int = MAX_TENURE_MONTHS,
) -> dict:
    """
    Estimate the CLV impact of reducing a segment's churn incidence by
    `pct_point_reduction` (e.g. 0.05 for 5 percentage points), applying a
    proportional-hazards scale to the segment's Kaplan-Meier curve so the
    cumulative churn incidence at `tmax` drops by that amount.
    """
    grp = df[df[segment_col] == segment]
    events = (grp[churn_col] == "Yes").astype(int)
    sf_old = fit_survival_curve(grp[duration_col], events, tmax)

    incidence_old = 1 - sf_old.iloc[-1]
    incidence_new = max(incidence_old - pct_point_reduction, 0.001)
    target_survival = 1 - incidence_new
    hazard_scale = np.log(target_survival) / np.log(sf_old.iloc[-1])
    sf_new = sf_old ** hazard_scale

    rmst_old = _trapz(sf_old.values, sf_old.index.values)
    rmst_new = _trapz(sf_new.values, sf_new.index.values)

    avg_charge = grp[monthly_charge_col].mean()
    clv_old = avg_charge * gross_margin * rmst_old
    clv_new = avg_charge * gross_margin * rmst_new
    n = len(grp)

    return {
        "segment": segment,
        "n_customers": n,
        "churn_incidence_old": incidence_old,
        "churn_incidence_new": incidence_new,
        "expected_lifetime_old": rmst_old,
        "expected_lifetime_new": rmst_new,
        "clv_per_customer_old": clv_old,
        "clv_per_customer_new": clv_new,
        "portfolio_clv_old": clv_old * n,
        "portfolio_clv_new": clv_new * n,
        "incremental_portfolio_clv": (clv_new - clv_old) * n,
    }


# ── summary stats ─────────────────────────────────────────────────────────────

def clv_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary table of CLV metrics by contract type."""
    return (
        df.groupby("Contract")
        .agg(
            n_customers=("clv", "count"),
            avg_clv=("clv", "mean"),
            median_clv=("clv", "median"),
            avg_monthly_charges=("MonthlyCharges", "mean"),
            avg_lifetime_months=("expected_lifetime_months", "mean"),
            avg_payback_months=("payback_months", "mean"),
            avg_ltv_cac=("ltv_cac_ratio", "mean"),
        )
        .round(2)
        .reset_index()
    )


def unit_economics_kpis(df: pd.DataFrame) -> dict:
    """High-level KPIs for the overview dashboard."""
    return {
        "avg_clv": df["clv"].mean(),
        "median_clv": df["clv"].median(),
        "avg_ltv_cac": df["ltv_cac_ratio"].mean(),
        "avg_payback_months": df["payback_months"].mean(),
        "pct_healthy_ltv_cac": (df["ltv_cac_ratio"] >= 3).mean(),
        "total_portfolio_clv": df["clv"].sum(),
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_clv_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="clv",
        color="clv_segment",
        nbins=60,
        title="Distribution of Customer Lifetime Value (CLV)",
        labels={"clv": "CLV ($)", "clv_segment": "Segment"},
        color_discrete_map={
            "High Value": "#2ecc71",
            "Medium Value": "#f39c12",
            "Low Value": "#e74c3c",
        },
        barmode="overlay",
        opacity=0.75,
    )
    fig.update_layout(height=400, xaxis_tickprefix="$", xaxis_tickformat=",")
    return fig


def plot_clv_by_contract(df: pd.DataFrame) -> go.Figure:
    summary = clv_summary(df)
    fig = px.bar(
        summary,
        x="Contract",
        y="avg_clv",
        color="Contract",
        text="avg_clv",
        title="Average CLV by Contract Type",
        labels={"avg_clv": "Average CLV ($)", "Contract": ""},
        color_discrete_sequence=["#e74c3c", "#f39c12", "#2ecc71"],
    )
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig.update_layout(height=400, yaxis_tickprefix="$", yaxis_tickformat=",", showlegend=False)
    return fig


def plot_ltv_cac_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter: monthly charges vs CLV, colored by contract, sized by expected lifetime."""
    sample = df.sample(min(1500, len(df)), random_state=42)
    fig = px.scatter(
        sample,
        x="MonthlyCharges",
        y="clv",
        color="Contract",
        size="expected_lifetime_months",
        size_max=14,
        opacity=0.6,
        title="CLV vs. Monthly Charges (bubble size = expected lifetime)",
        labels={
            "MonthlyCharges": "Monthly Charges ($)",
            "clv": "Customer Lifetime Value ($)",
            "expected_lifetime_months": "Expected Lifetime (months)",
        },
        hover_data=["expected_lifetime_months", "ltv_cac_ratio"],
    )
    fig.add_hline(
        y=3 * CAC,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"LTV = 3x CAC (${3*CAC:,.0f})",
        annotation_position="right",
    )
    fig.update_layout(height=500)
    return fig


def plot_payback_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="payback_months",
        color="Contract",
        nbins=40,
        title="CAC Payback Period Distribution (months)",
        labels={"payback_months": "Months to Recover CAC", "Contract": "Contract Type"},
        barmode="overlay",
        opacity=0.75,
    )
    fig.add_vline(
        x=12, line_dash="dash", line_color="red",
        annotation_text="12-month benchmark", annotation_position="top right",
    )
    fig.update_layout(height=400)
    return fig
