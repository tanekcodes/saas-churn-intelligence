"""
Customer Lifetime Value (CLV) and unit economics calculations.

CLV formula used here:
    CLV = (Monthly Revenue × Gross Margin) / Monthly Churn Rate

This is the "simple CLV" formula — interpretable and widely used in
SaaS finance. It answers: "If a customer pays $X/month, and customers
like them churn at rate r%, how much are they worth in total?"

Unit economics metrics:
  - CAC  : Customer Acquisition Cost (assumed constant for illustration)
  - LTV  : same as CLV
  - LTV:CAC ratio  — healthy SaaS = 3:1 or better
  - Payback period — months until CAC is recovered
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Assumed gross margin for a SaaS business (revenue - hosting/support costs)
GROSS_MARGIN = 0.75

# Assumed customer acquisition cost per customer (marketing + sales spend / new customers)
# In a real project this would come from finance data; here we use a reasonable assumption
CAC = 250.0


# ── CLV calculation ───────────────────────────────────────────────────────────

def calculate_clv(
    df: pd.DataFrame,
    churn_prob_col: str = "churn_probability",
    gross_margin: float = GROSS_MARGIN,
) -> pd.DataFrame:
    """
    Add CLV columns to the dataframe.

    churn_prob_col: name of the column with predicted churn probability (0–1)
    """
    df = df.copy()

    # Monthly churn rate for this customer = predicted churn probability
    # (probability of churning within the next month)
    monthly_churn_rate = df[churn_prob_col].clip(lower=0.001)  # avoid division by zero

    # Expected customer lifetime in months = 1 / churn_rate
    df["expected_lifetime_months"] = 1 / monthly_churn_rate

    # CLV = monthly revenue × gross margin × expected lifetime
    df["clv"] = df["MonthlyCharges"] * gross_margin * df["expected_lifetime_months"]

    return df


def add_unit_economics(df: pd.DataFrame, cac: float = CAC) -> pd.DataFrame:
    """Add CAC, LTV:CAC ratio, and payback period columns."""
    df = df.copy()
    df["cac"] = cac
    df["ltv_cac_ratio"] = df["clv"] / cac
    # Payback period: how many months until cumulative gross profit covers CAC
    # = CAC / (monthly_revenue × gross_margin)
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
    """Scatter: monthly charges vs CLV, colored by contract, sized by churn risk."""
    sample = df.sample(min(1500, len(df)), random_state=42)
    fig = px.scatter(
        sample,
        x="MonthlyCharges",
        y="clv",
        color="Contract",
        size="churn_probability",
        size_max=14,
        opacity=0.6,
        title="CLV vs. Monthly Charges (bubble size = churn risk)",
        labels={
            "MonthlyCharges": "Monthly Charges ($)",
            "clv": "Customer Lifetime Value ($)",
            "churn_probability": "Churn Probability",
        },
        hover_data=["expected_lifetime_months", "ltv_cac_ratio"],
    )
    # Healthy LTV:CAC reference line (LTV = 3 × CAC)
    fig.add_hline(
        y=3 * CAC,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"LTV = 3× CAC (${3*CAC:,.0f})",
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
