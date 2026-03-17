"""
MRR (Monthly Recurring Revenue) analysis and decomposition.

MRR decomposition breaks revenue movement into four buckets:
  - New MRR       : revenue from brand-new customers
  - Churned MRR   : revenue lost from customers who cancelled
  - Expansion MRR : extra revenue from existing customers upgrading
  - Contraction MRR: revenue lost from existing customers downgrading

The net change in MRR = New + Expansion - Churned - Contraction
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def build_monthly_mrr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct a month-by-month MRR time series from the snapshot dataset.

    For each customer, they contribute MonthlyCharges to MRR in every month
    from their signup_date up to either:
      - their churn month (if Churn == 1), or
      - the reference date (if still active)
    """
    rows = []
    ref_date = pd.Timestamp("2024-01-01")

    for _, customer in df.iterrows():
        signup = pd.Timestamp(customer["signup_date"])
        tenure = int(customer["tenure"])
        churned = customer["Churn"] == 1

        # Months this customer was active
        months = pd.date_range(signup, periods=tenure, freq="MS")
        for month in months:
            if month > ref_date:
                break
            rows.append(
                {
                    "month": month.to_period("M"),
                    "customerID": customer["customerID"],
                    "mrr": customer["MonthlyCharges"],
                    "churned": churned and (month == months[-1]),
                    "contract": customer.get("Contract", "Unknown"),
                }
            )

    long_df = pd.DataFrame(rows)
    return long_df


def aggregate_mrr(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to monthly MRR totals."""
    monthly = (
        long_df.groupby("month")
        .agg(
            total_mrr=("mrr", "sum"),
            active_customers=("customerID", "nunique"),
            churned_customers=("churned", "sum"),
        )
        .reset_index()
    )
    monthly["avg_revenue_per_customer"] = (
        monthly["total_mrr"] / monthly["active_customers"]
    )
    monthly["churn_rate"] = (
        monthly["churned_customers"] / monthly["active_customers"]
    )
    monthly["month_str"] = monthly["month"].astype(str)
    return monthly.sort_values("month")


def build_mrr_decomposition(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate MRR decomposition month-over-month.
    Since we don't have true expansion/contraction data, we model:
      - New MRR   : estimated from new customer additions
      - Churn MRR : estimated from churned customers * avg rev
      - Net change: total_mrr delta
    """
    decomp = monthly.copy().sort_values("month").reset_index(drop=True)
    decomp["mrr_change"] = decomp["total_mrr"].diff()
    decomp["churned_mrr"] = (
        decomp["churned_customers"] * decomp["avg_revenue_per_customer"]
    )
    decomp["new_mrr"] = decomp["mrr_change"] + decomp["churned_mrr"]
    decomp["new_mrr"] = decomp["new_mrr"].clip(lower=0)
    decomp = decomp.dropna(subset=["mrr_change"])
    return decomp


def plot_mrr_trend(monthly: pd.DataFrame) -> go.Figure:
    """Line chart of total MRR over time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month_str"],
            y=monthly["total_mrr"],
            mode="lines",
            fill="tozeroy",
            name="Total MRR",
            line=dict(color="#4C78A8", width=2),
            hovertemplate="Month: %{x}<br>MRR: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Monthly Recurring Revenue (MRR) Trend",
        xaxis_title="Month",
        yaxis_title="MRR ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        height=400,
        xaxis=dict(tickangle=45, nticks=24),
    )
    return fig


def plot_mrr_decomposition(decomp: pd.DataFrame) -> go.Figure:
    """Stacked bar chart showing MRR waterfall decomposition."""
    # Keep last 24 months for readability
    decomp = decomp.tail(24)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="New MRR",
            x=decomp["month_str"],
            y=decomp["new_mrr"],
            marker_color="#2ecc71",
            hovertemplate="New MRR: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Churned MRR (lost)",
            x=decomp["month_str"],
            y=-decomp["churned_mrr"],
            marker_color="#e74c3c",
            hovertemplate="Churned MRR: -$%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Net MRR Change",
            x=decomp["month_str"],
            y=decomp["mrr_change"],
            mode="lines+markers",
            line=dict(color="#2c3e50", width=2, dash="dot"),
            marker=dict(size=6),
            hovertemplate="Net Change: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="MRR Decomposition: New vs. Churned Revenue",
        xaxis_title="Month",
        yaxis_title="MRR ($)",
        barmode="relative",
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        height=450,
        xaxis=dict(tickangle=45, nticks=24),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_churn_rate_trend(monthly: pd.DataFrame) -> go.Figure:
    """Monthly churn rate over time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month_str"],
            y=monthly["churn_rate"],
            mode="lines+markers",
            name="Monthly Churn Rate",
            line=dict(color="#e74c3c", width=2),
            marker=dict(size=5),
            hovertemplate="Month: %{x}<br>Churn Rate: %{y:.1%}<extra></extra>",
        )
    )
    # Average line
    avg = monthly["churn_rate"].mean()
    fig.add_hline(
        y=avg,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Avg: {avg:.1%}",
        annotation_position="right",
    )
    fig.update_layout(
        title="Monthly Churn Rate",
        xaxis_title="Month",
        yaxis_title="Churn Rate",
        yaxis_tickformat=".1%",
        height=380,
        xaxis=dict(tickangle=45, nticks=24),
    )
    return fig
