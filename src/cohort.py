"""
Cohort retention analysis.

A "cohort" is the group of customers who signed up in the same calendar month.
We track what percentage of each cohort is still active N months later.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def build_cohort_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a retention matrix:
        rows    = cohort (signup month, e.g. "2018-01")
        columns = months since signup (0, 1, 2, ...)
        values  = retention rate (0.0 – 1.0)

    We infer "still active" from the fact that the customer has NOT churned
    and their tenure covers the given month offset.
    """
    df = df.copy()
    df["signup_month"] = pd.to_datetime(df["signup_date"]).dt.to_period("M")

    cohort_data = []

    cohort_groups = df.groupby("signup_month")
    for cohort_period, group in cohort_groups:
        cohort_size = len(group)
        # For each month offset, count customers who were still active
        max_offset = int(group["tenure"].max())
        for offset in range(0, min(max_offset + 1, 73)):
            # Customer is "active at offset" if their tenure >= offset
            # (they hadn't churned yet at that point in their lifecycle)
            active = (group["tenure"] >= offset).sum()
            cohort_data.append(
                {
                    "cohort": str(cohort_period),
                    "month_offset": offset,
                    "cohort_size": cohort_size,
                    "active_customers": active,
                    "retention_rate": active / cohort_size,
                }
            )

    result = pd.DataFrame(cohort_data)
    return result


def pivot_retention(cohort_df: pd.DataFrame, max_offset: int = 24) -> pd.DataFrame:
    """Pivot cohort table into a heatmap-ready matrix."""
    filtered = cohort_df[cohort_df["month_offset"] <= max_offset].copy()
    matrix = filtered.pivot(
        index="cohort", columns="month_offset", values="retention_rate"
    )
    matrix.index.name = "Cohort (Signup Month)"
    matrix.columns.name = "Months Since Signup"
    # Only keep cohorts with at least 12 months of history to avoid noise
    matrix = matrix.dropna(thresh=12)
    return matrix


def plot_cohort_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Interactive Plotly heatmap of cohort retention rates."""
    z = matrix.values
    x = [str(c) for c in matrix.columns]
    y = [str(i) for i in matrix.index]

    text = [[f"{v:.0%}" if not np.isnan(v) else "" for v in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            text=text,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Retention Rate", tickformat=".0%"),
            hoverongaps=False,
            hovertemplate=(
                "Cohort: %{y}<br>"
                "Month %{x}<br>"
                "Retention: %{text}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Cohort Retention Heatmap — % of Original Cohort Still Active",
        xaxis_title="Months Since Signup",
        yaxis_title="Signup Cohort",
        height=600,
        font=dict(size=11),
    )
    return fig


def plot_retention_curves(matrix: pd.DataFrame, n_cohorts: int = 8) -> go.Figure:
    """Line chart of retention curves for the most recent N cohorts."""
    recent = matrix.tail(n_cohorts)
    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for i, (cohort, row) in enumerate(recent.iterrows()):
        valid = row.dropna()
        fig.add_trace(
            go.Scatter(
                x=valid.index,
                y=valid.values,
                mode="lines+markers",
                name=str(cohort),
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=5),
                hovertemplate=f"Cohort {cohort}<br>Month %{{x}}: %{{y:.1%}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Retention Curves by Signup Cohort",
        xaxis_title="Months Since Signup",
        yaxis_title="Retention Rate",
        yaxis_tickformat=".0%",
        height=450,
        legend_title="Signup Cohort",
    )
    return fig
