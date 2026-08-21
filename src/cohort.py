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

from src.clv import _get_segment_series


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


# ── Net Revenue Retention (NRR) ──────────────────────────────────────────────
#
# Customer-count retention (above) answers "what % of customers stayed."
# NRR answers a different, often more important question for a subscription
# business: "of the REVENUE we started with in a cohort, how much do we still
# have N months later" -- and it explicitly separates that answer into why:
# how much was lost to churn, how much was lost to downgrades (contraction),
# and how much was gained from upgrades (expansion) among customers who
# stayed. A cohort can have terrible customer retention but healthy NRR if
# the customers who stay expand enough to offset the ones who leave -- or
# the reverse: great customer retention but eroding NRR if everyone who
# stays quietly downgrades. Customer-count retention alone can't tell those
# two situations apart; NRR is what a real subscription business actually
# reports to investors for exactly this reason.
#
# Standard formula:
#   NRR(t) = (Starting MRR + Expansion MRR - Contraction MRR - Churned MRR) / Starting MRR
#
# This requires the monthly revenue panel (data/generate_monthly_panel.py),
# not the cross-sectional snapshot table -- see that module's docstring for
# why the snapshot table alone can't support this metric.

def compute_nrr_by_cohort_month(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute month-by-month NRR for the overall customer base using the
    monthly revenue panel. Returns one row per panel month with starting
    MRR, expansion/contraction/churn dollar amounts, and the resulting NRR.

    Note: `month` in the panel is customer-relative (month 1 = each
    customer's own first billed month), so this aggregates NRR by "months
    since each customer's own signup" rather than by calendar month --
    consistent with how the customer-count cohort table above is built, and
    a fair way to compare a customer base of mixed signup dates on a single
    timeline.
    """
    rows = []
    months = sorted(panel["month"].unique())

    for m in months:
        this_month = panel[panel["month"] == m]
        starting_mrr = this_month["mrr"].sum() if m == months[0] else None

        expansion_mrr = 0.0
        contraction_mrr = 0.0
        churned_mrr = 0.0

        if m > months[0]:
            prev_month = panel[panel["month"] == m - 1]
            prev_by_cust = prev_month.set_index("customerID")["mrr"]
            starting_mrr = prev_by_cust.sum()

            cur_by_cust = this_month.set_index("customerID")["mrr"]

            # customers present last month but not this month = churned this month
            churned_ids = prev_by_cust.index.difference(cur_by_cust.index)
            churned_mrr = prev_by_cust.loc[churned_ids].sum()

            # customers present both months: compare mrr to classify expansion/contraction
            common_ids = prev_by_cust.index.intersection(cur_by_cust.index)
            delta = cur_by_cust.loc[common_ids] - prev_by_cust.loc[common_ids]
            expansion_mrr = delta[delta > 0].sum()
            contraction_mrr = -delta[delta < 0].sum()  # store as a positive "amount lost"

        ending_mrr = this_month["mrr"].sum()
        nrr = (
            (starting_mrr + expansion_mrr - contraction_mrr - churned_mrr) / starting_mrr
            if starting_mrr and starting_mrr > 0
            else np.nan
        )

        rows.append({
            "month": m,
            "starting_mrr": starting_mrr,
            "expansion_mrr": expansion_mrr,
            "contraction_mrr": contraction_mrr,
            "churned_mrr": churned_mrr,
            "ending_mrr": ending_mrr,
            "nrr": nrr,
        })

    return pd.DataFrame(rows)


def nrr_by_segment(panel: pd.DataFrame, customers: pd.DataFrame, segment_col: str = "Contract") -> pd.DataFrame:
    """
    NRR computed separately per segment (e.g. Contract type), rather than
    for the whole base at once -- this is usually the more actionable view,
    since it directly answers "which segment's revenue base is actually
    healthy" rather than a single blended number that can hide an unhealthy
    segment behind a healthy one.

    Works whether `segment_col` is a plain categorical column on `customers`
    or has been one-hot encoded (e.g. by src/preprocessing.py's
    `preprocess()`, which is what the dashboard actually passes in) -- same
    fix as src/clv.py needed for the same underlying reason: this dataframe
    gets used in two different shapes across the project, and the segment
    lookup has to work with both rather than assuming the raw-data shape.
    """
    segments = _get_segment_series(customers, segment_col)
    seg_lookup = pd.DataFrame({"customerID": customers["customerID"], segment_col: segments})

    merged = panel.merge(seg_lookup, on="customerID", how="left")
    results = []
    for seg, seg_panel in merged.groupby(segment_col):
        nrr_df = compute_nrr_by_cohort_month(seg_panel)
        nrr_df[segment_col] = seg
        results.append(nrr_df)
    return pd.concat(results, ignore_index=True)


def overall_nrr_summary(nrr_df: pd.DataFrame, window: int = 12) -> dict:
    """
    Summarize NRR over a trailing window of months (default 12, the
    standard annualized NRR reporting convention in SaaS).
    """
    recent = nrr_df.dropna(subset=["nrr"]).tail(window)
    return {
        "avg_nrr": recent["nrr"].mean(),
        "median_nrr": recent["nrr"].median(),
        "total_expansion_mrr": recent["expansion_mrr"].sum(),
        "total_contraction_mrr": recent["contraction_mrr"].sum(),
        "total_churned_mrr": recent["churned_mrr"].sum(),
        "months_in_window": len(recent),
    }


def plot_nrr_waterfall(nrr_df: pd.DataFrame, month: int) -> go.Figure:
    """Waterfall chart decomposing one month's NRR into its components."""
    row = nrr_df[nrr_df["month"] == month].iloc[0]
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Starting MRR", "Expansion", "Contraction", "Churned", "Ending MRR"],
        y=[
            row["starting_mrr"],
            row["expansion_mrr"],
            -row["contraction_mrr"],
            -row["churned_mrr"],
            0,  # 'total' measure computes this automatically
        ],
        text=[f"${v:,.0f}" for v in [row["starting_mrr"], row["expansion_mrr"],
                                       -row["contraction_mrr"], -row["churned_mrr"],
                                       row["ending_mrr"]]],
        connector={"line": {"color": "rgb(150,150,150)"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        increasing={"marker": {"color": "#2ecc71"}},
        totals={"marker": {"color": "#3498db"}},
    ))
    fig.update_layout(
        title=f"MRR Bridge — Month {month} (NRR: {row['nrr']:.1%})",
        height=450,
    )
    return fig


def plot_nrr_trend(nrr_df: pd.DataFrame) -> go.Figure:
    """Line chart of NRR over time, with the 100% breakeven line marked --
    NRR above 100% means expansion is outpacing churn+contraction even with
    zero new customers; below 100% means the existing base is shrinking in
    revenue terms regardless of new sales."""
    valid = nrr_df.dropna(subset=["nrr"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid["month"], y=valid["nrr"],
        mode="lines+markers", name="NRR",
        line=dict(color="#3498db", width=2),
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                   annotation_text="100% (breakeven)", annotation_position="right")
    fig.update_layout(
        title="Net Revenue Retention Over Time",
        xaxis_title="Month (since each customer's own signup)",
        yaxis_title="NRR",
        yaxis_tickformat=".0%",
        height=400,
    )
    return fig
