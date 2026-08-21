"""
Monthly revenue panel generator.

The base dataset (data/generate_data.py) is a cross-sectional snapshot: one row
per customer, a single tenure value, a single MonthlyCharges value. That's
sufficient for churn classification and Kaplan-Meier survival analysis, but it
is NOT sufficient for Net Revenue Retention (NRR) -- NRR is fundamentally a
longitudinal metric (Starting MRR + Expansion - Contraction - Churn, measured
cohort-over-time), and you cannot compute it from a single snapshot.

This module generates a synthetic monthly revenue panel: for each customer,
a month-by-month MRR trajectory from their signup month to either their
observed churn month or the end of the observation window (censored, still
active). Each customer has a chance, each month, of an expansion event
(upsell/add-on), a contraction event (downgrade), or -- at their known tenure,
if Churn == 'Yes' -- a churn event that zeroes their MRR from that month on.

This keeps the same honesty discipline as the rest of the project: this is
simulated data designed to make a real metric computable and demonstrable,
not a claim that this reflects any actual company's revenue history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42

# Monthly probability of an expansion event (customer upgrades / adds a service)
EXPANSION_PROB = 0.04
# Monthly probability of a contraction event (customer downgrades)
CONTRACTION_PROB = 0.03
# Expansion / contraction size as a multiplier on current MRR
EXPANSION_MULT_RANGE = (1.10, 1.35)
CONTRACTION_MULT_RANGE = (0.65, 0.90)

# Observation window: how many months of panel data to generate. Customers
# censored (still active) beyond their observed tenure simply continue: their
# generation stops at `min(tenure, PANEL_MONTHS)`.
PANEL_MONTHS = 72


def generate_monthly_panel(customers: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """
    Generate a (customerID, month, mrr, event) panel from the cross-sectional
    customer table.

    `month` is 1-indexed, relative to each customer's own signup (month 1 =
    their first billed month), NOT a shared calendar month -- this mirrors how
    a real subscription billing system tracks customer-relative billing
    cycles, and keeps cohort analysis (grouping by tenure-at-event) simple.

    `event` is one of: 'active', 'expansion', 'contraction', 'churn'.
    A customer's final row has event='churn' if Churn == 'Yes' in the source
    table; otherwise their panel simply ends at their observed tenure with
    event='active' (censored -- they were still a customer when the data was
    pulled, we just don't have data past that point).
    """
    rng = np.random.default_rng(seed)
    rows = []

    for _, cust in customers.iterrows():
        cid = cust["customerID"]
        tenure = int(cust["tenure"])
        base_mrr = float(cust["MonthlyCharges"])
        will_churn = cust["Churn"] == "Yes"
        months_to_generate = min(tenure, PANEL_MONTHS)

        current_mrr = base_mrr
        for m in range(1, months_to_generate + 1):
            event = "active"
            is_last_month = m == months_to_generate

            if is_last_month and will_churn:
                event = "churn"
                rows.append((cid, m, current_mrr, event))
                # MRR for a churned customer is reported as 0 from the month
                # AFTER their churn event in a real billing panel; we don't
                # emit a zero-row here since the panel simply ends -- absence
                # of further rows *is* the signal, same as real billing data.
                continue

            # expansion/contraction rolls, mutually exclusive per month
            roll = rng.random()
            if roll < EXPANSION_PROB:
                mult = rng.uniform(*EXPANSION_MULT_RANGE)
                current_mrr = round(current_mrr * mult, 2)
                event = "expansion"
            elif roll < EXPANSION_PROB + CONTRACTION_PROB:
                mult = rng.uniform(*CONTRACTION_MULT_RANGE)
                current_mrr = round(current_mrr * mult, 2)
                event = "contraction"

            rows.append((cid, m, current_mrr, event))

    panel = pd.DataFrame(rows, columns=["customerID", "month", "mrr", "event"])
    return panel


def save_panel(panel: pd.DataFrame, path: str = "data/raw/monthly_revenue_panel.csv") -> None:
    panel.to_csv(path, index=False)
    print(f"Saved {len(panel):,} rows to {path}")


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing import load_raw

    customers = load_raw()
    panel = generate_monthly_panel(customers)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    save_panel(panel)
