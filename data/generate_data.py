"""
Synthetic SaaS subscription dataset generator.
Produces ~7,000 rows that mirror the structure and statistical properties
of the IBM Telco Customer Churn dataset.

Run: python data/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 7_043


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── demographics ──────────────────────────────────────────────────────────
    gender = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])

    # ── contract & tenure ─────────────────────────────────────────────────────
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n,
        p=[0.55, 0.21, 0.24],
    )
    # tenure shaped by contract: month-to-month skewed short, two-year skewed long
    tenure = np.where(
        contract == "Month-to-month",
        rng.integers(1, 36, n),
        np.where(contract == "One year", rng.integers(12, 60, n), rng.integers(24, 72, n)),
    ).clip(1, 72)

    # ── services ──────────────────────────────────────────────────────────────
    phone = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multi_lines = np.where(
        phone == "No",
        "No phone service",
        rng.choice(["Yes", "No"], n, p=[0.42, 0.58]),
    )
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    def internet_addon(yes_p: float):
        return np.where(
            internet == "No",
            "No internet service",
            rng.choice(["Yes", "No"], n, p=[yes_p, 1 - yes_p]),
        )

    online_security = internet_addon(0.29)
    online_backup = internet_addon(0.34)
    device_protection = internet_addon(0.34)
    tech_support = internet_addon(0.29)
    streaming_tv = internet_addon(0.38)
    streaming_movies = internet_addon(0.39)

    # ── billing ───────────────────────────────────────────────────────────────
    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # monthly charges: base by internet type + addons
    base_charge = np.where(
        internet == "No", 20.0, np.where(internet == "DSL", 45.0, 70.0)
    )
    addon_charge = (
        (multi_lines == "Yes").astype(float) * 10
        + (online_security == "Yes").astype(float) * 8
        + (online_backup == "Yes").astype(float) * 8
        + (device_protection == "Yes").astype(float) * 8
        + (tech_support == "Yes").astype(float) * 8
        + (streaming_tv == "Yes").astype(float) * 8
        + (streaming_movies == "Yes").astype(float) * 8
    )
    monthly_charges = (base_charge + addon_charge + rng.normal(0, 2, n)).clip(18, 118)
    monthly_charges = monthly_charges.round(2)
    total_charges = (monthly_charges * tenure + rng.normal(0, 5, n)).clip(18).round(2)

    # ── churn label (logistic probability model) ───────────────────────────────
    log_odds = (
        -1.0  # baseline
        + 1.8 * (contract == "Month-to-month").astype(float)
        + 0.5 * (contract == "One year").astype(float)
        + 0.4 * (internet == "Fiber optic").astype(float)
        + 0.3 * senior.astype(float)
        - 0.5 * (online_security == "Yes").astype(float)
        - 0.4 * (tech_support == "Yes").astype(float)
        - 0.03 * tenure
        + 0.008 * monthly_charges
        + 0.4 * (payment == "Electronic check").astype(float)
        - 0.3 * (partner == "Yes").astype(float)
        + rng.normal(0, 0.3, n)
    )
    churn_prob = 1 / (1 + np.exp(-log_odds))
    churn = np.where(rng.random(n) < churn_prob, "Yes", "No")

    # ── customer IDs ──────────────────────────────────────────────────────────
    customer_ids = [f"CUST-{i:05d}" for i in range(1, n + 1)]

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multi_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )
    return df


if __name__ == "__main__":
    out = Path(__file__).parent / "raw" / "telco_churn.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(out, index=False)
    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"Generated {len(df):,} rows -> {out}")
    print(f"Churn rate: {churn_rate:.1%}")
    print(df.head(3).to_string())
