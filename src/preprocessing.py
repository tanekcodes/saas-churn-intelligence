"""
Data loading, cleaning, and feature engineering for the churn dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure the project root is importable when notebooks call this
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── constants ─────────────────────────────────────────────────────────────────

# Assume the company started signing up customers 72 months before a reference date
REFERENCE_DATE = pd.Timestamp("2024-01-01")

# All binary Yes/No columns
BINARY_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]

# Columns with a third "No <service>" value
SERVICE_COLS = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CATEGORICAL_COLS = ["gender", "InternetService", "Contract", "PaymentMethod"]


# ── loading ───────────────────────────────────────────────────────────────────

def load_raw(path: str | Path = None) -> pd.DataFrame:
    if path is None:
        path = ROOT / "data" / "raw" / "telco_churn.csv"
    df = pd.read_csv(path)
    # The real IBM dataset sometimes has TotalCharges as string with spaces
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


# ── cleaning ──────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows where TotalCharges couldn't be parsed (new customers, tenure=0)
    n_before = len(df)
    df = df.dropna(subset=["TotalCharges"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with missing TotalCharges")

    # Binary encoding: Yes → 1, No → 0
    for col in BINARY_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    # Service columns: Yes → 1, No → 0, "No <service>" → 0
    for col in SERVICE_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    # Rename for clarity
    df = df.rename(columns={"SeniorCitizen": "senior_citizen"})

    return df


# ── feature engineering ───────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Derive approximate signup date from tenure (months before reference date)
    df["signup_date"] = df["tenure"].apply(
        lambda t: REFERENCE_DATE - pd.DateOffset(months=int(t))
    )
    df["signup_month"] = df["signup_date"].dt.to_period("M")

    # Tenure buckets — useful for cohort grouping and EDA
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 36, 48, 60, 72],
        labels=["0-12 mo", "13-24 mo", "25-36 mo", "37-48 mo", "49-60 mo", "61-72 mo"],
    )

    # Number of add-on services purchased (proxy for engagement)
    addon_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_addons"] = df[addon_cols].sum(axis=1)

    # Revenue per month is already MonthlyCharges; add a log version for modeling
    df["log_monthly_charges"] = np.log1p(df["MonthlyCharges"])

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    return df


def preprocess(path: str | Path = None) -> pd.DataFrame:
    """Full pipeline: load → clean → feature engineer."""
    df = load_raw(path)
    df = clean(df)
    df = add_features(df)
    return df


def get_model_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for sklearn."""
    drop_cols = [
        "customerID", "Churn", "signup_date", "signup_month",
        "tenure_band", "TotalCharges",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    # Keep only numeric columns (get_dummies booleans become uint8 / bool)
    X = df[feature_cols].select_dtypes(include=[np.number, "bool"])
    X = X.astype(float)
    y = df["Churn"].astype(int)
    return X, y
