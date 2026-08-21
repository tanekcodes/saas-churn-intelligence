"""
Persistent database layer.

Every prior version of this project read data/raw/telco_churn.csv fresh, in
full, on every run -- fine for a one-off analysis, not representative of how
a real product actually works. A real subscription business has customers
signing up and billing events happening continuously; the analytics layer
should be able to ingest new data incrementally into a persistent store, not
re-read a static file from scratch every time.

This module builds that persistent layer on DuckDB (already used elsewhere in
this project for SQL analytics, so this keeps the stack consistent rather
than introducing a new database technology just for this).

Design:
  - `customers` table: one row per customer, upserted (a customer's status
    can change -- e.g. Churn flips from 'No' to 'Yes' -- so re-ingesting the
    same customerID updates the existing row rather than duplicating it)
  - `monthly_revenue` table: append-only. Billing events don't get rewritten
    after the fact in a real system, so new months are simply added.
  - `ingestion_log` table: records every ingestion run (what was loaded, how
    many rows, when) -- this is what the drift-detection module reads to
    know which "new" data hasn't been checked against the training baseline
    yet.

This simulates a live system by ingesting the monthly panel one month at a
time via `ingest_month()`, rather than loading all 72 months in a single
call -- which is what lets the drift-detection notebook demonstrate checking
each newly-arrived month against a fixed training baseline, the same way a
real monitoring job would run on a schedule.
"""

from __future__ import annotations

import duckdb
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = "data/vertex_churn.db"  # kept explicit rather than in-memory so state persists across runs


def get_connection(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(db_path)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customerID VARCHAR PRIMARY KEY,
            gender VARCHAR,
            SeniorCitizen INTEGER,
            Partner VARCHAR,
            Dependents VARCHAR,
            tenure INTEGER,
            PhoneService VARCHAR,
            MultipleLines VARCHAR,
            InternetService VARCHAR,
            OnlineSecurity VARCHAR,
            OnlineBackup VARCHAR,
            DeviceProtection VARCHAR,
            TechSupport VARCHAR,
            StreamingTV VARCHAR,
            StreamingMovies VARCHAR,
            Contract VARCHAR,
            PaperlessBilling VARCHAR,
            PaymentMethod VARCHAR,
            MonthlyCharges DOUBLE,
            TotalCharges DOUBLE,
            Churn VARCHAR,
            last_updated TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS monthly_revenue (
            customerID VARCHAR,
            month INTEGER,
            mrr DOUBLE,
            event VARCHAR,
            ingested_at TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_log (
            ingestion_id INTEGER,
            table_name VARCHAR,
            rows_ingested INTEGER,
            ingested_at TIMESTAMP,
            detail VARCHAR
        )
    """)


def _next_ingestion_id(con: duckdb.DuckDBPyConnection) -> int:
    result = con.execute("SELECT COALESCE(MAX(ingestion_id), 0) + 1 FROM ingestion_log").fetchone()
    return result[0]


def ingest_customers(con: duckdb.DuckDBPyConnection, customers: pd.DataFrame) -> int:
    """Upsert customer records. Re-running with the same customerIDs updates
    rather than duplicates -- this is what lets a customer's Churn flag or
    tenure change on re-ingestion without creating ghost duplicate rows,
    the same integrity requirement a real customer table would have."""
    df = customers.copy()
    df["last_updated"] = datetime.now(timezone.utc)

    con.register("staging_customers", df)
    con.execute("""
        INSERT INTO customers
        SELECT * FROM staging_customers
        ON CONFLICT (customerID) DO UPDATE SET
            tenure = EXCLUDED.tenure,
            Churn = EXCLUDED.Churn,
            MonthlyCharges = EXCLUDED.MonthlyCharges,
            TotalCharges = EXCLUDED.TotalCharges,
            last_updated = EXCLUDED.last_updated
    """)
    con.unregister("staging_customers")

    n = len(df)
    ingestion_id = _next_ingestion_id(con)
    con.execute(
        "INSERT INTO ingestion_log VALUES (?, 'customers', ?, ?, ?)",
        [ingestion_id, n, datetime.now(timezone.utc), f"upserted {n} customer records"]
    )
    return n


def ingest_month(con: duckdb.DuckDBPyConnection, panel: pd.DataFrame, month: int) -> int:
    """Ingest a single month's worth of billing events. Append-only, since
    billing history doesn't get rewritten after the fact. This is the
    function a scheduled monthly job would call in a real system, and it's
    the function the drift-detection notebook calls once per simulated
    month to demonstrate checking incoming data against the training
    baseline as it arrives."""
    month_slice = panel[panel["month"] == month].copy()
    month_slice["ingested_at"] = datetime.now(timezone.utc)

    con.register("staging_month", month_slice)
    con.execute("INSERT INTO monthly_revenue SELECT * FROM staging_month")
    con.unregister("staging_month")

    n = len(month_slice)
    ingestion_id = _next_ingestion_id(con)
    con.execute(
        "INSERT INTO ingestion_log VALUES (?, 'monthly_revenue', ?, ?, ?)",
        [ingestion_id, n, datetime.now(timezone.utc), f"month {month}: {n} billing events"]
    )
    return n


def query(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def ingestion_history(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return query(con, "SELECT * FROM ingestion_log ORDER BY ingestion_id")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing import load_raw

    con = get_connection()
    init_schema(con)

    customers = load_raw()
    n_cust = ingest_customers(con, customers)
    print(f"Ingested {n_cust} customers")

    panel = pd.read_csv("data/raw/monthly_revenue_panel.csv")
    max_month = panel["month"].max()
    for m in range(1, max_month + 1):
        n = ingest_month(con, panel, m)
    print(f"Ingested {max_month} months of billing events ({len(panel):,} total rows)")

    print()
    print(ingestion_history(con).tail(10))
