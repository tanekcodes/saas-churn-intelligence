-- =============================================================================
-- 01_setup_database.sql
-- Load raw CSV into DuckDB and create a clean working table.
-- Run this first before any other SQL scripts.
-- =============================================================================

-- Drop tables if they exist (safe to re-run)
DROP TABLE IF EXISTS raw_customers;
DROP TABLE IF EXISTS customers;

-- Load raw CSV directly — DuckDB can read CSV files natively
CREATE TABLE raw_customers AS
    SELECT * FROM read_csv_auto('data/raw/telco_churn.csv', header=true);

-- Preview
SELECT * FROM raw_customers LIMIT 5;

-- =============================================================================
-- Create a cleaned, typed working table
-- =============================================================================

CREATE TABLE customers AS
SELECT
    customerID,
    gender,
    SeniorCitizen::INTEGER                          AS senior_citizen,
    CASE WHEN Partner        = 'Yes' THEN 1 ELSE 0 END AS partner,
    CASE WHEN Dependents     = 'Yes' THEN 1 ELSE 0 END AS dependents,
    tenure::INTEGER                                 AS tenure,
    CASE WHEN PhoneService   = 'Yes' THEN 1 ELSE 0 END AS phone_service,
    CASE WHEN MultipleLines  = 'Yes' THEN 1 ELSE 0 END AS multiple_lines,
    InternetService                                 AS internet_service,
    CASE WHEN OnlineSecurity    = 'Yes' THEN 1 ELSE 0 END AS online_security,
    CASE WHEN OnlineBackup      = 'Yes' THEN 1 ELSE 0 END AS online_backup,
    CASE WHEN DeviceProtection  = 'Yes' THEN 1 ELSE 0 END AS device_protection,
    CASE WHEN TechSupport       = 'Yes' THEN 1 ELSE 0 END AS tech_support,
    CASE WHEN StreamingTV       = 'Yes' THEN 1 ELSE 0 END AS streaming_tv,
    CASE WHEN StreamingMovies   = 'Yes' THEN 1 ELSE 0 END AS streaming_movies,
    Contract                                        AS contract,
    CASE WHEN PaperlessBilling = 'Yes' THEN 1 ELSE 0 END AS paperless_billing,
    PaymentMethod                                   AS payment_method,
    MonthlyCharges::DOUBLE                          AS monthly_charges,
    TRY_CAST(TotalCharges AS DOUBLE)                AS total_charges,
    CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END       AS churned,

    -- Derived: approximate signup date from tenure
    (DATE '2024-01-01' - INTERVAL (tenure) MONTH)  AS signup_date,
    DATE_TRUNC('month',
        DATE '2024-01-01' - INTERVAL (tenure) MONTH
    )::DATE                                         AS signup_month

FROM raw_customers
WHERE TRY_CAST(TotalCharges AS DOUBLE) IS NOT NULL;  -- drop rows with missing TotalCharges

-- Confirm row count and churn rate
SELECT
    COUNT(*)                            AS total_customers,
    SUM(churned)                        AS churned_customers,
    ROUND(AVG(churned) * 100, 2)        AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)      AS avg_monthly_charges,
    ROUND(SUM(monthly_charges), 0)      AS total_mrr
FROM customers;
