-- =============================================================================
-- 05_unit_economics.sql
-- Unit economics: CLV, CAC payback, and LTV:CAC ratios.
--
-- Assumptions (clearly stated — always do this in a real project):
--   CAC  = $250 per acquired customer
--   Gross margin = 75%
--   Monthly churn rate used per customer = overall rate by contract type
-- =============================================================================

-- ── 1. Churn rates by contract (used as input to CLV) ─────────────────────────
WITH contract_churn AS (
    SELECT
        contract,
        AVG(churned)                AS monthly_churn_rate,
        AVG(monthly_charges)        AS avg_monthly_charges
    FROM customers
    GROUP BY contract
)
SELECT
    contract,
    ROUND(monthly_churn_rate * 100, 2)          AS churn_rate_pct,
    -- Expected customer lifetime = 1 / churn rate
    ROUND(1.0 / monthly_churn_rate, 1)          AS expected_lifetime_months,
    ROUND(avg_monthly_charges, 2)               AS avg_monthly_charges,
    -- CLV = monthly_rev × gross_margin × expected_lifetime
    ROUND(avg_monthly_charges * 0.75
          / monthly_churn_rate, 0)              AS avg_clv,
    -- CAC payback = CAC / (monthly_rev × gross_margin)
    ROUND(250.0 / (avg_monthly_charges * 0.75), 1) AS payback_months,
    -- LTV:CAC
    ROUND((avg_monthly_charges * 0.75 / monthly_churn_rate) / 250.0, 1) AS ltv_cac_ratio
FROM contract_churn
ORDER BY avg_clv DESC;


-- ── 2. CLV per customer (using contract-level churn rate) ─────────────────────
WITH contract_churn AS (
    SELECT contract, AVG(churned) AS contract_churn_rate
    FROM customers
    GROUP BY contract
)
SELECT
    c.customerID,
    c.contract,
    c.monthly_charges,
    c.tenure,
    c.churned,
    cc.contract_churn_rate,
    -- CLV
    ROUND(c.monthly_charges * 0.75 / cc.contract_churn_rate, 0)    AS clv,
    -- Payback period
    ROUND(250.0 / (c.monthly_charges * 0.75), 1)                   AS payback_months,
    -- LTV:CAC
    ROUND((c.monthly_charges * 0.75 / cc.contract_churn_rate) / 250.0, 2) AS ltv_cac_ratio
FROM customers c
JOIN contract_churn cc ON c.contract = cc.contract
ORDER BY clv DESC;


-- ── 3. What % of customers have a healthy LTV:CAC (>= 3x)? ───────────────────
WITH contract_churn AS (
    SELECT contract, AVG(churned) AS contract_churn_rate
    FROM customers
    GROUP BY contract
),
clv_table AS (
    SELECT
        c.customerID,
        c.contract,
        (c.monthly_charges * 0.75 / cc.contract_churn_rate) / 250.0 AS ltv_cac_ratio
    FROM customers c
    JOIN contract_churn cc ON c.contract = cc.contract
)
SELECT
    ROUND(AVG(ltv_cac_ratio), 2)                                AS avg_ltv_cac,
    ROUND(AVG(CASE WHEN ltv_cac_ratio >= 3 THEN 1.0 ELSE 0 END) * 100, 1) AS pct_healthy,
    ROUND(AVG(CASE WHEN ltv_cac_ratio < 1 THEN 1.0 ELSE 0 END) * 100, 1) AS pct_underwater
FROM clv_table;


-- ── 4. Revenue impact of reducing churn by 10% ───────────────────────────────
-- "If we reduced month-to-month churn by 10 percentage points, what is the
--  incremental CLV gain across those customers?"
WITH current AS (
    SELECT
        AVG(churned)                AS current_churn_rate,
        AVG(monthly_charges)        AS avg_monthly_charges,
        COUNT(*)                    AS n_customers
    FROM customers
    WHERE contract = 'Month-to-month'
),
scenario AS (
    SELECT
        current_churn_rate,
        GREATEST(current_churn_rate - 0.10, 0.01) AS reduced_churn_rate,
        avg_monthly_charges,
        n_customers
    FROM current
)
SELECT
    ROUND(current_churn_rate * 100, 1)          AS current_churn_pct,
    ROUND(reduced_churn_rate * 100, 1)          AS reduced_churn_pct,
    -- CLV per customer before and after
    ROUND(avg_monthly_charges * 0.75 / current_churn_rate, 0)  AS clv_before,
    ROUND(avg_monthly_charges * 0.75 / reduced_churn_rate, 0)  AS clv_after,
    -- Total portfolio impact
    ROUND(n_customers * avg_monthly_charges * 0.75 *
          (1/reduced_churn_rate - 1/current_churn_rate), 0)    AS total_incremental_clv
FROM scenario;
