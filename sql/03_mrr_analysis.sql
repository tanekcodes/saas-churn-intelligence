-- =============================================================================
-- 03_mrr_analysis.sql
-- Monthly Recurring Revenue (MRR) metrics.
-- =============================================================================

-- ── 1. Current MRR snapshot ───────────────────────────────────────────────────
-- Total MRR from all currently active (non-churned) customers
SELECT
    ROUND(SUM(monthly_charges), 0)      AS current_mrr,
    COUNT(*)                            AS active_customers,
    ROUND(AVG(monthly_charges), 2)      AS arpu,         -- Average Revenue Per User
    ROUND(MEDIAN(monthly_charges), 2)   AS median_arpu
FROM customers
WHERE churned = 0;


-- ── 2. MRR by contract type ───────────────────────────────────────────────────
SELECT
    contract,
    COUNT(*)                            AS active_customers,
    ROUND(SUM(monthly_charges), 0)      AS mrr,
    ROUND(AVG(monthly_charges), 2)      AS arpu,
    ROUND(
        SUM(monthly_charges) /
        (SELECT SUM(monthly_charges) FROM customers WHERE churned = 0) * 100,
        1
    )                                   AS pct_of_total_mrr
FROM customers
WHERE churned = 0
GROUP BY contract
ORDER BY mrr DESC;


-- ── 3. MRR by internet service ────────────────────────────────────────────────
SELECT
    internet_service,
    COUNT(*)                            AS active_customers,
    ROUND(SUM(monthly_charges), 0)      AS mrr,
    ROUND(AVG(monthly_charges), 2)      AS arpu
FROM customers
WHERE churned = 0
GROUP BY internet_service
ORDER BY mrr DESC;


-- ── 4. Approximate monthly MRR time series ────────────────────────────────────
-- Reconstruct MRR for each historical month by looking at which customers
-- were active in that month (tenure covers it)
WITH months AS (
    SELECT UNNEST(range(1, 73)) AS month_offset
),
customer_months AS (
    SELECT
        c.customerID,
        c.monthly_charges,
        c.contract,
        c.signup_month,
        m.month_offset,
        -- The calendar month this represents
        (c.signup_month + INTERVAL (m.month_offset - 1) MONTH)::DATE AS calendar_month
    FROM customers c
    CROSS JOIN months m
    WHERE m.month_offset <= c.tenure
      AND (c.signup_month + INTERVAL (m.month_offset - 1) MONTH)
            <= DATE '2024-01-01'
)
SELECT
    DATE_TRUNC('month', calendar_month)::DATE   AS month,
    COUNT(DISTINCT customerID)                  AS active_customers,
    ROUND(SUM(monthly_charges), 0)              AS mrr,
    ROUND(AVG(monthly_charges), 2)              AS arpu
FROM customer_months
GROUP BY 1
ORDER BY 1;


-- ── 5. Revenue concentration risk ────────────────────────────────────────────
-- What share of revenue comes from top 10% of customers?
WITH ranked AS (
    SELECT
        monthly_charges,
        NTILE(10) OVER (ORDER BY monthly_charges DESC) AS decile
    FROM customers
    WHERE churned = 0
)
SELECT
    decile,
    COUNT(*)                                    AS customers,
    ROUND(SUM(monthly_charges), 0)              AS mrr,
    ROUND(
        SUM(monthly_charges) /
        (SELECT SUM(monthly_charges) FROM customers WHERE churned = 0) * 100,
        1
    )                                           AS pct_of_total_mrr
FROM ranked
GROUP BY decile
ORDER BY decile;
