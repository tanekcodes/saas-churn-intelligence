-- =============================================================================
-- 02_churn_metrics.sql
-- Business-level churn analysis by key dimensions.
-- These are the queries a business analyst would run to understand *who* churns.
-- =============================================================================

-- ── 1. Overall churn rate ─────────────────────────────────────────────────────
SELECT
    COUNT(*)                        AS total_customers,
    SUM(churned)                    AS churned_count,
    ROUND(AVG(churned) * 100, 2)    AS churn_rate_pct
FROM customers;


-- ── 2. Churn rate by contract type ───────────────────────────────────────────
-- Key insight: month-to-month customers churn at much higher rates.
-- This is the most important business lever for reducing churn.
SELECT
    contract,
    COUNT(*)                        AS customers,
    SUM(churned)                    AS churned,
    ROUND(AVG(churned) * 100, 2)    AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)  AS avg_monthly_rev
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;


-- ── 3. Churn rate by internet service type ───────────────────────────────────
SELECT
    internet_service,
    COUNT(*)                        AS customers,
    ROUND(AVG(churned) * 100, 2)    AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)  AS avg_monthly_charges
FROM customers
GROUP BY internet_service
ORDER BY churn_rate_pct DESC;


-- ── 4. Churn rate by tenure band ─────────────────────────────────────────────
-- Customers churn most in their first year. After 2 years, they're sticky.
SELECT
    CASE
        WHEN tenure BETWEEN 0  AND 12 THEN '00-12 months'
        WHEN tenure BETWEEN 13 AND 24 THEN '13-24 months'
        WHEN tenure BETWEEN 25 AND 36 THEN '25-36 months'
        WHEN tenure BETWEEN 37 AND 48 THEN '37-48 months'
        WHEN tenure BETWEEN 49 AND 60 THEN '49-60 months'
        ELSE '61+ months'
    END AS tenure_band,
    COUNT(*)                        AS customers,
    ROUND(AVG(churned) * 100, 2)    AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)  AS avg_monthly_charges
FROM customers
GROUP BY 1
ORDER BY 1;


-- ── 5. Churn rate by payment method ──────────────────────────────────────────
-- Electronic check customers churn at disproportionately high rates.
SELECT
    payment_method,
    COUNT(*)                        AS customers,
    ROUND(AVG(churned) * 100, 2)    AS churn_rate_pct
FROM customers
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;


-- ── 6. Impact of support services on churn ───────────────────────────────────
-- Customers WITH tech support and online security churn significantly less.
-- This quantifies the business value of upselling these services.
SELECT
    tech_support,
    online_security,
    COUNT(*)                        AS customers,
    ROUND(AVG(churned) * 100, 2)    AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)  AS avg_rev
FROM customers
GROUP BY tech_support, online_security
ORDER BY churn_rate_pct DESC;


-- ── 7. Revenue at risk (monthly churned MRR) ──────────────────────────────────
-- How much MRR do we lose to churned customers?
SELECT
    contract,
    SUM(CASE WHEN churned = 1 THEN monthly_charges ELSE 0 END)  AS churned_mrr,
    SUM(monthly_charges)                                         AS total_mrr,
    ROUND(
        SUM(CASE WHEN churned = 1 THEN monthly_charges ELSE 0 END)
        / SUM(monthly_charges) * 100, 2
    )                                                            AS pct_mrr_at_risk
FROM customers
GROUP BY contract
ORDER BY churned_mrr DESC;


-- ── 8. High-value churned customers ──────────────────────────────────────────
-- Who are the top churned customers by revenue lost?
SELECT
    customerID,
    contract,
    tenure,
    monthly_charges,
    internet_service,
    tech_support,
    online_security
FROM customers
WHERE churned = 1
ORDER BY monthly_charges DESC
LIMIT 20;
