-- =============================================================================
-- 04_cohort_retention.sql
-- Cohort retention analysis in pure SQL.
-- Cohorts are defined by the month a customer signed up.
-- =============================================================================

-- ── 1. Cohort sizes ───────────────────────────────────────────────────────────
SELECT
    signup_month,
    COUNT(*)                        AS cohort_size,
    ROUND(AVG(monthly_charges), 2)  AS avg_arpu,
    ROUND(AVG(churned) * 100, 2)    AS eventual_churn_rate_pct
FROM customers
GROUP BY signup_month
ORDER BY signup_month;


-- ── 2. Cohort retention matrix ────────────────────────────────────────────────
-- For each cohort, what % of customers survived to month N?
-- "Survived to month N" = tenure >= N (they were still active at that point)
WITH cohort_base AS (
    SELECT
        signup_month,
        COUNT(*) AS cohort_size
    FROM customers
    GROUP BY signup_month
),
offsets AS (
    SELECT UNNEST(range(0, 25)) AS month_offset  -- 0 to 24
),
retention AS (
    SELECT
        c.signup_month,
        o.month_offset,
        COUNT(c.customerID) AS still_active
    FROM customers c
    CROSS JOIN offsets o
    WHERE c.tenure >= o.month_offset
    GROUP BY c.signup_month, o.month_offset
)
SELECT
    r.signup_month,
    r.month_offset,
    cb.cohort_size,
    r.still_active,
    ROUND(r.still_active::DOUBLE / cb.cohort_size * 100, 1) AS retention_pct
FROM retention r
JOIN cohort_base cb ON r.signup_month = cb.signup_month
ORDER BY r.signup_month, r.month_offset;


-- ── 3. Average retention by month offset (across all cohorts) ─────────────────
-- What is the "typical" retention curve for our product?
WITH offsets AS (
    SELECT UNNEST(range(0, 25)) AS month_offset
),
retention AS (
    SELECT
        o.month_offset,
        COUNT(c.customerID)                                     AS still_active,
        (SELECT COUNT(*) FROM customers)                        AS total_customers
    FROM customers c
    CROSS JOIN offsets o
    WHERE c.tenure >= o.month_offset
    GROUP BY o.month_offset
)
SELECT
    month_offset,
    still_active,
    ROUND(still_active::DOUBLE / total_customers * 100, 1) AS avg_retention_pct
FROM retention
ORDER BY month_offset;


-- ── 4. Cohort revenue retention ───────────────────────────────────────────────
-- Same as above but weighted by revenue — are high-value customers stickier?
WITH cohort_rev AS (
    SELECT
        signup_month,
        SUM(monthly_charges) AS cohort_mrr
    FROM customers
    GROUP BY signup_month
),
offsets AS (
    SELECT UNNEST(range(0, 25)) AS month_offset
),
rev_retention AS (
    SELECT
        c.signup_month,
        o.month_offset,
        SUM(c.monthly_charges) AS retained_mrr
    FROM customers c
    CROSS JOIN offsets o
    WHERE c.tenure >= o.month_offset
    GROUP BY c.signup_month, o.month_offset
)
SELECT
    rr.signup_month,
    rr.month_offset,
    cr.cohort_mrr,
    rr.retained_mrr,
    ROUND(rr.retained_mrr / cr.cohort_mrr * 100, 1) AS revenue_retention_pct
FROM rev_retention rr
JOIN cohort_rev cr ON rr.signup_month = cr.signup_month
ORDER BY rr.signup_month, rr.month_offset;
