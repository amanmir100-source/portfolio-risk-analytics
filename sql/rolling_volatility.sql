-- 21-day rolling volatility, annualised (sqrt(252) scaling).
-- Techniques: CTE pipeline + named WINDOW with an explicit ROWS frame.
-- SQLite has no STDDEV aggregate, so the sample standard deviation is
-- derived from first principles inside the window:
--   var = n/(n-1) * ( E[r^2] - E[r]^2 )   (Bessel-corrected)

WITH returns AS (
    SELECT
        date,
        ticker,
        close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS r
    FROM prices
),
rolling AS (
    SELECT
        date,
        ticker,
        AVG(r)     OVER w AS mean_r,
        AVG(r * r) OVER w AS mean_r2,
        COUNT(r)   OVER w AS n_obs
    FROM returns
    WHERE r IS NOT NULL
    WINDOW w AS (
        PARTITION BY ticker
        ORDER BY date
        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
    )
)
SELECT
    date,
    ticker,
    SQRT(MAX(mean_r2 - mean_r * mean_r, 0.0) * n_obs / (n_obs - 1)) * SQRT(252.0)
        AS ann_volatility_21d
FROM rolling
WHERE n_obs = 21          -- only emit full windows
ORDER BY ticker, date;
