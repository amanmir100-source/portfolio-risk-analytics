-- Per-asset performance summary joined with reference data and weights.
-- Techniques: FIRST_VALUE / LAST_VALUE with explicit frames, LAG for returns,
-- aggregate stats over a CTE, multi-table JOIN, COALESCE for unweighted assets.

WITH ordered AS (
    SELECT
        date,
        ticker,
        close,
        volume,
        close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS r,
        FIRST_VALUE(close) OVER (
            PARTITION BY ticker ORDER BY date
        ) AS first_close,
        LAST_VALUE(close) OVER (
            PARTITION BY ticker ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS last_close
    FROM prices
),
stats AS (
    SELECT
        ticker,
        COUNT(r)                            AS n_days,
        MIN(first_close)                    AS first_close,
        MIN(last_close)                     AS last_close,
        AVG(r)                              AS avg_daily_return,
        -- population stddev is fine at n ~ 1250; SQRT/MAX guard rounding noise
        SQRT(MAX(AVG(r * r) - AVG(r) * AVG(r), 0.0)) * SQRT(252.0) AS ann_volatility,
        MIN(r)                              AS worst_day,
        MAX(r)                              AS best_day,
        CAST(AVG(volume) AS INTEGER)        AS avg_volume
    FROM ordered
    WHERE r IS NOT NULL
    GROUP BY ticker
)
SELECT
    a.ticker,
    a.name,
    a.asset_class,
    COALESCE(w.weight, 0.0)                              AS portfolio_weight,
    s.n_days,
    ROUND((s.last_close / s.first_close - 1) * 100, 2)   AS total_return_pct,
    ROUND(s.ann_volatility * 100, 2)                     AS ann_volatility_pct,
    ROUND(s.worst_day * 100, 2)                          AS worst_day_pct,
    ROUND(s.best_day * 100, 2)                           AS best_day_pct,
    s.avg_volume
FROM stats s
JOIN assets a            ON a.ticker = s.ticker
LEFT JOIN portfolio_weights w ON w.ticker = s.ticker
ORDER BY portfolio_weight DESC, a.ticker;
