-- Five deepest drawdown points per asset (depth from running all-time peak).
-- Techniques: running MAX() with an explicit UNBOUNDED PRECEDING frame to
-- track the high-water mark, then RANK() over the derived drawdown column.

WITH running_peak AS (
    SELECT
        date,
        ticker,
        close,
        MAX(close) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS peak
    FROM prices
),
drawdowns AS (
    SELECT
        date,
        ticker,
        close,
        peak,
        close / peak - 1 AS drawdown,
        RANK() OVER (PARTITION BY ticker ORDER BY close / peak - 1) AS dd_rank
    FROM running_peak
)
SELECT
    ticker,
    date,
    ROUND(peak, 2)                AS peak_close,
    ROUND(close, 2)               AS trough_close,
    ROUND(drawdown * 100, 2)      AS drawdown_pct,
    dd_rank
FROM drawdowns
WHERE dd_rank <= 5
ORDER BY ticker, dd_rank, date;
