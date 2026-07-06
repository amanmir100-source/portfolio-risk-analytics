-- Month-end close and month-over-month return per asset.
-- Techniques: strftime() bucketing, ROW_NUMBER() to isolate each month's
-- final trading day, then LAG() over the monthly series.

WITH month_rows AS (
    SELECT
        strftime('%Y-%m', date) AS month,
        ticker,
        close,
        ROW_NUMBER() OVER (
            PARTITION BY ticker, strftime('%Y-%m', date)
            ORDER BY date DESC
        ) AS rn
    FROM prices
),
month_end AS (
    SELECT month, ticker, close AS month_end_close
    FROM month_rows
    WHERE rn = 1
)
SELECT
    month,
    ticker,
    month_end_close,
    month_end_close / LAG(month_end_close) OVER (PARTITION BY ticker ORDER BY month) - 1
        AS monthly_return
FROM month_end
ORDER BY ticker, month;
