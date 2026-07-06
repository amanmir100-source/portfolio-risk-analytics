-- Daily simple returns per asset.
-- Technique: LAG() window function partitioned by ticker, so each asset's
-- series is differenced independently within a single scan.

SELECT
    date,
    ticker,
    close,
    close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS daily_return
FROM prices
ORDER BY ticker, date;
