-- Data-quality check: trading dates where some assets have no close.
-- Technique: CROSS JOIN every trading date with every asset in the
-- reference table to build the full expected grid, then LEFT JOIN the
-- prices fact table and keep the rows with no match (an anti-join).
-- This is how two data sets are reconciled: expected vs actual.

WITH trading_dates AS (
    SELECT DISTINCT date FROM prices
),
expected AS (
    SELECT d.date, a.ticker
    FROM trading_dates AS d
    CROSS JOIN assets AS a
),
missing AS (
    SELECT e.date, e.ticker
    FROM expected AS e
    LEFT JOIN prices AS p
        ON p.date = e.date AND p.ticker = e.ticker
    WHERE p.ticker IS NULL
    ORDER BY e.date, e.ticker
)
SELECT
    date,
    COUNT(*)                   AS missing_count,
    GROUP_CONCAT(ticker, ', ') AS missing_tickers
FROM missing
GROUP BY date
ORDER BY date;
