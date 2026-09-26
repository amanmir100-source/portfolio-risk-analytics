-- Five deepest drawdown EPISODES per asset: peak, trough and recovery.
-- Ranking single days would list one crash five times (its five worst
-- days), so days are first grouped into episodes - a gaps-and-islands
-- problem. Every new high starts a new episode:
--   1. running MAX() gives the high-water mark,
--   2. a running SUM() of "new high" flags numbers the episodes,
--   3. each episode is summarised (trough via ROW_NUMBER), and its recovery
--      is the first day of the next episode, i.e. the day the old peak is
--      regained (NULL if it hasn't been yet),
--   4. RANK() orders the episodes by depth.

WITH ordered AS (
    SELECT
        date,
        ticker,
        close,
        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date) AS day_n,
        MAX(close) OVER (
            PARTITION BY ticker ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS peak
    FROM prices
),
episodes AS (
    SELECT
        *,
        SUM(CASE WHEN close >= peak THEN 1 ELSE 0 END) OVER (
            PARTITION BY ticker ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS episode_id
    FROM ordered
),
lows AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY ticker, episode_id ORDER BY close, date
        ) AS low_rank
    FROM episodes
),
summary AS (
    SELECT
        ticker,
        episode_id,
        MIN(day_n)                                 AS peak_day,
        MAX(day_n)                                 AS last_day,
        MAX(peak)                                  AS peak_close,
        MIN(CASE WHEN low_rank = 1 THEN day_n END) AS trough_day,
        MIN(close)                                 AS trough_close
    FROM lows
    GROUP BY ticker, episode_id
    HAVING MIN(close) < MAX(peak)          -- ignore days with no decline
),
ranked AS (
    SELECT
        s.*,
        s.trough_close / s.peak_close - 1 AS drawdown,
        RANK() OVER (PARTITION BY s.ticker
                     ORDER BY s.trough_close / s.peak_close) AS dd_rank
    FROM summary AS s
)
SELECT
    r.ticker,
    pk.date                                AS peak_date,
    ROUND(r.peak_close, 2)                 AS peak_close,
    tr.date                                AS trough_date,
    ROUND(r.trough_close, 2)               AS trough_close,
    ROUND(r.drawdown * 100, 2)             AS drawdown_pct,
    rc.date                                AS recovery_date,
    r.trough_day - r.peak_day              AS days_to_trough,
    rc.day_n - r.peak_day                  AS days_to_recover,
    r.dd_rank
FROM ranked AS r
JOIN ordered AS pk ON pk.ticker = r.ticker AND pk.day_n = r.peak_day
JOIN ordered AS tr ON tr.ticker = r.ticker AND tr.day_n = r.trough_day
LEFT JOIN ordered AS rc ON rc.ticker = r.ticker AND rc.day_n = r.last_day + 1
WHERE r.dd_rank <= 5
ORDER BY r.ticker, r.dd_rank, pk.date;
