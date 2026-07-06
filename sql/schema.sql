-- Schema for the portfolio risk analytics database.
-- Normalised design: reference data (assets, weights) separated from the
-- prices fact table, with FK + CHECK constraints enforcing integrity.

CREATE TABLE IF NOT EXISTS assets (
    ticker      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    asset_class TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio_weights (
    ticker TEXT PRIMARY KEY REFERENCES assets (ticker),
    weight REAL NOT NULL CHECK (weight >= 0 AND weight <= 1)
);

CREATE TABLE IF NOT EXISTS prices (
    date   TEXT NOT NULL,                          -- ISO-8601 (YYYY-MM-DD)
    ticker TEXT NOT NULL REFERENCES assets (ticker),
    close  REAL NOT NULL CHECK (close > 0),
    volume INTEGER CHECK (volume >= 0),
    PRIMARY KEY (date, ticker)
);

-- Covering index for the per-ticker time-series scans used by every
-- window-function query.
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices (ticker, date);
