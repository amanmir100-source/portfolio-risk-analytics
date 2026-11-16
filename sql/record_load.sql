-- Append one audit row for the load that just happened (see load_log in schema.sql).
INSERT INTO load_log (loaded_at, source, row_count, first_date, last_date, content_sha256)
VALUES (?, ?, ?, ?, ?, ?);
