CREATE TABLE IF NOT EXISTS {table} (
  id           BIGSERIAL PRIMARY KEY,
  case_id      TEXT,
  label        TEXT,
  prompt       TEXT,
  output       TEXT,
  ok           BOOLEAN,
  error        TEXT,
  latency_ms   INTEGER,
  status       INTEGER,
  endpoint     TEXT,
  created_at   TIMESTAMPTZ DEFAULT now()
);