CREATE TABLE IF NOT EXISTS cases (
  id TEXT PRIMARY KEY,
  pack TEXT NOT NULL,
  version INT NOT NULL,
  label TEXT,
  prompt TEXT NOT NULL,
  expected JSONB,
  checks JSONB,
  tags TEXT[] DEFAULT '{}',
  props JSONB DEFAULT '{}'::jsonb,
  source JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS cases_pack_idx ON cases(pack);
CREATE INDEX IF NOT EXISTS cases_tags_gin ON cases USING GIN(tags);
CREATE INDEX IF NOT EXISTS cases_props_gin ON cases USING GIN(props);
