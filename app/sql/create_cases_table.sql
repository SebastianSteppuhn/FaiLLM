CREATE TABLE IF NOT EXISTS public.{table_name} (
        id        TEXT PRIMARY KEY,
        pack      TEXT,
        version   INTEGER,
        label     TEXT,
        prompt    TEXT,
        tags      TEXT[],
        props     JSONB,
        source    JSONB,
        expected  JSONB,
        checks    JSONB
    );