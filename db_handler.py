# db_handler.py — replace add_cases() and helpers with the below

import os, re, pathlib
from typing import Dict, Any, List, Iterable

from dotenv import load_dotenv
import psycopg
from psycopg import sql
from psycopg.types.json import Json  # psycopg3 JSON adapter

ROOT = pathlib.Path(__file__).resolve().parent
SQL_DIR = ROOT / "sql"

load_dotenv()

SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _connect():
    return psycopg.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
        dbname=os.getenv("PGDATABASE", "postgres"),
        autocommit=True,
    )

def _load_sql(name: str) -> str:
    path = SQL_DIR / name
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _get_table_columns(conn, table: str) -> List[str]:
    q = """
      SELECT column_name
      FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = %s
      ORDER BY ordinal_position
    """
    with conn.cursor() as cur:
        cur.execute(q, (table,))
        return [r[0] for r in cur.fetchall()]

def _pgify(value):
    return Json(value) if isinstance(value, (dict, list)) else value

# Keep for your run tables
def create_run_table(table_name: str) -> None:
    if not SAFE.match(table_name):
        raise ValueError("Unsafe table name. Use letters/digits/underscore; not starting with a digit.")
    ddl_template = _load_sql("create_run_table.sql")
    ddl = sql.SQL(ddl_template).format(table=sql.Identifier(table_name))
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(ddl)

def insert_rows(table_name: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    if not SAFE.match(table_name):
        raise ValueError("Unsafe table name.")
    cols = ["case_id","label","prompt","output","ok","error","latency_ms","status","endpoint"]
    insert_template = _load_sql("insert_rows.sql")
    stmt = sql.SQL(insert_template).format(
        table=sql.Identifier(table_name),
        fields=sql.SQL(", ").join(map(sql.Identifier, cols)),
        placeholders=sql.SQL(", ").join(sql.Placeholder() * len(cols)),
    )
    values = [tuple(r.get(k) for k in cols) for r in rows]
    with _connect() as conn, conn.cursor() as cur:
        cur.executemany(stmt, values)




CASE_COL_CANDIDATES = ["id","pack","version","label","prompt","tags","props","source","expected","checks"]
JSONB_COLS = {"props","source","expected","checks"}  # only these get Json()

def _get_table_columns(conn, table: str) -> list[str]:
    q = """
      SELECT column_name
      FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = %s
      ORDER BY ordinal_position
    """
    with conn.cursor() as cur:
        cur.execute(q, (table,))
        return [r[0] for r in cur.fetchall()]

def _normalize_case_for_cols(case: dict, cols: list[str]) -> dict:
    out = dict(case)
    if "tags" in cols:
        # ensure text[] friendly value
        tags = out.get("tags", [])
        if not isinstance(tags, (list, tuple)):
            tags = [str(tags)]
        out["tags"] = [str(t) for t in tags]
    if "props" in cols:
        out.setdefault("props", {})
    if "source" in cols:
        out.setdefault("source", {})
    # don't add expected/checks unless they already exist in the case or the table,
    # and even then we won't force defaults here.
    return out

def _adapt_value(col: str, val):
    # Only wrap JSONB columns. Leave lists (tags) alone so psycopg maps them to text[].
    if col in JSONB_COLS:
        from psycopg.types.json import Json
        # if val is None, Postgres jsonb NULL is fine; otherwise wrap dict/list
        return Json(val if val is not None else None)
    # tags must be a Python list/tuple of strings for text[]
    if col == "tags":
        if val is None:
            return []
        if not isinstance(val, (list, tuple)):
            return [str(val)]
        return [str(x) for x in val]
    return val

def add_cases(cases):
    cases = list(cases)
    if not cases:
        return
    with _connect() as conn, conn.cursor() as cur:
        table_cols = _get_table_columns(conn, "cases")
        cols_to_use = [c for c in CASE_COL_CANDIDATES if c in table_cols]
        if "id" not in cols_to_use:
            raise RuntimeError("cases table must have an 'id' column")

        placeholders = sql.SQL(", ").join(sql.Placeholder() * len(cols_to_use))
        insert_cols_sql = sql.SQL(", ").join(map(sql.Identifier, cols_to_use))
        update_assignments = sql.SQL(", ").join(
            sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c))
            for c in cols_to_use if c != "id"
        )

        stmt = sql.SQL("""
            INSERT INTO cases ({cols})
            VALUES ({vals})
            ON CONFLICT (id) DO UPDATE SET
            {updates}
        """).format(cols=insert_cols_sql, vals=placeholders, updates=update_assignments)

        batch = []
        for case in cases:
            norm = _normalize_case_for_cols(case, cols_to_use)
            row = tuple(_adapt_value(c, norm.get(c)) for c in cols_to_use)
            batch.append(row)

        cur.executemany(stmt, batch)
