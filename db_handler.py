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


def _normalize_tags(tags: Any) -> List[str]:
    if tags is None:
        return []
    if not isinstance(tags, (list, tuple)):
        raise ValueError("tags must be a list of strings")
    return [str(t) for t in tags]

def _case_params(case: Dict[str, Any]) -> Dict[str, Any]:
    required = ["id", "pack", "version", "prompt", "checks"]
    missing = [k for k in required if k not in case]
    if missing:
        raise ValueError(f"Missing required case fields: {', '.join(missing)}")

    return {
        "id": str(case["id"]),
        "pack": str(case["pack"]),
        "version": int(case["version"]),
        "label": case.get("label"),
        "prompt": case["prompt"],
        "expected": Json(case.get("expected")),  # None ok
        "checks": Json(case["checks"]),
        "tags": _normalize_tags(case.get("tags")),
        "props": Json(case.get("props") or {}),
        "source": Json(case.get("source") or {}),
    }

def add_case(case: Dict[str, Any]) -> None:
    """Insert or update a single case JSON into the cases table."""
    params = _case_params(case)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(_load_sql("upsert_case.sql"), params)

def add_cases(cases: Iterable[Dict[str, Any]]) -> None:
    """Bulk insert/update multiple cases efficiently."""
    cases = list(cases)
    if not cases:
        return
    param_list = [_case_params(c) for c in cases]
    with _connect() as conn, conn.cursor() as cur:
        cur.executemany(_load_sql("upsert_case.sql"), param_list)
