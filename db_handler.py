import os, re, pathlib
from typing import  Dict, Any, List

from dotenv import load_dotenv
import psycopg
from psycopg import sql

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
