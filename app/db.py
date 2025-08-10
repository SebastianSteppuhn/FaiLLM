import re
from typing import Any, Dict, List
import psycopg
from psycopg import sql
from .settings import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE
from pathlib import Path

SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SQL_DIR = Path(__file__).parent / "sql"

def connect() -> psycopg.Connection:
    return psycopg.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DATABASE,
        autocommit=True,
    )

def list_public_tables() -> List[str]:
    """List public tables in the current database."""
    q = sql.SQL(load_sql("list_public_tables.sql"))
    with connect() as conn, conn.cursor() as cur:
        cur.execute(q)
        return [r[0] for r in cur.fetchall()]

def create_cases_table(table_name: str) -> None:
    """Create a new 'cases-like' table with the expected columns."""
    if not SAFE_IDENT.match(table_name):
        raise ValueError("Unsafe table name. Use letters/digits/underscore; not starting with a digit.")
    ddl = sql.SQL(load_sql("create_cases_table.sql")).format(table_name=sql.Identifier(table_name))
    with connect() as conn, conn.cursor() as cur:
        cur.execute(ddl)


def load_sql(filename: str) -> str:
    return (SQL_DIR / filename).read_text()

def get_table_columns(conn: psycopg.Connection, table: str) -> List[str]:
    q = sql.SQL(load_sql("get_table_columns.sql")).format(
    table_name=sql.Literal(table)  
    )
    with conn.cursor() as cur:
        cur.execute(q) 
        return [r[0] for r in cur.fetchall()]

def create_run_table(table_name: str) -> None:
    if not SAFE_IDENT.match(table_name):
        raise ValueError("Unsafe table name.")
    ddl = sql.SQL(load_sql("create_run_table.sql")).format(table=sql.Identifier(table_name))
    with connect() as conn, conn.cursor() as cur:
        cur.execute(ddl)

def insert_rows(table_name: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    if not SAFE_IDENT.match(table_name):
        raise ValueError("Unsafe table name.")
    cols = ["case_id","label","prompt","output","ok","error","latency_ms","status","endpoint"]
    stmt = sql.SQL(load_sql("insert_template.sql")).format(
        table=sql.Identifier(table_name),
        fields=sql.SQL(", ").join(map(sql.Identifier, cols)),
        placeholders=sql.SQL(", ").join(sql.Placeholder() * len(cols)),
    )
    values = [tuple(r.get(k) for k in cols) for r in rows]
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(stmt, values)
