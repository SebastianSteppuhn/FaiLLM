import os
import json
from pathlib import Path
from dotenv import load_dotenv
import psycopg
from psycopg import sql
from psycopg.types.json import Json

load_dotenv()

ROOT = Path(__file__).resolve().parent
SQL_DIR = ROOT / "sql"


def get_connection():
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


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute(_load_sql("create_case_table.sql"))
        print("Created or verified `cases` table exists.")

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

JSONY = (dict, list)  # things we want to send to jsonb

def _pgify(value):
    # Wrap dict/list so psycopg adapts them to jsonb
    return Json(value) if isinstance(value, JSONY) else value

def upsert_cases(conn, cases_iterable):
    cols = ["id", "pack", "version", "label", "prompt", "expected", "checks", "tags", "props", "source"]
    insert_sql = sql.SQL(f"""
        INSERT INTO cases ({', '.join(cols)})
        VALUES ({', '.join(['%s'] * len(cols))})
        ON CONFLICT (id) DO UPDATE SET
          pack = EXCLUDED.pack,
          version = EXCLUDED.version,
          label = EXCLUDED.label,
          prompt = EXCLUDED.prompt,
          expected = EXCLUDED.expected,
          checks = EXCLUDED.checks,
          tags = EXCLUDED.tags,
          props = EXCLUDED.props,
          source = EXCLUDED.source
    """)
    with conn.cursor() as cur:
        batch = []
        for case in cases_iterable:
            row = tuple(_pgify(case.get(k)) for k in cols)
            batch.append(row)
            if len(batch) >= 500:
                cur.executemany(insert_sql, batch)
                batch.clear()
        if batch:
            cur.executemany(insert_sql, batch)
    print("Upserted cases into the table.")

def main():
    bench_path = Path("./datasets/llm_generated/test.jsonl")
    if not bench_path.exists():
        print(f"Benchmark dataset not found at {bench_path}")
        return

    conn = get_connection()
    create_table(conn)

    print(f"Loading cases from {bench_path}...")
    cases = list(load_jsonl(bench_path))
    print(f"Read {len(cases)} cases.")

    upsert_cases(conn, cases)
    print("Done loading benchmark cases.")

if __name__ == "__main__":
    main()
