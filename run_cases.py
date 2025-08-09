# run_cases.py
from __future__ import annotations

import os
import argparse
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import psycopg
from psycopg import sql

from ollama_interface import chat  # your Ollama client
from db_handler import create_run_table, insert_rows

load_dotenv()

DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

def get_connection():
    return psycopg.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
        dbname=os.getenv("PGDATABASE", "postgres"),
        autocommit=True,
    )

def fetch_cases(
    conn,
    packs: Optional[List[str]] = None,
    version: Optional[int] = None,
    label_like: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    where = []
    params: List[Any] = []

    if packs:
        where.append(sql.SQL("pack = ANY(%s)"))
        params.append(packs)
    if version is not None:
        where.append(sql.SQL("version = %s"))
        params.append(version)
    if label_like:
        where.append(sql.SQL("label ILIKE %s"))
        params.append(label_like)

    q = sql.SQL("SELECT id, pack, version, label, prompt FROM cases")
    if where:
        q = q + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where)
    q = q + sql.SQL(" ORDER BY id")
    if limit:
        q = q + sql.SQL(" LIMIT %s")
        params.append(limit)

    with conn.cursor() as cur:
        cur.execute(q, params)
        rows = cur.fetchall()

    return [
        {"id": r[0], "pack": r[1], "version": r[2], "label": r[3], "prompt": r[4]}
        for r in rows
    ]

def main():
    ap = argparse.ArgumentParser(description="Run cases from DB using Ollama (no checks).")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--packs", nargs="*", help="Filter by pack(s)")
    ap.add_argument("--version", type=int, help="Filter by version")
    ap.add_argument("--label-like", help="ILIKE filter for labels, e.g. %json%")
    ap.add_argument("--limit", type=int, help="Max cases to run")
    ap.add_argument("--system", help="Optional system prompt")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--no-artifacts", action="store_true", help="Skip writing local artifact files")
    args = ap.parse_args()

    run_table = "attacks_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    create_run_table(run_table)

    with get_connection() as conn:
        cases = fetch_cases(
            conn,
            packs=args.packs,
            version=args.version,
            label_like=args.label_like,
            limit=args.limit,
        )

    if not cases:
        print("No cases matched the filters.")
        return

    rows_for_db = []
    passed = failed = 0
    by_pack: Dict[str, Dict[str, int]] = {}

    # optional local artifacts
    if not args.no_artifacts:
        from pathlib import Path
        artifact_dir = Path("artifacts") / run_table
        artifact_dir.mkdir(parents=True, exist_ok=True)
        results_jsonl = artifact_dir / "results.jsonl"
        summary_json = artifact_dir / "summary.json"
        fout = results_jsonl.open("w", encoding="utf-8")
    else:
        artifact_dir = None
        results_jsonl = None
        summary_json = None
        fout = None

    try:
        for c in cases:
            res = chat(
                c["prompt"],
                base_url=args.base_url,
                model=args.model,
                system=args.system,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            text = (res.get("text") or "").strip()
            ok = bool(text) and bool(res.get("ok"))

            row = {
                "case_id": c["id"],
                "label": c.get("label"),
                "prompt": c["prompt"],
                "output": text,
                "ok": ok,
                "error": res.get("error"),
                "latency_ms": res.get("latency_ms"),
                "status": res.get("status"),
                "endpoint": res.get("endpoint") or "chat",
            }
            rows_for_db.append(row)

            if fout:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            # counters
            passed += int(ok)
            failed += int(not ok)
            pack_stats = by_pack.setdefault(c.get("pack") or "default", {"passed": 0, "failed": 0, "total": 0})
            pack_stats["total"] += 1
            if ok: pack_stats["passed"] += 1
            else: pack_stats["failed"] += 1

            preview = (text[:100] if text else "").replace("\n", " ")
            print(f"[{c['pack']}/{c['label']}] {'OK' if ok else 'FAIL'} {res.get('latency_ms')}ms → {preview}")

    finally:
        if fout:
            fout.close()

    # write to run table in DB
    insert_rows(run_table, rows_for_db)

    # write summary artifact (optional)
    if summary_json:
        summary = {
            "table": run_table,
            "model": args.model,
            "base_url": args.base_url,
            "passed": passed,
            "failed": failed,
            "total": len(cases),
            "by_pack": by_pack,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "artifact_dir": str(artifact_dir),
            "results_jsonl": str(results_jsonl),
        }
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\nSummary:", json.dumps(summary, indent=2))
    else:
        print(f"\nRun complete. Table: {run_table} | total={len(cases)} passed={passed} failed={failed}")

if __name__ == "__main__":
    main()