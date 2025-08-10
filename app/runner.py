# Batch runner that executes prompts against the selected LLM.
from __future__ import annotations
import os, json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from psycopg import sql

from .llm_client import LLMClient
from .db import connect, create_run_table, insert_rows
from .cases import get_cases_table


def _select_cases_from_db(
    packs: Optional[List[str]],
    version: Optional[int],
    label_like: Optional[str],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    """Query the active cases table directly, applying the same filters as the UI."""
    table = get_cases_table()


    where_parts = []
    params: List[Any] = []

    if packs:
        where_parts.append(sql.SQL("pack = ANY(%s)"))
        params.append(packs)

    if version is not None:
        where_parts.append(sql.SQL("version = %s"))
        params.append(int(version))

    if label_like:
        where_parts.append(sql.SQL("label ILIKE %s"))
        params.append(str(label_like))

    where_sql = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_parts) if where_parts else sql.SQL("")

    q = sql.SQL("""
        SELECT id, pack, version, label, prompt
        FROM {}
    """).format(sql.Identifier(table)) + where_sql + sql.SQL(" ORDER BY id DESC ")

    if limit:
        q = q + sql.SQL(" LIMIT %s")
        params.append(int(limit))

    with connect() as conn, conn.cursor() as cur:
        cur.execute(q, params)
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


def run_cases(
    base_url: Optional[str],
    model: str,
    system: Optional[str],
    temperature: float,
    timeout: int,
    packs: Optional[List[str]] = None,
    version: Optional[int] = None,
    label_like: Optional[str] = None,
    limit: Optional[int] = None,
    write_artifacts: bool = True,
    provider: str = "ollama",
    api_key: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    run_table = "attacks_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    create_run_table(run_table)


    cases = _select_cases_from_db(packs=packs, version=version, label_like=label_like, limit=limit)

    if not cases:

        active_tbl = get_cases_table()
        with connect() as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(active_tbl)))
            total_in_table = cur.fetchone()[0]
            cur.execute(sql.SQL("SELECT DISTINCT pack FROM {}").format(sql.Identifier(active_tbl)))
            packs_present = [r[0] for r in cur.fetchall() if r[0] is not None]
            cur.execute(sql.SQL("SELECT DISTINCT version FROM {}").format(sql.Identifier(active_tbl)))
            versions_present = [r[0] for r in cur.fetchall() if r[0] is not None]

        return (
            run_table,
            pd.DataFrame(),
            {
                "table": run_table,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "passed": 0,
                "failed": 0,
                "total": 0,
                "by_pack": {},
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "active_cases_table": active_tbl,
                "diagnostics": {
                    "rows_in_active_table": total_in_table,
                    "distinct_packs": packs_present,
                    "distinct_versions": versions_present,
                    "filters_used": {
                        "packs": packs,
                        "version": version,
                        "label_like": label_like,
                        "limit": limit,
                    },
                },
                "hint": "Runner found no rows matching the filters. Check filters above or the active table content.",
            },
        )

    client = LLMClient(
        provider=provider or "ollama",
        model=model,
        temperature=temperature,
        timeout=timeout,
        base_url=base_url,
        api_key=api_key,
        reasoning_effort=reasoning_effort,
    )

    rows_for_db: List[Dict[str, Any]] = []
    passed = failed = 0
    by_pack: Dict[str, Dict[str, int]] = {}

    artifact_dir = results_path = summary_path = None
    fout = None
    try:
        if write_artifacts:
            artifact_dir = os.path.join("artifacts", run_table)
            os.makedirs(artifact_dir, exist_ok=True)
            results_path = os.path.join(artifact_dir, "results.jsonl")
            summary_path = os.path.join(artifact_dir, "summary.json")
            fout = open(results_path, "w", encoding="utf-8")

        for c in cases:
            res = client.chat(c["prompt"], system=system)
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
                "endpoint": res.get("endpoint") or provider,
                "model": model,
            }
            rows_for_db.append(row)
            if fout:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            passed += int(ok)
            failed += int(not ok)

            pack_name = (c.get("pack") or "default")
            stats = by_pack.setdefault(pack_name, {"passed": 0, "failed": 0, "total": 0})
            stats["total"] += 1
            if ok:
                stats["passed"] += 1
            else:
                stats["failed"] += 1
    finally:
        if fout:
            fout.close()

    insert_rows(run_table, rows_for_db)

    summary = {
        "table": run_table,
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "passed": passed,
        "failed": failed,
        "total": len(cases),
        "by_pack": by_pack,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "artifact_dir": artifact_dir,
        "results_jsonl": results_path,
    }
    if summary_path:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    preview = [
        {
            "case_id": r["case_id"],
            "label": r["label"],
            "ok": r["ok"],
            "latency_ms": r["latency_ms"],
            "status": r["status"],
            "endpoint": r["endpoint"],
            "output_preview": (r["output"] or "")[:200].replace("\n", " "),
        }
        for r in rows_for_db
    ]
    return run_table, pd.DataFrame(preview), summary
