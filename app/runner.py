from __future__ import annotations
import os, json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from .ollama_client import OllamaClient
from .cases import fetch_cases
from .db import create_run_table, insert_rows

def run_cases(base_url: str, model: str, system: Optional[str], temperature: float, timeout: int,
              packs: Optional[List[str]]=None, version: Optional[int]=None, label_like: Optional[str]=None,
              limit: Optional[int]=None, write_artifacts: bool=True) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    run_table = "attacks_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    create_run_table(run_table)
    cases = fetch_cases(packs=packs, version=version, label_like=label_like, limit=limit)
    if not cases:
        return run_table, pd.DataFrame(), {"table": run_table, "model": model, "base_url": base_url,
                                           "passed": 0, "failed": 0, "total": 0, "by_pack": {},
                                           "timestamp": datetime.now().isoformat(timespec="seconds")}
    client = OllamaClient(base_url=base_url, model=model, temperature=temperature, timeout=timeout)
    rows_for_db: List[Dict[str, Any]] = []
    passed = failed = 0
    by_pack: Dict[str, Dict[str, int]] = {}
    artifact_dir = results_path = summary_path = None
    if write_artifacts:
        artifact_dir = os.path.join("artifacts", run_table)
        os.makedirs(artifact_dir, exist_ok=True)
        results_path = os.path.join(artifact_dir, "results.jsonl")
        summary_path = os.path.join(artifact_dir, "summary.json")
        fout = open(results_path, "w", encoding="utf-8")
    else:
        fout = None
    try:
        for c in cases:
            res = client.chat(c["prompt"], system=system)
            text = (res.get("text") or "").strip()
            ok = bool(text) and bool(res.get("ok"))
            row = {"case_id": c["id"], "label": c.get("label"), "prompt": c["prompt"], "output": text,
                   "ok": ok, "error": res.get("error"), "latency_ms": res.get("latency_ms"),
                   "status": res.get("status"), "endpoint": res.get("endpoint") or "chat"}
            rows_for_db.append(row)
            if fout:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            passed += int(ok)
            failed += int(not ok)
            pack_stats = by_pack.setdefault(c.get("pack") or "default", {"passed":0,"failed":0,"total":0})
            pack_stats["total"] += 1
            if ok: pack_stats["passed"] += 1
            else: pack_stats["failed"] += 1
    finally:
        if fout:
            fout.close()
    insert_rows(run_table, rows_for_db)
    summary = {"table": run_table, "model": model, "base_url": base_url, "passed": passed, "failed": failed,
               "total": len(cases), "by_pack": by_pack, "timestamp": datetime.now().isoformat(timespec="seconds"),
               "artifact_dir": artifact_dir, "results_jsonl": results_path}
    if summary_path:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    preview = [{"case_id": r["case_id"], "label": r["label"], "ok": r["ok"], "latency_ms": r["latency_ms"],
                "status": r["status"], "endpoint": r["endpoint"],
                "output_preview": (r["output"] or "")[:200].replace("\n"," ")} for r in rows_for_db]
    return run_table, pd.DataFrame(preview), summary