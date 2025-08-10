# Case generator for producing edge-case prompts and packs.
from __future__ import annotations
import json, time
from typing import Any, Dict, List, Tuple, Optional
from .llm_client import LLMClient
from .cases import add_cases, is_valid_case, normalize_case

def generate_cases(*, prompt_text, base_url, model, temperature, timeout, batches, sleep_sec,
                   provider: str = "ollama", api_key: Optional[str] = None):
    logs: List[str] = []
    total_kept = 0
    total_errors = 0
    client = LLMClient(provider=provider, model=model, temperature=temperature,
                       timeout=timeout, base_url=base_url, api_key=api_key)
    for b in range(1, int(batches)+1):
        res = client.chat(prompt_text)
        if not res.get("ok"):
            total_errors += 1
            logs.append(f"[batch {b}] ERROR {res.get('status')}: {res.get('error')}")
            time.sleep(sleep_sec); continue
        lines = [ln for ln in (res.get("text") or "").splitlines() if ln.strip()]
        parsed: List[Dict[str, Any]] = []
        for ln in lines:
            try:
                obj = json.loads(ln)
                ok, why = is_valid_case(obj)
                if not ok:
                    total_errors += 1
                    logs.append(f"[batch {b}] drop: {why} → {ln[:100]}")
                    continue
                parsed.append(normalize_case(obj))
            except Exception:
                total_errors += 1
                logs.append(f"[batch {b}] bad json → {ln[:100]}")
        if parsed:
            inserted = add_cases(parsed)
            total_kept += inserted
            logs.append(f"[batch {b}/{batches}] lines={len(lines)} → inserted {inserted} (total {total_kept})")
        else:
            logs.append(f"[batch {b}/{batches}] lines={len(lines)} → inserted 0 (total {total_kept})")
        time.sleep(sleep_sec)
    logs.append(f"Done. Inserted {total_kept} cases. Errors: {total_errors}")
    return total_kept, total_errors, logs
