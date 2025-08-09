from __future__ import annotations
import argparse, json, time, uuid
from pathlib import Path
from typing import Dict, Any, List

from ollama_interface import chat
from db_handler import add_cases

ALLOWED_PACKS = {
    "core_injections", "malformed_json", "unicode_noise", "long_context",
    "html_fragments", "base64_noise", "contradictions", "code_switching",
}

def is_valid_case(obj: dict) -> tuple[bool, str]:
    required = ["pack", "version", "label", "prompt"]
    for k in required:
        if k not in obj:
            return False, f"missing_{k}"
    if obj["pack"] not in ALLOWED_PACKS:
        return False, f"bad_pack:{obj['pack']}"
    return True, "ok"

def normalize_case(obj: dict) -> dict:
    if not obj.get("id"):
        import uuid
        key = f"{obj['pack']}::{obj['label']}::{obj['prompt']}"
        obj["id"] = f"case_{uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:24]}"
    obj.setdefault("tags", [])
    obj.setdefault("props", {})
    obj.setdefault("source", {})
    # no expected/checks here
    return obj 

def main():
    ap = argparse.ArgumentParser(
        description="Generate LLM cases via Ollama and write directly to Postgres cases table (no checks)."
    )
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--prompt-file", help="Path to meta-prompt (text)")
    ap.add_argument("--prompt-text", help="Inline meta-prompt")
    ap.add_argument("--batches", type=int, default=20, help="Number of LLM calls")
    ap.add_argument("--sleep", type=float, default=0.7, help="Seconds between calls")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    elif args.prompt_text:
        prompt = args.prompt_text.strip()
    else:
        raise SystemExit("Provide --prompt-file or --prompt-text")

    total_kept = 0
    total_errors = 0

    for b in range(1, args.batches + 1):
        res = chat(
            prompt,
            base_url=args.base_url,
            model=args.model,
            system=None,
            temperature=args.temperature,
            timeout=args.timeout,
        )

        if not res.get("ok"):
            total_errors += 1
            print(f"[batch {b}] ERROR {res.get('status')}: {res.get('error')}")
            time.sleep(args.sleep)
            continue

        lines = [ln for ln in (res.get("text") or "").splitlines() if ln.strip()]
        parsed: List[Dict[str, Any]] = []

        for ln in lines:
            try:
                obj = json.loads(ln)
                ok, why = is_valid_case(obj)
                if not ok:
                    total_errors += 1
                    continue
                obj = normalize_case(obj)
                parsed.append(obj)
            except Exception:
                total_errors += 1

        if parsed:
            add_cases(parsed)
            total_kept += len(parsed)

        print(f"[batch {b}/{args.batches}] lines={len(lines)} → inserted {len(parsed)} (total {total_kept})")
        time.sleep(args.sleep)

    print(f"Done. Inserted {total_kept} cases. Errors: {total_errors}")

if __name__ == "__main__":
    main()
