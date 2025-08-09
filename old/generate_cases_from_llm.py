import argparse, json, time, uuid, random, sys
from pathlib import Path
import requests

ALLOWED_PACKS = {
    "core_injections", "malformed_json", "unicode_noise", "long_context",
    "html_fragments", "base64_noise", "contradictions", "code_switching",
}

def ollama_chat(prompt: str, base_url: str, model: str, temperature: float = 0.2, timeout: int = 120) -> str:
    """Return plain text from /api/chat with fallback to /api/generate."""
    base = base_url.rstrip("/")
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(f"{base}/api/chat", json=payload, timeout=timeout)
    if r.status_code == 404:
        r = requests.post(
            f"{base}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
            timeout=timeout,
        )
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content") or data.get("response", "") or ""

def read_prompt(path: str | None, default_text: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    if default_text:
        return default_text.strip()
    print("ERROR: Provide --prompt-file or --prompt-text.", file=sys.stderr)
    sys.exit(2)

def is_valid_case(obj: dict) -> tuple[bool, str]:
    required = ["id", "pack", "version", "label", "prompt", "tags", "props", "expected", "source"]
    for k in required:
        if k not in obj:
            return False, f"missing_{k}"
    if obj["pack"] not in ALLOWED_PACKS:
        return False, f"bad_pack:{obj['pack']}"
    if not isinstance(obj.get("tags"), list):
        return False, "tags_not_list"
    if not isinstance(obj.get("props"), dict):
        return False, "props_not_obj"

#    if "checks" in obj and obj["checks"] is not None and not isinstance(obj["checks"], list):
#        return False, "checks_not_list"
    return True, "ok"

def ensure_id(obj: dict, counter: int) -> None:

    if not obj.get("id"):
        obj["id"] = f"case_{counter:06d}"

def main():
    ap = argparse.ArgumentParser(description="Generate test cases from an LLM via Ollama and save as JSONL.")
    ap.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    ap.add_argument("--prompt-file", help="Path to the generator meta-prompt (text)")
    ap.add_argument("--prompt-text", help="Inline generator meta-prompt (use instead of --prompt-file)")
    ap.add_argument("--batches", type=int, default=20, help="How many LLM calls to make")
    ap.add_argument("--sleep", type=float, default=0.7, help="Seconds to sleep between batches")
    ap.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    ap.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds")
    ap.add_argument("--out", default="testdata_generated.jsonl", help="Output JSONL file")
    ap.add_argument("--max-errors", type=int, default=50, help="Abort if total parse/validation errors exceed this")
    args = ap.parse_args()

    prompt = read_prompt(args.prompt_file, args.prompt_text)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set() 
    if out_path.exists():

        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen.add((rec.get("pack"), rec.get("label"), rec.get("prompt")))
                except Exception:
                    pass

    total_written = 0
    total_errors = 0
    id_counter = 1

    with out_path.open("a", encoding="utf-8") as out:
        for b in range(1, args.batches + 1):

            tries = 0
            while True:
                tries += 1
                try:
                    text = ollama_chat(prompt, args.base_url, args.model, args.temperature, args.timeout)
                    break
                except Exception as e:
                    if tries >= 3:
                        print(f"[batch {b}] ERROR: {e}", file=sys.stderr)
                        total_errors += 1
                        if total_errors > args.max_errors:
                            print("Too many errors, aborting.", file=sys.stderr)
                            sys.exit(1)
                        text = ""
                        break
                    time.sleep(1.5 * tries)

            lines = [ln for ln in text.splitlines() if ln.strip()]
            parsed = []
            for ln in lines:
                try:
                    obj = json.loads(ln)
                    ok, why = is_valid_case(obj)
                    if not ok:
                        total_errors += 1
                        continue
                    key = (obj["pack"], obj["label"], obj["prompt"])
                    if key in seen:
                        continue
                    ensure_id(obj, id_counter); id_counter += 1

                    if "checks" not in obj or obj["checks"] is None:
                        obj["checks"] = []
                    parsed.append(obj)
                    seen.add(key)
                except Exception:
                    total_errors += 1

            for obj in parsed:
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total_written += len(parsed)

            print(f"[batch {b}/{args.batches}] got {len(lines)} lines → kept {len(parsed)} (total {total_written})")

            time.sleep(args.sleep + random.random() * 0.3)

    print(f"Done. Wrote {total_written} cases to {out_path}. Errors: {total_errors}")

if __name__ == "__main__":
    main()

