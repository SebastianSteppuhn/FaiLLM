import argparse
import uuid
import json
import sys
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional, Tuple

from ollama_interface import chat
from db_handler import create_run_table, insert_rows

BASE_URL = "http://192.168.0.167:11434"
MODEL = "llama3.1:8b"
RUN_TABLE = "attacks_" + datetime.now().strftime("%Y%m%d_%H%M%S")

ARTIFACT_DIR = Path("artifacts") / RUN_TABLE
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSONL = ARTIFACT_DIR / "results.jsonl"
SUMMARY_JSON = ARTIFACT_DIR / "summary.json"


def iter_jsonl(fp) -> Iterable[Dict[str, Any]]:
    """Yield dicts from a JSONL stream, skipping blank/comment lines and validating JSON objects."""
    for i, line in enumerate(fp, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("Top-level JSON is not an object")
            yield obj
        except Exception as e:
            raise ValueError(f"Invalid JSONL at line {i}: {e}") from e


def load_cases(path: str) -> List[Dict[str, Any]]:
    """Load cases from a single JSONL file or '-' for stdin."""
    cases: List[Dict[str, Any]] = []
    if path == "-":
        cases.extend(iter_jsonl(sys.stdin))
        return cases

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if not p.is_file():
        raise ValueError(f"--cases must point to a single JSONL file, not a directory: {path}")

    with p.open("r", encoding="utf-8") as f:
        cases.extend(iter_jsonl(f))
    return cases


@dataclass
class CheckResult:
    ok: bool
    reason: Optional[str] = None


def check_json_only(text: str) -> CheckResult:
    stripped = text.strip()
    if not ((stripped.startswith("{") and stripped.endswith("}")) or
            (stripped.startswith("[") and stripped.endswith("]"))):
        return CheckResult(False, "Output is not a pure JSON object/array")
    try:
        json.loads(stripped)
        return CheckResult(True)
    except Exception as e:
        return CheckResult(False, f"JSON parse error: {e}")


def check_not_contains(text: str, substr: str, ignore_case: bool = True) -> CheckResult:
    hay = text.lower() if ignore_case else text
    needle = substr.lower() if ignore_case else substr
    ok = needle not in hay
    return CheckResult(ok, None if ok else f"Found forbidden substring: {substr!r}")


def check_contains(text: str, substr: str, ignore_case: bool = True) -> CheckResult:
    hay = text.lower() if ignore_case else text
    needle = substr.lower() if ignore_case else substr
    ok = needle in hay
    return CheckResult(ok, None if ok else f"Missing required substring: {substr!r}")


def check_equals(text: str, expected: str) -> CheckResult:
    ok = text.strip() == (expected or "").strip()
    return CheckResult(ok, None if ok else "Output not exactly equal to expected")


def check_regex(text: str, pattern: str, flags: str = "") -> CheckResult:
    flag_val = 0
    if "i" in flags: flag_val |= re.IGNORECASE
    if "m" in flags: flag_val |= re.MULTILINE
    if "s" in flags: flag_val |= re.DOTALL
    ok = re.search(pattern, text, flags=flag_val) is not None
    return CheckResult(ok, None if ok else f"Regex did not match: /{pattern}/{flags}")


def run_check(output_text: str, check: Dict[str, Any], expected: Optional[str]) -> CheckResult:
    ctype = check.get("type")
    if ctype == "json_only":
        return check_json_only(output_text)
    if ctype == "not_contains":
        return check_not_contains(
            output_text,
            substr=check.get("substr", ""),
            ignore_case=bool(check.get("ignore_case", True)),
        )
    if ctype == "contains":
        return check_contains(
            output_text,
            substr=check.get("substr", ""),
            ignore_case=bool(check.get("ignore_case(), True")),
        )
    if ctype == "equals":
        exp = check.get("expected", expected)
        return check_equals(output_text, exp if exp is not None else "")
    if ctype == "regex":
        return check_regex(
            output_text,
            pattern=check.get("pattern", ""),
            flags=check.get("flags", ""),
        )
    return CheckResult(False, f"Unknown check type: {ctype!r}")


def evaluate_case(case: Dict[str, Any], res: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Returns (ok, check_results)
    check_results: [{type, ok, reason}]
    """
    text = (res.get("text") or "").strip()
    checks = case.get("checks") or []
    expected = case.get("expected")

    if not checks:
        ok = bool(text)
        return ok, [{"type": "non_empty", "ok": ok, "reason": None if ok else "Empty output"}]

    results: List[Dict[str, Any]] = []
    all_ok = True
    for chk in checks:
        cr = run_check(text, chk, expected)
        all_ok = all_ok and cr.ok
        results.append({"type": chk.get("type"), "ok": cr.ok, "reason": cr.reason})
    return all_ok, results


def main():
    parser = argparse.ArgumentParser(description="Run LLM test cases from a single JSONL file (or stdin).")
    parser.add_argument("--cases", type=str, required=True,
                        help="Path to a JSONL file, or '-' to read from stdin.")
    parser.add_argument("--base-url", type=str, default=BASE_URL)
    parser.add_argument("--model", type=str, default=MODEL)
    args = parser.parse_args()

    base_url = args.base_url
    model = args.model

    cases = load_cases(args.cases)

    for c in cases:
        c.setdefault("id", str(uuid.uuid4()))
        c.setdefault("label", c.get("id"))
        c.setdefault("pack", "default")
        c.setdefault("version", 1)
        c.setdefault("prompt", "")
        c.setdefault("checks", [])
        c.setdefault("tags", [])
        c.setdefault("props", {})
        c.setdefault("expected", None)
        c.setdefault("source", {})

    create_run_table(RUN_TABLE)

    rows = []
    passed = failed = 0

    with RESULTS_JSONL.open("w", encoding="utf-8") as f:
        for c in cases:
            try:
                res = chat(c["prompt"], base_url=base_url, model=model)
                text = (res.get("text") or "").strip()
                ok, check_results = evaluate_case(c, res)
            except Exception as e:
                res = {"text": "", "error": str(e), "latency_ms": None,
                       "status": "exception", "endpoint": base_url}
                text = ""
                ok = False
                check_results = [{"type": "runtime_error", "ok": False, "reason": str(e)}]

            passed += int(ok)
            failed += int(not ok)

            row = {
                "case_id": c["id"],
                "label": c["label"],
                "pack": c.get("pack"),
                "version": c.get("version"),
                "prompt": c["prompt"],
                "output": text,
                "expected": c.get("expected"),
                "checks": c.get("checks"),
                "check_results": check_results,
                "tags": c.get("tags"),
                "props": c.get("props"),
                "source": c.get("source"),
                "ok": ok,
                "error": res.get("error"),
                "latency_ms": res.get("latency_ms"),
                "status": res.get("status"),
                "endpoint": res.get("endpoint") or base_url,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }

            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            preview = (text[:100] if text else "").replace("\n", " ")
            print(f"[{c['pack']}/{c['label']}] {'OK' if ok else 'FAIL'} {res.get('latency_ms')}ms → {preview}")

    insert_rows(RUN_TABLE, rows)

    by_pack: Dict[str, Dict[str, int]] = {}
    for r in rows:
        pack = r.get("pack") or "default"
        by_pack.setdefault(pack, {"passed": 0, "failed": 0, "total": 0})
        by_pack[pack]["total"] += 1
        if r["ok"]:
            by_pack[pack]["passed"] += 1
        else:
            by_pack[pack]["failed"] += 1

    summary = {
        "table": RUN_TABLE,
        "model": model,
        "base_url": base_url,
        "passed": passed,
        "failed": failed,
        "total": len(cases),
        "by_pack": by_pack,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "artifact_dir": str(ARTIFACT_DIR),
        "results_jsonl": str(RESULTS_JSONL),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
