# Utilities for managing adversarial cases in the database.
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
from psycopg import sql
from psycopg.types.json import Json
from .db import connect, get_table_columns

import re


CASES_TABLE = "cases"

ALLOWED_PACKS = {
    "core_injections", "malformed_json", "unicode_noise", "long_context",
    "html_fragments", "base64_noise", "contradictions", "code_switching",
}
CASE_COL_CANDIDATES = ["id","pack","version","label","prompt","tags","props","source","expected","checks"]
JSONB_COLS = {"props","source","expected","checks"}

_CTRL_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')

def _clean_text(s: str) -> str:
    if s is None:
        return s
    return _CTRL_RE.sub('', s)

def deep_clean(obj):
    """Recursively clean strings inside lists/tuples/dicts for DB safety."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return _clean_text(obj)
    if isinstance(obj, dict):
        return {deep_clean(k): deep_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [deep_clean(x) for x in obj]
        return tuple(t) if isinstance(obj, tuple) else t
    return obj

def set_cases_table(name: str) -> None:
    global CASES_TABLE
    CASES_TABLE = name

def get_cases_table() -> str:
    return CASES_TABLE

def is_valid_case(obj: Dict[str, Any]) -> Tuple[bool, str]:
    for k in ["pack","version","label","prompt"]:
        if k not in obj:
            return False, f"missing_{k}"
    if obj["pack"] not in ALLOWED_PACKS:
        return False, f"bad_pack:{obj['pack']}"
    return True, "ok"

def normalize_case(obj: Dict[str, Any]) -> Dict[str, Any]:
    import uuid
    if not obj.get("id"):
        key = f"{obj['pack']}::{obj['label']}::{obj['prompt']}"
        obj["id"] = f"case_{uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:24]}"
    obj.setdefault("tags", [])
    obj.setdefault("props", {})
    obj.setdefault("source", {})
    return obj

def _normalize_for_cols(case: Dict[str, Any], cols: List[str]) -> Dict[str, Any]:
    out = dict(case)
    if "tags" in cols:
        tags = out.get("tags", [])
        if not isinstance(tags, (list, tuple)):
            tags = [str(tags)]
        out["tags"] = [str(t) for t in tags]
    if "props" in cols:
        out.setdefault("props", {})
    if "source" in cols:
        out.setdefault("source", {})
    return out

def _adapt_value(col: str, val):
    if col in JSONB_COLS:
        return Json(val if val is not None else None)
    if col == "tags":
        if val is None:
            return []
        if not isinstance(val, (list, tuple)):
            return [str(val)]
        return [str(x) for x in val]
    return val

def add_cases(cases: Iterable[Dict[str, Any]]) -> int:
    cases = list(cases)
    if not cases:
        return 0
    with connect() as conn, conn.cursor() as cur:
        active = get_cases_table()
        table_cols = get_table_columns(conn, active)

        cols = [c for c in CASE_COL_CANDIDATES if c in table_cols]
        if "id" not in cols:
            raise RuntimeError("cases table must have an 'id' column")

        placeholders = sql.SQL(", ").join(sql.Placeholder() * len(cols))
        insert_cols = sql.SQL(", ").join(map(sql.Identifier, cols))
        updates = sql.SQL(", ").join(
            sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c))
            for c in cols if c != "id"
        )

        stmt = sql.SQL(
            "INSERT INTO {} ({cols}) VALUES ({vals}) ON CONFLICT (id) DO UPDATE SET {updates}"
        ).format(sql.Identifier(active), cols=insert_cols, vals=placeholders, updates=updates)

        batch = []
        for case in cases:

            norm = _normalize_for_cols(normalize_case(case), cols)
            row = tuple(_adapt_value(c, norm.get(c)) for c in cols)
            batch.append(row)

        sanitized_batch = [tuple(deep_clean(v) for v in row) for row in batch]

        cur.executemany(stmt, sanitized_batch)
        return len(batch)

def fetch_cases(
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

    table_ident = sql.Identifier(get_cases_table())
    q = sql.SQL("SELECT id, pack, version, label, prompt FROM {}").format(table_ident)
    if where:
        q = q + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where)
    q = q + sql.SQL(" ORDER BY id")
    if limit:
        q = q + sql.SQL(" LIMIT %s")
        params.append(limit)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(q, params)
        rows = cur.fetchall()

    return [
        {"id": r[0], "pack": r[1], "version": r[2], "label": r[3], "prompt": r[4]}
        for r in rows
    ]