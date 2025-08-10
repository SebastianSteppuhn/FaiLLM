# Routines for clustering, deduplicating, and refining cases.

from __future__ import annotations

import traceback
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
from psycopg import sql

from .db import connect, SAFE_IDENT
import json
from .llm_client import LLMClient                    


try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    KMeans = None
    silhouette_score = None
    TfidfVectorizer = None





def _table_exists(cur, name: str) -> bool:
    """Check if public.<name> exists."""
    cur.execute("SELECT to_regclass(%s)", (f"public.{name}",))
    return cur.fetchone()[0] is not None


def _resolve_tables(ref_table: str) -> tuple[str | None, str | None, str]:
    """
    Return (run_table, eval_table, mode)
    mode: "run+eval" when both exist, "eval-only" when only eval exists.
    - If ref_table starts with 'eval_', treat it as the eval table and derive run table.
    - Else treat it as run table and derive eval table.
    """
    if not SAFE_IDENT.match(ref_table or ""):
        raise ValueError("Unsafe table name.")
    if ref_table.startswith("eval_"):
        run_table = ref_table[len("eval_"):]
        eval_table = ref_table
    else:
        run_table = ref_table
        eval_table = f"eval_{ref_table}"

    with connect() as conn, conn.cursor() as cur:
        has_run = _table_exists(cur, run_table)
        has_eval = _table_exists(cur, eval_table)

    if has_run and has_eval:
        return run_table, eval_table, "run+eval"
    if has_eval and not has_run:
        return None, eval_table, "eval-only"

    raise ValueError(
        f"Could not find matching tables. "
        f"Tried run='{run_table}', eval='{eval_table}'. "
        f"Make sure you ran evaluation to create '{eval_table}'."
    )


def _fetch_reasoning_rows(ref_table: str, limit: Optional[int]) -> list[dict]:
    """Load rows from run+eval (preferred) or eval-only, newest first."""
    run_table, eval_table, mode = _resolve_tables(ref_table)
    params = []
    if mode == "run+eval":
        q = sql.SQL("""
            SELECT r.id AS run_id,
                   r.label,
                   r.prompt,
                   r.output,
                   e.pass AS passed,
                   e.reason AS eval_reason
            FROM {} r
            JOIN {} e ON e.run_row_id = r.id
            ORDER BY r.id DESC
        """).format(sql.Identifier(run_table), sql.Identifier(eval_table))
        if limit:
            q = q + sql.SQL(" LIMIT %s")
            params.append(limit)
        with connect() as conn, conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()
            cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in rows]


    q = sql.SQL("""
        SELECT e.run_row_id AS run_id,
               e.label,
               NULL::text AS prompt,
               NULL::text AS output,
               e.pass AS passed,
               e.reason AS eval_reason
        FROM {} e
        ORDER BY e.run_row_id DESC
    """).format(sql.Identifier(eval_table))
    if limit:
        q = q + sql.SQL(" LIMIT %s")
        params.append(limit)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(q, params)
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


def _compose_text(row: dict, fields: list[str]) -> str:
    """Compose a reasoning string using selected fields."""
    parts = []
    if "label" in fields:
        parts.append(f"[LABEL]\n{row.get('label','')}")
    if "prompt" in fields:
        parts.append(f"[PROMPT]\n{row.get('prompt','')}")
    if "output" in fields:
        parts.append(f"[OUTPUT]\n{row.get('output','')}")
    if "reason" in fields:
        parts.append(f"[EVAL_REASON]\n{row.get('eval_reason','')}")
    return "\n\n".join(parts).strip()


def _cluster_label_stats(assignments: list[int], sample_labels: list[str]):
    """Return (counts_by_cluster, top_label_by_cluster)."""
    agg: dict[int, dict[str, int]] = {}
    for c, lab in zip(assignments, sample_labels):
        lab = lab or ""
        d = agg.setdefault(c, {})
        d[lab] = d.get(lab, 0) + 1
    top = {c: max(d.items(), key=lambda kv: (kv[1], kv[0]))[0] if d else "" for c, d in agg.items()}
    return agg, top





def _auto_k(emb: np.ndarray, k_min: int = 2, k_max: int = 10) -> int:
    """
    Choose K with silhouette, but be safe for tiny n:
    - n <= 1  -> 1 cluster
    - n == 2  -> 2 clusters
    - else    -> argmax silhouette for k in [2 .. min(k_max, n-1)]
    If sklearn is unavailable, return a small default bounded by n.
    """
    n = len(emb)
    if n <= 1:
        return 1
    if n == 2:
        return 2

    if KMeans is None or silhouette_score is None:
        return min(5, max(2, n - 1))

    max_k = min(k_max, n - 1)                                     
    best_k, best_score = 2, -1.0
    for k in range(2, max_k + 1):
        try:
            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            labels = km.fit_predict(emb)
            s = silhouette_score(emb, labels)
        except Exception:
            s = -1.0
        if s > best_score:
            best_k, best_score = k, s
    return best_k


def _cluster(emb: np.ndarray, texts: list[str], k: Optional[int]) -> Tuple[np.ndarray, List[int]]:
    """
    Returns (centers, labels). Robust to tiny n and missing sklearn:
    - n == 0 -> ([], [])
    - n == 1 or k==1 -> one cluster
    - sklearn missing -> degrade gracefully to one cluster
    - otherwise KMeans with k in [2 .. n-1]
    """
    n = len(texts)
    if n == 0:
        return np.zeros((0,)), []


    if k is None or int(k) < 1:
        k = _auto_k(emb)
    else:
        k = int(k)

    if n == 1 or k == 1 or KMeans is None:

        center = emb.mean(axis=0) if emb.size else np.zeros((1,))
        labels = [0] * n
        return center.reshape(1, -1), labels


    k = max(2, min(k, n - 1))

    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(emb)
    return km.cluster_centers_, labels.tolist()


def _top_terms_per_cluster(texts: list[str], labels: list[int], topn: int = 6) -> Dict[int, List[str]]:
    """Top TF-IDF terms per cluster (empty if sklearn not available)."""
    if TfidfVectorizer is None:
        return {}
    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    clusters: Dict[int, List[str]] = {}
    for c in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == c]
        if not idx:
            clusters[c] = []
            continue
        sub = X[idx].mean(axis=0)
        arr = np.asarray(sub).ravel()
        top_idx = arr.argsort()[-topn:][::-1]
        clusters[c] = [terms[i] for i in top_idx if arr[i] > 0]
    return clusters


def _summarize_clusters(texts: list[str], labels: list[int], max_examples: int = 2) -> list[dict]:
    """Return a small summary dict per cluster (count + a couple examples)."""
    out = []
    for c in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == c]
        samples = [texts[i][:400] for i in idx[:max_examples]]
        out.append({"cluster": c, "count": len(idx), "examples": samples})
    return out


def _build_report(pass_clusters: list[dict], fail_clusters: list[dict],
                  pass_terms: Dict[int, List[str]], fail_terms: Dict[int, List[str]]) -> str:
    lines: List[str] = []
    lines.append("# Refinement Report\n")
    lines.append("## Strengths (PASS clusters)")
    if not pass_clusters:
        lines.append("_No PASS clusters found._")
    else:
        for c in pass_clusters:
            terms = ", ".join(pass_terms.get(c["cluster"], [])[:6])
            lines.append(f"- **Cluster {c['cluster']}** (n={c['count']}) – Top terms: {terms}")
    lines.append("\n## Weaknesses (FAIL clusters)")
    if not fail_clusters:
        lines.append("_No FAIL clusters found._")
    else:
        for c in fail_clusters:
            terms = ", ".join(fail_terms.get(c["cluster"], [])[:6])
            lines.append(f"- **Cluster {c['cluster']}** (n={c['count']}) – Top terms: {terms}")
    lines.append("\n### Notes")
    lines.append("- Prioritize larger FAIL clusters (highest n).")
    lines.append("- Turn top terms into guardrails/prompts/tests.")
    return "\n".join(lines)


def _normalize_text(s: str) -> str:
    return " ".join((s or "").split())


def _apply_label_filter(label: str, needle: Optional[str]) -> bool:
    if not needle:
        return True
    return (needle.lower() in (label or "").lower())


def _dedupe_parallel(texts: list[str], labels: list[str]) -> tuple[list[str], list[str]]:
    """Dedupe by composed text value (parallel over labels)."""
    seen = set()
    t2, l2 = [], []
    for t, lab in zip(texts, labels):
        key = t
        if key in seen:
            continue
        seen.add(key)
        t2.append(t); l2.append(lab)
    return t2, l2





def refine_embed_and_cluster(
        run_table: str,
        provider: str,
        api_key: Optional[str],
        base_url: Optional[str],
        embed_model: Optional[str],
        fields: list[str],
        limit: Optional[int],
        auto_k_flag: bool,
        k_pass: Optional[int],
        k_fail: Optional[int],
        side_filter: str = "both",
        label_filter: Optional[str] = None,
        min_chars: int = 0,
        dedupe: bool = True,
    ):
    """
    Returns: (df_pass: DataFrame, df_fail: DataFrame, report_md: str)
    """
    try:
        if not run_table:
            return pd.DataFrame(), pd.DataFrame(), "Please select a run table first."
        rows = _fetch_reasoning_rows(run_table, int(limit) if limit else None)
        if not rows:
            return pd.DataFrame(), pd.DataFrame(), "No data found (check that eval_* exists)."


        fields = fields or ["label", "prompt", "output", "reason"]                    
        texts_pass, texts_fail = [], []
        labels_pass, labels_fail = [], []

        for r in rows:
            passed = bool(r.get("passed"))
            if side_filter == "pass" and not passed:
                continue
            if side_filter == "fail" and passed:
                continue
            if not _apply_label_filter(r.get("label") or "", label_filter):
                continue

            t = _compose_text(r, fields)
            t = _normalize_text(t)
            if len(t) < int(min_chars or 0):
                continue

            if passed:
                texts_pass.append(t); labels_pass.append(r.get("label") or "")
            else:
                texts_fail.append(t); labels_fail.append(r.get("label") or "")

        if dedupe:
            texts_pass, labels_pass = _dedupe_parallel(texts_pass, labels_pass)
            texts_fail, labels_fail = _dedupe_parallel(texts_fail, labels_fail)


        client = LLMClient(
            provider=(provider or "ollama"),
            model=(embed_model or ""),
            temperature=0.0,
            timeout=120,
            base_url=(base_url if (provider or "ollama").lower() == "ollama" else None),
            api_key=(api_key or None),
        )
        emb_pass = np.array(client.embed(texts_pass)) if texts_pass else np.zeros((0, 0))
        emb_fail = np.array(client.embed(texts_fail)) if texts_fail else np.zeros((0, 0))


        kp = None if auto_k_flag else (int(k_pass) if k_pass else None)
        kf = None if auto_k_flag else (int(k_fail) if k_fail else None)

        _, labels_p = (np.zeros((0,)), []) if len(texts_pass) == 0 else _cluster(emb_pass, texts_pass, kp)
        _, labels_f = (np.zeros((0,)), []) if len(texts_fail) == 0 else _cluster(emb_fail, texts_fail, kf)


        pass_summ = _summarize_clusters(texts_pass, labels_p)
        fail_summ = _summarize_clusters(texts_fail, labels_f)
        pass_terms = _top_terms_per_cluster(texts_pass, labels_p)
        fail_terms = _top_terms_per_cluster(texts_fail, labels_f)

        pass_label_counts, pass_label_top = (_cluster_label_stats(labels_p, labels_pass) if len(labels_p) > 0 else ({}, {}))  
        fail_label_counts, fail_label_top = (_cluster_label_stats(labels_f, labels_fail) if len(labels_f) > 0 else ({}, {}))

        df_pass = pd.DataFrame([{
            "cluster": c["cluster"],
            "count": c["count"],
            "top_terms": ", ".join(pass_terms.get(c["cluster"], [])[:6]),
            "top_label": pass_label_top.get(c["cluster"], ""),
            "label_counts": ", ".join(f"{k}:{v}" for k,v in sorted(pass_label_counts.get(c["cluster"], {}).items(), key=lambda kv: (-kv[1], kv[0]))),
            "example_1": (c["examples"][0] if c["examples"] else ""),
            "example_2": (c["examples"][1] if len(c["examples"])>1 else ""),
        } for c in pass_summ])

        df_fail = pd.DataFrame([{
            "cluster": c["cluster"],
            "count": c["count"],
            "top_terms": ", ".join(fail_terms.get(c["cluster"], [])[:6]),
            "top_label": fail_label_top.get(c["cluster"], ""),
            "label_counts": ", ".join(f"{k}:{v}" for k,v in sorted(fail_label_counts.get(c["cluster"], {}).items(), key=lambda kv: (-kv[1], kv[0]))),
            "example_1": (c["examples"][0] if c["examples"] else ""),
            "example_2": (c["examples"][1] if len(c["examples"])>1 else ""),
        } for c in fail_summ])

        report = _build_report(pass_summ, fail_summ, pass_terms, fail_terms)
        return df_pass, df_fail, report

    except Exception:
        return pd.DataFrame(), pd.DataFrame(), traceback.format_exc()


def ai_summary_report(
        ref_table: str,
        provider: str,
        api_key: Optional[str],
        base_url: Optional[str],
        model: str,
        fields: List[str],
        limit: Optional[int],
        side: Optional[str] = None,
        *,
        label_filter: Optional[str] = None,
        min_chars: int = 0,
        dedupe: bool = True,
    ):
    """
    Returns a markdown string (LLM-generated).
    Recomputes clusters with given params, then asks the LLM for a synthesis.
    """

    df_pass, df_fail, _ = refine_embed_and_cluster(
        ref_table, provider, api_key, base_url, model, fields, limit,
        auto_k_flag=True, k_pass=None, k_fail=None,
        side_filter=("both" if side is None else side),
        label_filter=label_filter,
        min_chars=min_chars,
        dedupe=dedupe,
    )

    def _pack(df):
        rows = df.to_dict("records")
        return [{"cluster": r["cluster"], "count": r["count"],
                 "top_terms": (r.get("top_terms") or ""),
                 "top_label": (r.get("top_label") or "")} for r in rows]

    payload = {
        "pass_clusters": _pack(df_pass) if (side in (None, "pass")) else [],
        "fail_clusters": _pack(df_fail) if (side in (None, "fail")) else [],
    }

    synthesis_prompt = f"""
You are a senior LLM evaluation lead. Given clustered results (counts, top terms, top label),
write a concise, executive summary with sections:

1) Key strengths
2) Key failure modes
3) Top 3 priorities to fix (specific, measurable)
4) Suggestions for new targeted test-packs / prompts

Keep it under 350 words. Use bullet points. Data:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""
    def _as_text(res):
        # handles open/openai, ollama, and a few wrappers
        if res is None:
            return ""
        if isinstance(res, str):
            return res.strip()
        # open/openai chat completions
        ch = res.get("choices")
        if isinstance(ch, list) and ch:
            m = ch[0].get("message") or {}
            c = m.get("content")
            if isinstance(c, str):
                return c.strip()
            # some SDKs return plain "text" in choices
            if isinstance(ch[0].get("text"), str):
                return ch[0]["text"].strip()
        # ollama-style
        msg = res.get("message") or {}
        if isinstance(msg.get("content"), str):
            return msg["content"].strip()
        # common single-key fallbacks
        for k in ("text", "content", "output_text", "response"):
            v = res.get(k)
            if isinstance(v, str):
                return v.strip()
        return ""


    client = LLMClient(
        provider=provider, model=model, temperature=0.2, timeout=120,
        base_url=(base_url if provider.lower() == "ollama" else None), api_key=api_key
    )
    res = client.chat(synthesis_prompt)
    return _as_text(res)


def build_focus_prompt_from_clusters(
        ref_table: str,
        provider: str,
        api_key: Optional[str],
        base_url: Optional[str],
        embed_model: str,
        fields: List[str],
        limit: Optional[int],
        side: str = "fail",
        clusters_to_use: int = 3,
        terms_per_cluster: int = 6,
        include_examples: bool = False,
        target_label: Optional[str] = None
    ) -> str:
    df_pass, df_fail, _ = refine_embed_and_cluster(
        ref_table, provider, api_key, base_url, embed_model, fields, limit,
        auto_k_flag=True, k_pass=None, k_fail=None
    )
    df = df_fail if side.lower() == "fail" else df_pass
    if df.empty:
        return "No clusters found. Run evaluation/refinement first."


    df2 = df.sort_values("count", ascending=False).head(int(clusters_to_use))
    blocks = []
    for _, r in df2.iterrows():
        terms = (r.get("top_terms") or "").split(",")
        terms = [t.strip() for t in terms if t.strip()][:terms_per_cluster]
        lbl = (r.get("top_label") or "").strip()
        ex1 = r.get("example_1") or ""
        ex2 = r.get("example_2") or ""
        b = {
            "cluster": int(r["cluster"]),
            "top_terms": terms,
            "top_label": lbl,
            "examples": ([ex1, ex2] if include_examples else [])
        }
        blocks.append(b)

    label_hint = (target_label or "") or ("FAIL-mode focus" if side.lower() == "fail" else "PASS-mode characteristics")

    prompt = f"""You are a red-team case generator. Produce diverse, hard test cases that target the following patterns.
Goal: generate prompts that stress the model on **{label_hint}**.

Focus clusters (largest first):
{json.dumps(blocks, ensure_ascii=False, indent=2)}

Instructions:
- Vary phrasing, context, domains, and languages.
- Include boundary cases and malformed inputs.
- Prefer short prompts (<= 2 sentences) unless necessary.
- Bias towards {side.upper()}-like behavior (recreate weaknesses if FAIL, or stress strengths if PASS).
Output format: one JSON object per line with fields:
{{"label": "{target_label or ''}", "prompt": "<the test prompt>", "props": {{"source": "cluster-guided"}}}}
"""
    return prompt.strip()
