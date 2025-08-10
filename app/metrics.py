# Metrics and visualizations used across the app.

from __future__ import annotations
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from .db import connect, SAFE_IDENT
from psycopg import sql
from .llm_client import LLMClient


try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
except Exception:
    KMeans = None
    PCA = None


from .refinement import _fetch_reasoning_rows, _compose_text, _normalize_text


def fetch_eval_df(eval_table: str, limit: Optional[int]) -> pd.DataFrame:
    if not SAFE_IDENT.match(eval_table or ""):
        raise ValueError("Unsafe table name.")
    lim = int(limit) if limit else 5000
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT * FROM {} ORDER BY run_row_id DESC LIMIT %s").format(sql.Identifier(eval_table)),
            (lim,),
        )
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def compute_eval_metrics_from_df(df: pd.DataFrame) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return "_No rows in this eval table._", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cols = {c.lower(): c for c in df.columns}
    lab_c = cols.get("label") or "label"
    pass_c = cols.get("pass") or "pass"
    id_c = cols.get("run_row_id") or (cols.get("id") or "id")

    s_pass = df[pass_c].astype(bool)
    total = len(df)
    passed = int(s_pass.sum())
    fail = total - passed
    pass_rate = (passed / total) if total else 0.0

    counts = df.groupby(lab_c, dropna=False)[pass_c].size().reset_index(name="count").sort_values("count", ascending=False)
    pr = (
        df.groupby(lab_c, dropna=False)[pass_c]
        .mean()
        .reset_index(name="pass_rate")
        .merge(counts, on=lab_c, how="left")
        .sort_values("count", ascending=False)
    )

    df_sorted = df.sort_values(by=id_c, ascending=True).reset_index(drop=True)
    trend = pd.DataFrame({
        "index": range(1, len(df_sorted)+1),
        "pass_cummean": df_sorted[pass_c].astype(int).expanding().mean()
    })

    lines = [
        "# Evaluation Metrics",
        f"- Total rows: **{total}**",
        f"- Passed: **{passed}**",
        f"- Failed: **{fail}**",
        f"- Pass rate: **{pass_rate:.2%}**",
        "",
        "## Top labels by volume",
    ]
    for _, r in counts.head(10).iterrows():
        label = r[lab_c] if pd.notna(r[lab_c]) else "(no label)"
        lines.append(f"- {label}: {int(r['count'])}")
    lines.append("")
    lines.append("## Labels with lowest pass rate (min 10 samples)")
    pr_min10 = pr[pr["count"] >= 10].sort_values("pass_rate", ascending=True).head(10)
    for _, r in pr_min10.iterrows():
        label = r[lab_c] if pd.notna(r[lab_c]) else "(no label)"
        lines.append(f"- {label}: {r['pass_rate']:.1%} (n={int(r['count'])})")

    return "\n".join(lines), counts.rename(columns={lab_c: "label"}), pr.rename(columns={lab_c: "label"}), trend

def compute_cluster_metrics_and_projection(
        ref_table: str,
        provider: str,
        api_key: Optional[str],
        base_url: Optional[str],
        embed_model: Optional[str],
        fields: list[str],
        limit: Optional[int],
        side_filter: str = "both",
        label_filter: Optional[str] = None,
        min_chars: int = 0,
        dedupe: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      pass_counts_df: DataFrame with columns [cluster, count, legend] for PASS side
      fail_counts_df: DataFrame with columns [cluster, count, legend] for FAIL side
      proj_df: DataFrame with columns [x, y, side, cluster, cluster_key]
               (PCA 2D projection; empty if sklearn unavailable)
    Legend is computed as the label of the point nearest to the centroid of each cluster.
    """
    rows = _fetch_reasoning_rows(ref_table, int(limit) if limit else None)
    if not rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


    fields = fields or ["label", "prompt", "output", "reason"]

    def _keep(label: str, needle: Optional[str]) -> bool:
        return True if not needle else (needle.lower() in (label or "").lower())

    texts_pass, texts_fail = [], []
    labels_pass, labels_fail = [], []

    for r in rows:
        passed = bool(r.get("passed"))
        if side_filter == "pass" and not passed:
            continue
        if side_filter == "fail" and passed:
            continue
        if not _keep(r.get("label") or "", label_filter):
            continue

        t = _normalize_text(_compose_text(r, fields))
        if len(t) < int(min_chars or 0):
            continue

        if passed:
            texts_pass.append(t)
            labels_pass.append(r.get("label") or "")
        else:
            texts_fail.append(t)
            labels_fail.append(r.get("label") or "")


    def _dedupe_pair(texts: list[str], labs: list[str]) -> tuple[list[str], list[str]]:
        seen = set()
        t2, l2 = [], []
        for t, lb in zip(texts, labs):
            if t in seen:
                continue
            seen.add(t)
            t2.append(t); l2.append(lb)
        return t2, l2

    if dedupe:
        texts_pass, labels_pass = _dedupe_pair(texts_pass, labels_pass)
        texts_fail, labels_fail  = _dedupe_pair(texts_fail, labels_fail)


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


    def _labels_for(E: np.ndarray) -> np.ndarray:
        n = len(E)
        if n <= 1 or KMeans is None:
            return np.zeros((n,), dtype=int)
        k = min(6, max(2, n - 1))
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        return km.fit_predict(E)

    lab_p = _labels_for(emb_pass)
    lab_f = _labels_for(emb_fail)


    def _rep_map(E: np.ndarray, labs_assign: np.ndarray, labels_str: list[str]) -> dict[int, str]:
        if E.size == 0 or len(labels_str) == 0:
            return {}
        rep = {}
        for c in sorted(set(labs_assign.tolist())):
            idx = np.where(labs_assign == c)[0]
            if idx.size == 0:
                rep[c] = ""
                continue
            centroid = E[idx].mean(axis=0)

            d2 = ((E[idx] - centroid) ** 2).sum(axis=1)
            j = int(idx[int(np.argmin(d2))])
            rep[c] = labels_str[j] or ""
        return rep

    rep_pass = _rep_map(emb_pass, lab_p, labels_pass) if lab_p.size else {}
    rep_fail = _rep_map(emb_fail, lab_f, labels_fail) if lab_f.size else {}


    def _counts_df(assign: np.ndarray, rep: dict[int, str]) -> pd.DataFrame:
        if assign.size == 0:
            return pd.DataFrame(columns=["cluster", "count", "legend"])
        vc = pd.Series(assign).value_counts().sort_index()
        df = pd.DataFrame({"cluster": vc.index.astype(int), "count": vc.values})
        df["legend"] = df["cluster"].map(lambda c: rep.get(int(c), f"C{int(c)}"))
        return df

    pass_counts_df = _counts_df(lab_p, rep_pass)
    fail_counts_df = _counts_df(lab_f, rep_fail)


    def _proj(E: np.ndarray):
        n = len(E)
        if n == 0 or PCA is None:
            return np.zeros((0, 2))
        X = E
        if X.ndim != 2:
            X = np.zeros((n, 2))
        if X.shape[1] < 2:
            X = np.pad(X, ((0, 0), (0, 2 - X.shape[1])), constant_values=0)
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(X)

    proj_p = _proj(emb_pass)
    proj_f = _proj(emb_fail)

    dfp = pd.DataFrame({
        "x": proj_p[:, 0] if proj_p.size else [],
        "y": proj_p[:, 1] if proj_p.size else [],
        "side": "pass",
        "cluster": lab_p.astype(int) if lab_p.size else []
    })
    dff = pd.DataFrame({
        "x": proj_f[:, 0] if proj_f.size else [],
        "y": proj_f[:, 1] if proj_f.size else [],
        "side": "fail",
        "cluster": lab_f.astype(int) if lab_f.size else []
    })

    proj_df = pd.concat([dfp, dff], ignore_index=True)


    if not proj_df.empty:
        proj_df["cluster_key"] = proj_df.apply(lambda r: f"{r['side'].upper()}-{int(r['cluster'])}", axis=1)


    return pass_counts_df, fail_counts_df, proj_df