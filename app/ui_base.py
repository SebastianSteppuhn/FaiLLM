# Core UI helpers and state shared across tabs.
# Core UI helpers and state shared across tabs.

from __future__ import annotations

import json
import pathlib
import re
import traceback
from typing import Optional

import gradio as gr
import pandas as pd
import requests
from psycopg import sql

from .cases import get_cases_table, set_cases_table
from .db import SAFE_IDENT, connect, create_cases_table, list_public_tables
from .evaluator import DEFAULT_RUBRIC, evaluate_run_table
from .generator import generate_cases
from .runner import run_cases
from .settings import OLLAMA_BASE_URL, OLLAMA_MODEL, PROMPTS_DIR
from .refinement import (
    refine_embed_and_cluster as ui_refine_embed_and_cluster,
    ai_summary_report,
    build_focus_prompt_from_clusters,
)




VALID_EXTS: set[str] = {".txt", ".md", ".json", ".prompt"}
FILENAME_SAFE = re.compile(r"^[A-Za-z0-9._\- ]+$")

try:
    with open("./app/style.css", "r", encoding="utf-8") as f:
        CSS = f.read()
except FileNotFoundError:
    CSS = ""


def _ensure_prompts_dir() -> pathlib.Path:
    p = pathlib.Path(PROMPTS_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parse_int(val: object, *, default: Optional[int] = None) -> Optional[int]:
    try:
        if val is None:
            return default
        s = str(val).strip()
        return int(s) if s != "" else default
    except Exception:
        return default


def list_ollama_models(base_url: str | None) -> list[str]:
    """Return sorted list of model names from Ollama."""
    try:
        base = (base_url or OLLAMA_BASE_URL).rstrip("/")
        resp = requests.get(f"{base}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json() or {}
        names = [m.get("name") for m in data.get("models", []) if isinstance(m, dict)]
        uniq = {n for n in names if isinstance(n, str) and n}
        return sorted(uniq, key=str.lower)
    except Exception:
        return []


def ui_refresh_models(base_url: str | None):
    models = list_ollama_models(base_url)
    value = models[0] if models else None
    return gr.update(choices=models, value=value)


def _refresh_dropdown_with_preserve(choices: list[str], current: str | None):
    """Return a gr.update that preserves the current value if still present."""
    value = current if current in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def list_prompt_files() -> list[str]:
    base = _ensure_prompts_dir()
    files = [f.name for f in base.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTS]
    files.sort(key=str.lower)
    return files


def read_prompt_file(name: str | None) -> str:
    if not name:
        return ""
    path = _ensure_prompts_dir() / name
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def save_prompt_file_as(name: str | None, content: str | None) -> tuple[bool, str]:
    base = _ensure_prompts_dir()
    safe_name = (name or "").strip()
    if not safe_name:
        return False, "Please provide a file name."
    if not FILENAME_SAFE.match(safe_name):
        return False, "Filename contains invalid characters. Allowed: letters, numbers, spaces, dot, underscore, dash."
    if pathlib.Path(safe_name).suffix.lower() not in VALID_EXTS:
        safe_name = safe_name + ".txt"
    path = base / safe_name
    try:
        path.write_text(content or "", encoding="utf-8")
        return True, f"Saved: {path}"
    except Exception as e:
        return False, f"Save failed: {type(e).__name__}: {e}"


def ui_list_cases_tables() -> list[str]:
    return list_public_tables()


def ui_use_cases_table(selected: str | None):
    if not selected:
        return "Select a table first.", get_cases_table(), gr.update(choices=ui_list_cases_tables())
    if not SAFE_IDENT.match(selected):
        return "Unsafe table name. Use letters/digits/underscore; not starting with a digit.", get_cases_table(), gr.update(choices=ui_list_cases_tables())
    set_cases_table(selected)
    return f"Using cases table: {selected}", get_cases_table(), gr.update(choices=ui_list_cases_tables(), value=selected)


def ui_create_new_cases_table(new_name: str | None):
    current = get_cases_table()
    if not new_name:
        return "Enter a table name.", current, gr.update(choices=ui_list_cases_tables())
    if not SAFE_IDENT.match(new_name):
        return "Unsafe table name. Use letters/digits/underscore; not starting with a digit.", current, gr.update(choices=ui_list_cases_tables())
    create_cases_table(new_name)
    set_cases_table(new_name)
    choices = ui_list_cases_tables()
    return f"Created & switched to: {new_name}", new_name, gr.update(choices=choices, value=new_name)


def ui_reload_cases_tables(current_value: str | None):
    """Reload public tables list and preserve current selection if possible."""
    return _refresh_dropdown_with_preserve(ui_list_cases_tables(), current_value)


def ui_reload_prompt_files(current_value: str | None):
    """Reload prompt file list and preserve current selection if possible."""
    return _refresh_dropdown_with_preserve(list_prompt_files(), current_value)


def ui_reload_models(base_url: str | None, current_value: str | None):
    """Reload models from Ollama and preserve current selection if possible."""
    choices = list_ollama_models(base_url)
    value = current_value if current_value in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)



def _effective_base_url(provider: str | None, base_url: str | None):
    provider = (provider or "ollama").lower()
    if provider == "openai":

        return None

    return (base_url or OLLAMA_BASE_URL)


def ui_generate_cases(
        prompt_text,
        refine_prompt_text,
        append_refine,
        base_url,
        model,
        temperature,
        timeout,
        batches,
        sleep_sec,
        provider,
        api_key,
    ):
    try:

        base = (prompt_text or "").strip()
        refine = (refine_prompt_text or "").strip()
        if append_refine and refine:
            prompt_text = f"{base}\n\n---\n# Additional refinement\n{refine}"
        else:
            prompt_text = base

        kept, errors, logs = generate_cases(
            prompt_text=prompt_text,
            base_url=_effective_base_url(provider, base_url),
            model=model or OLLAMA_MODEL,
            temperature=float(temperature),
            timeout=int(timeout),
            batches=int(batches),
            sleep_sec=float(sleep_sec),
            provider=provider or "ollama",
            api_key=(api_key or None),
        )
        return "\n".join(logs), f"Inserted: {kept} | Errors: {errors}"
    except Exception:
        return traceback.format_exc(), "Failed. See logs."

def ui_run_cases(base_url, model, system_prompt, temperature, timeout, packs_csv, version, label_like, limit, write_artifacts, provider, api_key):
    packs = [p.strip() for p in (packs_csv or "").split(",") if p.strip()] or None
    version_int = _parse_int(version)
    limit_int = _parse_int(limit)
    try:
        table, df, summary = run_cases(
            base_url=_effective_base_url(provider, base_url),
            model=model or OLLAMA_MODEL,
            system=(system_prompt or None),
            temperature=float(temperature),
            timeout=int(timeout),
            packs=packs,
            version=version_int,
            label_like=(label_like or None),
            limit=limit_int,
            write_artifacts=bool(write_artifacts),
            provider=provider or "ollama",
            api_key=(api_key or None),
        )
        return table, df, json.dumps(summary, indent=2)
    except Exception:
        return "", pd.DataFrame(), traceback.format_exc()


def ui_browse_table(table_name: str, limit: int | None):
    try:
        limit_int = _parse_int(limit, default=200) or 200
        if not SAFE_IDENT.match(table_name or ""):
            return pd.DataFrame(), "Unsafe table name."
        with connect() as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT * FROM {} ORDER BY id DESC LIMIT %s").format(sql.Identifier(table_name)), (limit_int,))
            rows = cur.fetchall()
            cols = [desc.name for desc in cur.description]
        return pd.DataFrame(rows, columns=cols), "OK"
    except Exception:
        return pd.DataFrame(), traceback.format_exc()


def ui_load_prompt(selected_name: str | None):
    return gr.update(value=read_prompt_file(selected_name))


def ui_save_prompt_as(new_name: str | None, content: str | None):
    ok, msg = save_prompt_file_as(new_name, content)
    choices = list_prompt_files()
    selected = None
    if ok:
        if new_name in choices:
            selected = new_name
        elif (new_name + ".txt") in choices:
            selected = new_name + ".txt"
    return msg, gr.update(choices=choices, value=selected)


def ui_evaluate_cases(run_table, base_url, model, rubric, system_prompt, temperature, timeout, limit, sleep, provider, api_key):
    try:
        eval_table, df, summary = evaluate_run_table(
            run_table=(run_table or "").strip(),
            base_url=_effective_base_url(provider, base_url),
            model=model or OLLAMA_MODEL,
            rubric=rubric,
            system=(system_prompt or None),
            temperature=float(temperature),
            timeout=int(timeout),
            limit=_parse_int(limit),
            batch_sleep=float(sleep),
            provider=provider or "ollama",
            api_key=(api_key or None),
        )
        return eval_table, df, json.dumps(summary, indent=2)
    except Exception:
        return "", pd.DataFrame(), traceback.format_exc()





def ui_reload_models_provider(provider: str, base_url: str | None, current_value: str | None):
    """Reload model list depending on provider. For OpenAI: free text only."""
    provider = (provider or "ollama").lower()
    if provider == "ollama":
        choices = list_ollama_models(base_url)
        value = current_value if current_value in choices else (choices[0] if choices else None)
        return gr.update(choices=choices, value=value)

    return gr.update(choices=[], value=current_value)


def ui_toggle_base_url(provider: str):
    """Show base_url field only for Ollama."""
    return gr.update(visible=(provider.lower() == "ollama"))


def ui_use_cases_table_dropdown(selected: str | None):
    """Switch active cases table and return only the dropdown update (no status boxes)."""
    _, _, dd = ui_use_cases_table(selected)
    return dd


def ui_create_new_cases_table_dropdown(new_name: str | None):
    """Create + switch cases table and return only the dropdown update."""
    _, _, dd = ui_create_new_cases_table(new_name)
    return dd


def ui_ai_summary(
    ref_table,
    sum_provider,
    sum_api_key,
    sum_base_url,
    embed_model,
    fields,
    limit,
    side,
    label_f,
    minc,
    dd,
    summary_model,
):
    chosen_model = (summary_model or embed_model or "").strip()
    if not chosen_model:
        msg = "Please select a **Summary LLM model**."
        return msg, msg


    provider = (sum_provider or "ollama").lower()
    base_url_effective = _effective_base_url(provider, sum_base_url)

    md = ai_summary_report(
        ref_table, provider, (sum_api_key or None), base_url_effective, chosen_model, fields, limit,
        side=(None if side == "both" else side),
        label_filter=label_f, min_chars=int(minc or 0), dedupe=bool(dd),
    )
    md = md or "_No summary_"
    return md, md


def _write_text_to_pdf(text: str) -> Optional[str]:
    """
    Write plain/markdown-ish text to a simple multi-page PDF.
    Returns a temp file path or None if reportlab isn't installed.
    """
    try:
        import tempfile, textwrap
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm

        fd, path = tempfile.mkstemp(suffix=".pdf")

        import os; os.close(fd)

        c = canvas.Canvas(path, pagesize=A4)
        width, height = A4
        left, top, bottom = 2*cm, height - 2*cm, 2*cm
        y = top
        wrap_cols = 95                        

        for line in (text or "").splitlines():
            wrapped_lines = textwrap.wrap(line, width=wrap_cols) or [""]
            for wl in wrapped_lines:
                c.drawString(left, y, wl)
                y -= 14
                if y < bottom:
                    c.showPage()
                    y = top

        c.save()
        return path
    except Exception:
        return None

def ui_export_pdf(text: str):
    """
    Gradio callback: returns a file path (or None) for a <gr.File>.
    """
    path = _write_text_to_pdf(text or "")

    if path is None:

        import tempfile
        fd, alt = tempfile.mkstemp(suffix=".txt")
        import os; os.close(fd)
        with open(alt, "w", encoding="utf-8") as f:
            f.write("Install reportlab to export PDF (pip install reportlab)\n\n")
            f.write(text or "")
        return alt
    return path

def ui_list_eval_tables() -> list[str]:
    """List only evaluation tables (prefix 'eval_')."""
    return [t for t in list_public_tables() if t.startswith("eval_")]


def _compute_eval_metrics_from_df(df: pd.DataFrame) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Given an eval_* dataframe with columns ['run_row_id','label','pass','reason',...],
    return:
      - markdown summary text
      - counts_by_label (label, count)
      - pass_rate_by_label (label, pass_rate, n)
      - pass_over_index (index, pass_cummean) for a quick trend proxy
    """
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
        lines.append(f"- {r[lab_c] or '(no label)'}: {int(r['count'])}")
    lines.append("")
    lines.append("## Labels with lowest pass rate (min 10 samples)")
    pr_min10 = pr[pr["count"] >= 10].sort_values("pass_rate", ascending=True).head(10)
    for _, r in pr_min10.iterrows():
        lines.append(f"- {r[lab_c] or '(no label)'}: {r['pass_rate']:.1%} (n={int(r['count'])})")

    return "\n".join(lines), counts, pr, trend


def ui_eval_metrics_load(eval_table: str, limit: Optional[int]):
    """Load rows from an eval_* table and compute metrics & plots data."""
    try:
        if not SAFE_IDENT.match(eval_table or ""):
            return "_Unsafe table name_", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        limit_int = _parse_int(limit, default=5000) or 5000
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT * FROM {} ORDER BY run_row_id DESC LIMIT %s").format(sql.Identifier(eval_table)),
                (limit_int,),
            )
            rows = cur.fetchall()
            cols = [desc.name for desc in cur.description]

        df = pd.DataFrame(rows, columns=cols)
        md, counts, pr, trend = _compute_eval_metrics_from_df(df)
        return md, counts, pr, trend
    except Exception:
        return traceback.format_exc(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def _apply_plot_style():
    """
    Set a soft, clean plot style that matches the app UI.
    Uses seaborn if available, otherwise falls back to matplotlib rcParams.
    """
    try:
        import seaborn as sns                
        sns.set_theme(
            style="whitegrid",
            context="talk",
            font_scale=0.9,
            rc={
                "axes.facecolor": "#ffffff",
                "figure.facecolor": "#ffffff",
                "axes.edgecolor": "#E5E7EB",
                "grid.color": "#E5E7EB",
                "axes.titleweight": "600",
                "axes.titlelocation": "left",
                "legend.frameon": True,
            },
        )

        sns.set_palette("pastel")
        return "seaborn"
    except Exception:
        import matplotlib as mpl
        mpl.rcParams.update({
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "axes.edgecolor": "#E5E7EB",
            "grid.color": "#E5E7EB",
            "axes.titleweight": "600",
            "axes.titlelocation": "left",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.frameon": True,
        })
        return "mpl"





def build_app():
    with gr.Blocks(title="LLM Stress Test – DB + Ollama/OpenAI", theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown("# 🧪 LLM Stress Test\nGenerate adversarial cases and run them against Ollama or OpenAI, with results stored in Postgres.")

