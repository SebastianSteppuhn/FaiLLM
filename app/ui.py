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

def ui_generate_cases(prompt_text, base_url, model, temperature, timeout, batches, sleep_sec):
    try:
        kept, errors, logs = generate_cases(
            prompt_text=prompt_text,
            base_url=base_url or OLLAMA_BASE_URL,
            model=model or OLLAMA_MODEL,
            temperature=float(temperature),
            timeout=int(timeout),
            batches=int(batches),
            sleep_sec=float(sleep_sec),
        )
        return "\n".join(logs), f"Inserted: {kept} | Errors: {errors}"
    except Exception:
        return traceback.format_exc(), "Failed. See logs."

def ui_run_cases(base_url, model, system_prompt, temperature, timeout, packs_csv, version, label_like, limit, write_artifacts):
    packs = [p.strip() for p in (packs_csv or "").split(",") if p.strip()] or None
    version_int = _parse_int(version)
    limit_int = _parse_int(limit)
    try:
        table, df, summary = run_cases(
            base_url=base_url or OLLAMA_BASE_URL,
            model=model or OLLAMA_MODEL,
            system=(system_prompt or None),
            temperature=float(temperature),
            timeout=int(timeout),
            packs=packs,
            version=version_int,
            label_like=(label_like or None),
            limit=limit_int,
            write_artifacts=bool(write_artifacts),
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

def ui_evaluate_cases(run_table, base_url, model, rubric, system_prompt, temperature, timeout, limit, sleep):
    try:
        eval_table, df, summary = evaluate_run_table(
            run_table=(run_table or "").strip(),
            base_url=base_url or OLLAMA_BASE_URL,
            model=model or OLLAMA_MODEL,
            rubric=rubric,
            system=(system_prompt or None),
            temperature=float(temperature),
            timeout=int(timeout),
            limit=_parse_int(limit),
            batch_sleep=float(sleep),
        )
        return eval_table, df, json.dumps(summary, indent=2)
    except Exception:
        return "", pd.DataFrame(), traceback.format_exc()


def build_app():
    with gr.Blocks(title="LLM Stress Test – DB + Ollama", theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown("# 🧪 LLM Stress Test\nGenerate adversarial cases and run them against an Ollama model, with results stored in Postgres.")

        with gr.Tabs():
            with gr.Tab("Database"):
                current_tbl = gr.Textbox(label="Active cases table (in-use)", value=get_cases_table(), interactive=False)

                with gr.Row(equal_height=True):
                    existing_dropdown = gr.Dropdown(
                        label="Existing tables (public schema)",
                        choices=ui_list_cases_tables(),
                        value=None,
                        scale=9,
                    )
                    existing_reload = gr.Button("🔄", scale=1, min_width=0)
                use_btn = gr.Button("Use selected", variant="primary")

                new_tbl_name = gr.Textbox(label="Create new cases table (name)")
                create_btn = gr.Button("Create & switch", variant="secondary")

                status_db = gr.Textbox(label="Status", interactive=False)


                existing_reload.click(ui_reload_cases_tables, inputs=[existing_dropdown], outputs=[existing_dropdown])
                use_btn.click(ui_use_cases_table, inputs=[existing_dropdown], outputs=[status_db, current_tbl, existing_dropdown])
                create_btn.click(ui_create_new_cases_table, inputs=[new_tbl_name], outputs=[status_db, current_tbl, existing_dropdown])

            with gr.Tab("Generate Cases"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row(equal_height=True):
                            prompt_list = gr.Dropdown(label="Prompt files", choices=list_prompt_files(), value=None, scale=9)
                            prompt_reload = gr.Button("🔄", scale=1, min_width=0)
                        new_name = gr.Textbox(label="Save As (filename, e.g. my_prompt.txt)")
                        save_btn = gr.Button("💾 Save As", variant="secondary")
                    with gr.Column(scale=2):
                        prompt_editor = gr.Textbox(label="Prompt editor", lines=16)

                with gr.Row(equal_height=True):
                    base_url = gr.Textbox(label="Ollama Base URL", value=OLLAMA_BASE_URL, scale=5)
                    model = gr.Dropdown(
                        label="Model (from Ollama)",
                        choices=list_ollama_models(OLLAMA_BASE_URL),
                        value=OLLAMA_MODEL if OLLAMA_MODEL in list_ollama_models(OLLAMA_BASE_URL) else None,
                        allow_custom_value=True,
                        scale=5,
                    )
                    model_reload = gr.Button("🔄", scale=1, min_width=0)

                with gr.Row():
                    temperature = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=0.2, label="Temperature")
                    timeout = gr.Number(value=180, precision=0, label="Timeout (s)")
                with gr.Row():
                    batches = gr.Number(value=20, precision=0, label="Batches")
                    sleep_sec = gr.Number(value=0.7, label="Sleep between calls (s)")
                run_btn = gr.Button("Generate & Upsert into cases", variant="primary")
                logs = gr.Textbox(label="Logs", lines=12)
                summary = gr.Textbox(label="Summary", interactive=False)

                prompt_list.change(ui_load_prompt, inputs=[prompt_list], outputs=[prompt_editor])
                prompt_reload.click(ui_reload_prompt_files, inputs=[prompt_list], outputs=[prompt_list])
                save_btn.click(ui_save_prompt_as, inputs=[new_name, prompt_editor], outputs=[summary, prompt_list])
                model_reload.click(ui_reload_models, inputs=[base_url, model], outputs=[model])
                run_btn.click(ui_generate_cases, inputs=[prompt_editor, base_url, model, temperature, timeout, batches, sleep_sec], outputs=[logs, summary])

            with gr.Tab("Run Cases"):
                base_url2 = gr.Textbox(label="Ollama Base URL", value=OLLAMA_BASE_URL)
                model2 = gr.Textbox(label="Model", value=OLLAMA_MODEL)
                system_prompt2 = gr.Textbox(label="Optional system prompt", lines=3)
                with gr.Row():
                    temperature2 = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=0.2, label="Temperature")
                    timeout2 = gr.Number(value=180, precision=0, label="Timeout (s)")
                with gr.Row():
                    packs_csv = gr.Textbox(label="Packs (comma-separated)")
                    version = gr.Number(label="Version (optional)", precision=0)
                    label_like = gr.Textbox(label="Label ILIKE filter (e.g. %json%)")
                    limit = gr.Number(label="Limit", precision=0)
                    write_artifacts = gr.Checkbox(label="Write artifacts to ./artifacts", value=True)

                run_cases_btn = gr.Button("Run Selected Cases", variant="primary")
                table_name = gr.Textbox(label="Run table name", interactive=False)
                df_out = gr.Dataframe(label="Run preview", interactive=False)
                summary_out = gr.Code(label="Summary JSON", language="json")
                run_cases_btn.click(ui_run_cases, inputs=[base_url2, model2, system_prompt2, temperature2, timeout2, packs_csv, version, label_like, limit, write_artifacts], outputs=[table_name, df_out, summary_out])

            with gr.Tab("Evaluate Cases"):
                with gr.Row(equal_height=True):
                    eval_run_table = gr.Dropdown(label="Run table to evaluate", choices=ui_list_cases_tables(), value=None, scale=9)
                    eval_run_table_reload = gr.Button("🔄", scale=1, min_width=0)

                with gr.Row(equal_height=True):
                    eval_base_url = gr.Textbox(label="Evaluator Ollama Base URL", value=OLLAMA_BASE_URL, scale=5)
                    eval_model = gr.Dropdown(
                        label="Evaluation Model",
                        choices=list_ollama_models(OLLAMA_BASE_URL),
                        value=OLLAMA_MODEL if OLLAMA_MODEL in list_ollama_models(OLLAMA_BASE_URL) else None,
                        allow_custom_value=True,
                        scale=5,
                    )
                    eval_model_reload = gr.Button("🔄", scale=1, min_width=0)

                rubric = gr.Textbox(label="Evaluation rubric", value=DEFAULT_RUBRIC, lines=8)
                eval_system_prompt = gr.Textbox(label="Optional evaluator system prompt", lines=2)
                with gr.Row():
                    eval_temp = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=0.0, label="Temperature")
                    eval_timeout = gr.Number(value=120, precision=0, label="Timeout (s)")
                    eval_limit = gr.Number(value=None, label="Limit rows (optional)", precision=0)
                    eval_sleep = gr.Number(value=0.15, label="Sleep between eval calls (s)")
                eval_btn = gr.Button("Evaluate run table", variant="primary")
                eval_table_name = gr.Textbox(label="Evaluation table name", interactive=False)
                eval_df = gr.Dataframe(label="Evaluation preview", interactive=False)
                eval_summary = gr.Code(label="Evaluation summary", language="json")

                eval_run_table_reload.click(ui_reload_cases_tables, inputs=[eval_run_table], outputs=[eval_run_table])
                eval_model_reload.click(ui_reload_models, inputs=[eval_base_url, eval_model], outputs=[eval_model])
                eval_btn.click(
                    ui_evaluate_cases,
                    inputs=[eval_run_table, eval_base_url, eval_model, rubric, eval_system_prompt, eval_temp, eval_timeout, eval_limit, eval_sleep],
                    outputs=[eval_table_name, eval_df, eval_summary],
                )

            with gr.Tab("Browse Run Table"):
                table_in = gr.Textbox(label="Run table name")
                limit_in = gr.Number(value=200, precision=0, label="Max rows")
                browse_btn = gr.Button("Load table")
                table_df = gr.Dataframe(label="Results", interactive=False)
                status = gr.Textbox(label="Status", interactive=False)
                browse_btn.click(ui_browse_table, inputs=[table_in, limit_in], outputs=[table_df, status])

        gr.Markdown("Made with ❤️  – Set PROMPTS_DIR to point at your folder (default: ./prompts).")
    return demo
