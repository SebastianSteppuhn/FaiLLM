# Tab builders for the Gradio interface.
# Tab builders for the Gradio interface.
from .ui_base import *

def build_tab_generate_cases():
    with gr.Tab("Generate Cases"):
        gr.Markdown("## Cases Table")
        with gr.Row(equal_height=True):
            prompt_case_tables = gr.Dropdown(
                label="Existing cases tables (cases)",
                choices=ui_list_cases_tables(),
                value=None,
                scale=9,
            )
            prompt_case_tables_reload = gr.Button("🔄", scale=1, min_width=0)
            prompt_use_btn = gr.Button("Use selected", variant="primary", scale=2)

        with gr.Row():
            new_tbl_name = gr.Textbox(label="Create new cases table (name)")
            create_btn = gr.Button("Create & switch", variant="secondary")

        gr.Markdown("## Prompts")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row(equal_height=True):
                    prompt_list = gr.Dropdown(
                        label="Prompt files",
                        choices=list_prompt_files(),
                        value=None,
                        scale=9,
                    )
                    prompt_reload = gr.Button("🔄", scale=1, min_width=0)
                new_name = gr.Textbox(label="Save As (filename, e.g. my_prompt.txt)")
                save_btn = gr.Button("💾 Save As", variant="secondary")
            with gr.Column(scale=2):
                prompt_editor = gr.Textbox(label="Base prompt", lines=12)
                refine_prompt_editor = gr.Textbox(
                    label="Refinement prompt (optional; appended to base)",
                    lines=8
                )
                append_refine = gr.Checkbox(value=True, label="Append refinement prompt on Generate")

        with gr.Accordion("Cluster-Guided Prompt (from Refinement)", open=False):
            focus_table = gr.Dropdown(
                label="Run or Eval table (eval_attacks)",
                choices=ui_list_cases_tables(),
                value=None,
            )
            focus_reload = gr.Button("🔄", scale=1, min_width=0)

            # Provider/Model just for the builder (self-contained)
            with gr.Row(equal_height=True):
                focus_provider = gr.Dropdown(label="Provider", choices=["ollama", "openai"], value="ollama", scale=2)
                focus_api_key = gr.Textbox(label="API Key (OpenAI)", type="password", placeholder="sk-...", scale=4)
                focus_base_url = gr.Textbox(label="Base URL (Ollama)", value=OLLAMA_BASE_URL, scale=6, visible=True)

            with gr.Row(equal_height=True):
                focus_model = gr.Dropdown(
                    label="Embedding Model",
                    choices=list_ollama_models(OLLAMA_BASE_URL),
                    value=None,
                    allow_custom_value=True,
                    scale=10,
                )
                focus_model_reload = gr.Button("🔄", scale=2, min_width=0)

            # Local limit (don’t reuse limit_in from Browse tab)
            focus_limit = gr.Number(value=500, precision=0, label="Max rows for clustering")

            with gr.Row():
                focus_side = gr.Radio(choices=["fail", "pass"], value="fail", label="Side to target")
                focus_clusters = gr.Number(value=3, precision=0, label="Clusters to include")
                focus_terms = gr.Number(value=6, precision=0, label="Terms per cluster")
                focus_examples = gr.Checkbox(value=False, label="Include short examples")

            target_label = gr.Textbox(
                label="Optional target label to stamp on new cases",
                placeholder="e.g., prompt-injection, bad-json"
            )
            build_focus_btn = gr.Button("Build Focus Prompt", variant="secondary")

            # wiring
            focus_reload.click(ui_reload_cases_tables, inputs=[focus_table], outputs=[focus_table])

            focus_provider.change(ui_toggle_base_url, inputs=[focus_provider], outputs=[focus_base_url])
            focus_model_reload.click(ui_reload_models_provider, inputs=[focus_provider, focus_base_url, focus_model], outputs=[focus_model])
            focus_provider.change(ui_reload_models_provider, inputs=[focus_provider, focus_base_url, focus_model], outputs=[focus_model])
            focus_base_url.change(ui_reload_models_provider, inputs=[focus_provider, focus_base_url, focus_model], outputs=[focus_model])

            def ui_build_focus_prompt(ref_table, provider, api_key, base_url, model, limit, side, clusters, terms, include_examples, tgt_label):
                txt = build_focus_prompt_from_clusters(
                    ref_table=ref_table,
                    provider=provider,
                    api_key=api_key or None,
                    base_url=(base_url if (provider or "ollama").lower() == "ollama" else None),
                    embed_model=model,
                    fields=["label", "prompt", "output", "reason"],
                    limit=int(limit) if limit else None,
                    side=side,
                    clusters_to_use=int(clusters or 3),
                    terms_per_cluster=int(terms or 6),
                    include_examples=bool(include_examples),
                    target_label=(tgt_label or None),
                )
                return gr.update(value=txt)

            build_focus_btn.click(
                ui_build_focus_prompt,
                inputs=[
                    focus_table, focus_provider, focus_api_key, focus_base_url, focus_model,
                    focus_limit, focus_side, focus_clusters, focus_terms, focus_examples, target_label
                ],
                outputs=[refine_prompt_editor],
            )

        gr.Markdown("## Provider & Model")
        with gr.Row(equal_height=True):
            gen_provider = gr.Dropdown(label="Provider", choices=["ollama", "openai"], value="ollama", scale=2)
            gen_api_key = gr.Textbox(label="API Key (OpenAI)", type="password", placeholder="sk-...", scale=4)
            base_url = gr.Textbox(label="Base URL (Ollama)", value=OLLAMA_BASE_URL, scale=6, visible=True)

        with gr.Row(equal_height=True):
            model = gr.Dropdown(
                label="Model",
                choices=list_ollama_models(OLLAMA_BASE_URL),
                value=OLLAMA_MODEL if OLLAMA_MODEL in list_ollama_models(OLLAMA_BASE_URL) else None,
                allow_custom_value=True,
                scale=10,
            )
            model_reload = gr.Button("🔄", scale=2, min_width=0)

        with gr.Accordion("Advanced generation settings", open=False):
            with gr.Row():
                temperature = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=1, label="Temperature")
                timeout = gr.Number(value=180, precision=0, label="Timeout (s)")
            with gr.Row():
                batches = gr.Number(value=5, precision=0, label="Batches")
                sleep_sec = gr.Number(value=0.7, label="Sleep between calls (s)")

        run_btn = gr.Button("Generate & Upsert into cases", variant="primary")
        with gr.Row():
            logs = gr.Textbox(label="Logs", lines=12)
            summary = gr.Textbox(label="Summary", interactive=False)

        # wiring: cases table select/create
        prompt_case_tables_reload.click(ui_reload_cases_tables, inputs=[prompt_case_tables], outputs=[prompt_case_tables])
        prompt_use_btn.click(ui_use_cases_table_dropdown, inputs=[prompt_case_tables], outputs=[prompt_case_tables])
        create_btn.click(ui_create_new_cases_table_dropdown, inputs=[new_tbl_name], outputs=[prompt_case_tables])

        # wiring: prompts/models/run
        prompt_list.change(ui_load_prompt, inputs=[prompt_list], outputs=[prompt_editor])
        prompt_reload.click(ui_reload_prompt_files, inputs=[prompt_list], outputs=[prompt_list])

        gen_provider.change(ui_toggle_base_url, inputs=[gen_provider], outputs=[base_url])
        model_reload.click(ui_reload_models_provider, inputs=[gen_provider, base_url, model], outputs=[model])
        gen_provider.change(ui_reload_models_provider, inputs=[gen_provider, base_url, model], outputs=[model])
        base_url.change(ui_reload_models_provider, inputs=[gen_provider, base_url, model], outputs=[model])

        save_btn.click(ui_save_prompt_as, inputs=[new_name, prompt_editor], outputs=[summary, prompt_list])
        run_btn.click(
            ui_generate_cases,
            inputs=[
                prompt_editor, refine_prompt_editor, append_refine,
                base_url, model, temperature, timeout, batches, sleep_sec, gen_provider, gen_api_key
            ],
            outputs=[logs, summary],
        )

    # =========================
    # Run Cases
    # =========================

def build_tab_run_cases():
    with gr.Tab("Run Cases"):
        gr.Markdown("## Cases Table (select)")
        with gr.Row(equal_height=True):
            run_case_tables = gr.Dropdown(
                label="Existing cases tables (cases)",
                choices=ui_list_cases_tables(),
                value=None,
                scale=9,
            )
            run_case_tables_reload = gr.Button("🔄", scale=1, min_width=0)
            run_use_btn = gr.Button("Use selected", variant="primary", scale=2)

        gr.Markdown("## Provider & Model")
        with gr.Row(equal_height=True):
            run_provider = gr.Dropdown(label="Provider", choices=["ollama", "openai"], value="ollama", scale=2)
            run_api_key = gr.Textbox(label="API Key (OpenAI)", type="password", placeholder="sk-...", scale=4)
            base_url2 = gr.Textbox(label="Base URL (Ollama)", value=OLLAMA_BASE_URL, scale=6, visible=True)

        with gr.Row(equal_height=True):
            run_model = gr.Dropdown(
                label="Model",
                choices=list_ollama_models(OLLAMA_BASE_URL),
                value=OLLAMA_MODEL if OLLAMA_MODEL in list_ollama_models(OLLAMA_BASE_URL) else None,
                allow_custom_value=True,
                scale=10,
            )
            run_model_reload = gr.Button("🔄", scale=2, min_width=0)

        with gr.Accordion("Run settings", open=True):
            system_prompt2 = gr.Textbox(label="Optional system prompt", lines=3)
            with gr.Row():
                temperature2 = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=0.2, label="Temperature")
                timeout2 = gr.Number(value=180, precision=0, label="Timeout (s)")
            with gr.Row():
                packs_csv = gr.Textbox(label="Packs (comma-separated)")
                version = gr.Number(label="Version (optional)", value=1)
                label_like = gr.Textbox(label="Label ILIKE filter (e.g. %json%)")
                limit = gr.Number(label="Limit", precision=0)
                write_artifacts = gr.Checkbox(label="Write artifacts to ./artifacts", value=True)

        run_cases_btn = gr.Button("Run Selected Cases", variant="primary")
        table_name = gr.Textbox(label="Run table name", interactive=False)
        df_out = gr.Dataframe(label="Run preview", interactive=False)
        summary_out = gr.Code(label="Summary JSON", language="json")

        # wiring
        run_case_tables_reload.click(ui_reload_cases_tables, inputs=[run_case_tables], outputs=[run_case_tables])
        run_use_btn.click(ui_use_cases_table_dropdown, inputs=[run_case_tables], outputs=[run_case_tables])

        run_provider.change(ui_toggle_base_url, inputs=[run_provider], outputs=[base_url2])
        run_model_reload.click(ui_reload_models_provider, inputs=[run_provider, base_url2, run_model], outputs=[run_model])
        run_provider.change(ui_reload_models_provider, inputs=[run_provider, base_url2, run_model], outputs=[run_model])
        base_url2.change(ui_reload_models_provider, inputs=[run_provider, base_url2, run_model], outputs=[run_model])

        run_cases_btn.click(
            ui_run_cases,
            inputs=[base_url2, run_model, system_prompt2, temperature2, timeout2, packs_csv, version, label_like, limit, write_artifacts, run_provider, run_api_key],
            outputs=[table_name, df_out, summary_out],
        )

    # =========================
    # Evaluate Cases
    # =========================

def build_tab_evaluate_cases():
    with gr.Tab("Evaluate Cases"):
        gr.Markdown("## Cases Table (select)")
        with gr.Row(equal_height=True):
            eval_case_tables = gr.Dropdown(
                label="Existing cases tables (public)",
                choices=ui_list_cases_tables(),
                value=None,
                scale=9,
            )
            eval_case_tables_reload = gr.Button("🔄", scale=1, min_width=0)
            eval_use_btn = gr.Button("Use selected", variant="primary", scale=2)

        gr.Markdown("## Provider & Model")
        with gr.Row(equal_height=True):
            eval_provider = gr.Dropdown(label="Provider", choices=["ollama", "openai"], value="ollama", scale=2)
            eval_api_key = gr.Textbox(label="API Key (OpenAI)", type="password", placeholder="sk-...", scale=4)
            eval_base_url = gr.Textbox(label="Ollama Base URL (Evaluator)", value=OLLAMA_BASE_URL, scale=6, visible=True)

        with gr.Row(equal_height=True):
            eval_model = gr.Dropdown(
                label="Evaluation Model",
                choices=list_ollama_models(OLLAMA_BASE_URL),
                value=OLLAMA_MODEL if OLLAMA_MODEL in list_ollama_models(OLLAMA_BASE_URL) else None,
                allow_custom_value=True,
                scale=10,
            )
            eval_model_reload = gr.Button("🔄", scale=2, min_width=0)

        with gr.Accordion("Evaluation settings", open=True):
            rubric = gr.Textbox(label="Evaluation rubric", value=DEFAULT_RUBRIC, lines=8)
            eval_system_prompt = gr.Textbox(label="Optional evaluator system prompt", lines=2)
            with gr.Row():
                eval_temperature = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=1, label="Temperature")
                eval_timeout = gr.Number(value=120, precision=0, label="Timeout (s)")
                eval_limit = gr.Number(value=None, label="Limit rows (optional)", precision=0)
                eval_sleep = gr.Number(value=0.15, label="Sleep between eval calls (s)")

        # explicit run-table picker (same as your original flow)
        eval_run_table = gr.Dropdown(
            label="Run table to evaluate (attacks)",
            choices=ui_list_cases_tables(),
            value=None,
        )
        eval_run_table_reload = gr.Button("🔄 Refresh run tables")

        eval_btn = gr.Button("Evaluate run table", variant="primary")
        eval_table_name = gr.Textbox(label="Evaluation table name", interactive=False)
        eval_df = gr.Dataframe(label="Evaluation preview", interactive=False)
        eval_summary = gr.Code(label="Evaluation summary", language="json")

        # wiring
        eval_case_tables_reload.click(ui_reload_cases_tables, inputs=[eval_case_tables], outputs=[eval_case_tables])
        eval_use_btn.click(ui_use_cases_table_dropdown, inputs=[eval_case_tables], outputs=[eval_case_tables])

        eval_provider.change(ui_toggle_base_url, inputs=[eval_provider], outputs=[eval_base_url])
        eval_model_reload.click(ui_reload_models_provider, inputs=[eval_provider, eval_base_url, eval_model], outputs=[eval_model])
        eval_provider.change(ui_reload_models_provider, inputs=[eval_provider, eval_base_url, eval_model], outputs=[eval_model])
        eval_run_table_reload.click(ui_reload_cases_tables, inputs=[eval_run_table], outputs=[eval_run_table])

        eval_btn.click(
            ui_evaluate_cases,
            inputs=[eval_run_table, eval_base_url, eval_model, rubric, eval_system_prompt, eval_temperature, eval_timeout, eval_limit, eval_sleep, eval_provider, eval_api_key],
            outputs=[eval_table_name, eval_df, eval_summary],
        )

    # =========================
    # Refinement
    # =========================

def build_tab_refinement():
    with gr.Tab("Refinement"):
        gr.Markdown("## Select Run Table")
        refine_run_tables = gr.Dropdown(
            label="Run tables (eval_attacks)",
            choices=ui_list_cases_tables(),
            value=None,
            scale=9,
        )
        refine_run_tables_reload = gr.Button("🔄", scale=1, min_width=0)
        refine_use_btn = gr.Button("Use selected", variant="primary", scale=2)

        gr.Markdown("## Embedding Provider & Model")
        with gr.Row(equal_height=True):
            refine_provider = gr.Dropdown(label="Provider", choices=["ollama", "openai"], value="ollama", scale=2)
            refine_api_key = gr.Textbox(label="API Key (OpenAI)", type="password", placeholder="sk-...", scale=4)
            refine_base_url = gr.Textbox(label="Ollama Base URL", value=OLLAMA_BASE_URL, scale=6, visible=True)

        with gr.Row(equal_height=True):
            refine_model = gr.Dropdown(
                label="Embedding Model",
                choices=list_ollama_models(OLLAMA_BASE_URL),
                value=None,
                allow_custom_value=True,
                scale=10,
            )
            refine_model_reload = gr.Button("🔄", scale=2, min_width=0)

        with gr.Accordion("Settings", open=True):
            fields = gr.CheckboxGroup(
                label="Fields to embed (reasoning source)",
                choices=[("label","label"), ("prompt","prompt"), ("output","output"), ("reason","reason")],
                value=["label", "prompt", "output", "reason"],
            )
            with gr.Row():
                refine_limit = gr.Number(value=500, precision=0, label="Max rows")
                auto_k = gr.Checkbox(label="Auto-K (Silhouette)", value=True)
            with gr.Row():
                k_pass = gr.Number(value=6, precision=0, label="K (Pass)", interactive=True)
                k_fail = gr.Number(value=6, precision=0, label="K (Fail)", interactive=True)

        # Filters must be defined BEFORE any `.click` that uses them
        with gr.Accordion("Filters", open=False):
            with gr.Row():
                side_filter = gr.Radio(choices=["both","pass","fail"], value="both", label="Side")
                label_filter = gr.Textbox(label="Label contains (substring, case-insensitive)", placeholder="e.g., json, jailbreak")
            with gr.Row():
                min_chars = gr.Number(value=0, precision=0, label="Min. composed text length")
                dedupe = gr.Checkbox(value=True, label="Dedupe identical texts")

        refine_btn = gr.Button("Analyze & Cluster", variant="primary")

        with gr.Row():
            pass_df = gr.Dataframe(label="PASS – Clusters", interactive=False)
            fail_df = gr.Dataframe(label="FAIL – Clusters", interactive=False)
        refine_report = gr.Markdown()

        with gr.Row(equal_height=True):
            summary_provider = gr.Dropdown(label="Provider", choices=["ollama", "openai"], value="ollama", scale=2)
            summary_api_key = gr.Textbox(label="API Key (OpenAI)", type="password", placeholder="sk-...", scale=4)
            summary_base_url = gr.Textbox(label="Ollama Base URL (Summary)", value=OLLAMA_BASE_URL, scale=6, visible=True)

        with gr.Row(equal_height=True):
            summary_model = gr.Dropdown(
                label="Summary LLM Model",
                choices=list_ollama_models(OLLAMA_BASE_URL),
                value=None,
                allow_custom_value=True,   # allows OpenAI models (free text)
                scale=10,
            )
            summary_model_reload = gr.Button("🔄", scale=2, min_width=0)

        # Show/hide base URL for provider, reload model list, etc.
        summary_provider.change(ui_toggle_base_url, inputs=[summary_provider], outputs=[summary_base_url])
        summary_model_reload.click(ui_reload_models_provider, inputs=[summary_provider, summary_base_url, summary_model], outputs=[summary_model])
        summary_provider.change(ui_reload_models_provider, inputs=[summary_provider, summary_base_url, summary_model], outputs=[summary_model])
        summary_base_url.change(ui_reload_models_provider, inputs=[summary_provider, summary_base_url, summary_model], outputs=[summary_model])

        # --- Summary actions & outputs (PDF export supported) ---
        sum_btn = gr.Button("Generate AI Summary (LLM)", variant="secondary")
        sum_md = gr.Markdown()
        sum_text_hidden = gr.Textbox(visible=False)
        export_summary_pdf_btn = gr.Button("Export Summary as PDF")
        summary_pdf_file = gr.File(label="Summary PDF", interactive=False)

        # Generate & export
        sum_btn.click(
            ui_ai_summary,
            inputs=[
                refine_run_tables,           # ref_table
                summary_provider,            # provider for SUMMARY (not embedding)
                summary_api_key,             # api key for SUMMARY (if OpenAI)
                summary_base_url,            # ollama base url (if ollama)
                refine_model,                # embedding model (fallback only)
                fields, refine_limit,
                side_filter, label_filter, min_chars, dedupe,
                summary_model,               # summary chat model
            ],
            outputs=[sum_md, sum_text_hidden],
        )
        export_summary_pdf_btn.click(ui_export_pdf, inputs=[sum_text_hidden], outputs=[summary_pdf_file])

        # Reload choices for the Summary LLM dropdown
        summary_model_reload.click(
            ui_reload_models_provider,
            inputs=[refine_provider, refine_base_url, summary_model],
            outputs=[summary_model],
        )
        refine_provider.change(
            ui_reload_models_provider,
            inputs=[refine_provider, refine_base_url, summary_model],
            outputs=[summary_model],
        )
        refine_base_url.change(
            ui_reload_models_provider,
            inputs=[refine_provider, refine_base_url, summary_model],
            outputs=[summary_model],
        )

        # Main analysis call (single binding)
        refine_btn.click(
            ui_refine_embed_and_cluster,
            inputs=[refine_run_tables, refine_provider, refine_api_key, refine_base_url, refine_model,
                    fields, refine_limit, auto_k, k_pass, k_fail,
                    side_filter, label_filter, min_chars, dedupe],
            outputs=[pass_df, fail_df, refine_report],
        )

        # wiring
        refine_run_tables_reload.click(ui_reload_cases_tables, inputs=[refine_run_tables], outputs=[refine_run_tables])
        refine_use_btn.click(ui_use_cases_table_dropdown, inputs=[refine_run_tables], outputs=[refine_run_tables])

        refine_provider.change(ui_toggle_base_url, inputs=[refine_provider], outputs=[refine_base_url])
        refine_model_reload.click(ui_reload_models_provider, inputs=[refine_provider, refine_base_url, refine_model], outputs=[refine_model])
        refine_provider.change(ui_reload_models_provider, inputs=[refine_provider, refine_base_url, refine_model], outputs=[refine_model])
        refine_base_url.change(ui_reload_models_provider, inputs=[refine_provider, refine_base_url, refine_model], outputs=[refine_model])

        # Auto-K controls (enable/disable K fields)
        def _toggle_k(auto_flag: bool):
            return (gr.update(interactive=not auto_flag),
                    gr.update(interactive=not auto_flag))
        auto_k.change(_toggle_k, inputs=[auto_k], outputs=[k_pass, k_fail])

    # =========================
    # Browse Run Table
    # =========================

def build_tab_evaluation_metrics():
    with gr.Tab("Evaluation Metrics"):
        gr.Markdown("## Select Evaluation Table")
        with gr.Row(equal_height=True):
            eval_tables_dd = gr.Dropdown(
                label="Evaluation tables (eval_attacks)",
                choices=[t for t in ui_list_cases_tables() if t.startswith("eval_")],
                value=None,
                scale=9,
            )
            eval_tables_reload = gr.Button("🔄", scale=1, min_width=0)

        # Cluster metrics inputs (reuse embedding controls for analysis)
        gr.Markdown("### Cluster Analysis Controls")
        with gr.Row(equal_height=True):
            cm_provider = gr.Dropdown(label="Provider", choices=["ollama", "openai"], value="ollama", scale=2)
            cm_api_key = gr.Textbox(label="API Key (OpenAI)", type="password", placeholder="sk-...", scale=4)
            cm_base_url = gr.Textbox(label="Ollama Base URL (Clusters)", value=OLLAMA_BASE_URL, scale=6, visible=True)

        with gr.Row(equal_height=True):
            cm_model = gr.Dropdown(
                label="Embedding Model",
                choices=list_ollama_models(OLLAMA_BASE_URL),
                value=None,
                allow_custom_value=True,
                scale=10,
            )
            cm_model_reload = gr.Button("🔄", scale=2, min_width=0)

        with gr.Accordion("Filters", open=False):
            with gr.Row():
                cm_side_filter = gr.Radio(choices=["both","pass","fail"], value="both", label="Side")
                cm_label_filter = gr.Textbox(label="Label contains (substring, case-insensitive)", placeholder="e.g., json, jailbreak")
            with gr.Row():
                cm_min_chars = gr.Number(value=0, precision=0, label="Min. composed text length")
                cm_dedupe = gr.Checkbox(value=True, label="Dedupe identical texts")

        metrics_limit = gr.Number(value=5000, precision=0, label="Max rows to load")
        metrics_run_btn = gr.Button("Compute Metrics", variant="primary")

        # --- Cluster metrics FIRST ---
        gr.Markdown("### Cluster Metrics")
        cluster_info = gr.Markdown("Runs a quick clustering over embedded texts (PASS/FAIL).")
        # Cleaner pies (with legend)
        pass_pie = gr.Plot(label="PASS clusters (pie)")
        fail_pie = gr.Plot(label="FAIL clusters (pie)")
        # Point cloud scatter with legends
        cluster_scatter = gr.Plot(label="Point Cloud (PCA 2D) — color = cluster, marker = side")

        # --- Evaluation overview (text + PDF export), no extra bar/line charts ---
        gr.Markdown("### Evaluation Overview")
        metrics_md = gr.Markdown()
        metrics_text_hidden = gr.Textbox(visible=False)
        export_metrics_pdf_btn = gr.Button("Export Metrics as PDF")
        metrics_pdf_file = gr.File(label="Metrics PDF", interactive=False)

        # Wiring: reload lists
        eval_tables_reload.click(
            lambda cur: gr.update(
                choices=[t for t in ui_list_cases_tables() if t.startswith("eval_")],
                value=cur if cur in [t for t in ui_list_cases_tables() if t.startswith("eval_")] else None
            ),
            inputs=[eval_tables_dd], outputs=[eval_tables_dd]
        )

        cm_provider.change(ui_toggle_base_url, inputs=[cm_provider], outputs=[cm_base_url])
        cm_model_reload.click(ui_reload_models_provider, inputs=[cm_provider, cm_base_url, cm_model], outputs=[cm_model])
        cm_provider.change(ui_reload_models_provider, inputs=[cm_provider, cm_base_url, cm_model], outputs=[cm_model])
        cm_base_url.change(ui_reload_models_provider, inputs=[cm_provider, cm_base_url, cm_model], outputs=[cm_model])

        # Main metrics callback (returns only 5 outputs now)
        def _compute_all_metrics(eval_table, limit,
                                prov, key, base, model,
                                side, labf, minc, dd):
            from .metrics import fetch_eval_df, compute_eval_metrics_from_df, compute_cluster_metrics_and_projection
            from .ui import _effective_base_url  # reuse helper
            be = _effective_base_url(prov, base)

            # Eval overview text
            try:
                df = fetch_eval_df(eval_table, limit)
                md, _counts, _pr, _trend = compute_eval_metrics_from_df(df)  # we only keep md
            except Exception as e:
                md = f"Error loading eval: {type(e).__name__}: {e}"

            # Cluster metrics
            try:
                p_counts, f_counts, proj = compute_cluster_metrics_and_projection(
                    ref_table=eval_table,
                    provider=prov, api_key=(key or None), base_url=be, embed_model=model,
                    fields=["label","prompt","output","reason"],
                    limit=limit, side_filter=side, label_filter=labf, min_chars=int(minc or 0), dedupe=bool(dd),
                )
            except Exception:
                p_counts, f_counts, proj = (None, None, None)

            # --- prettier plots (matplotlib) ---
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D

            def _shorten(s: str, n: int = 28) -> str:
                s = (s or "").strip()
                return s if len(s) <= n else (s[: n - 1] + "…")

            def _pie_fig(counts_df, title):
                fig, ax = plt.subplots(figsize=(5.2, 5.2))
                if counts_df is None or counts_df.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center")
                    ax.axis("off")
                    return fig

                labels = counts_df.get("legend", pd.Series(dtype=str)).tolist()
                labels = [(_shorten(lbl) if lbl else f"C{int(c)}") for lbl, c in zip(labels, counts_df["cluster"].tolist())]
                sizes = counts_df["count"].tolist()

                def _autopct_gen(vals):
                    total = sum(vals)
                    def _fmt(pct):
                        n = int(round(pct * total / 100.0))
                        return f"{pct:.0f}%\n(n={n})"
                    return _fmt

                wedges, _, autotexts = ax.pie(
                    sizes,
                    labels=None,  # legend instead
                    autopct=_autopct_gen(sizes),
                    startangle=90,
                    wedgeprops=dict(linewidth=0.8, edgecolor="white"),
                    textprops=dict(fontsize=9)
                )
                ax.axis("equal")
                ax.set_title(title)

                # Legend maps color → representative label
                ax.legend(
                    wedges, labels, title="Clusters",
                    loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=True
                )
                fig.tight_layout()
                return fig

            def _scatter_fig(df, pass_counts, fail_counts):
                fig, ax = plt.subplots(figsize=(7.5, 6.2))
                if df is None or df.empty:
                    ax.text(0.5, 0.5, "No projection available\n(install scikit-learn)", ha="center", va="center")
                    ax.set_axis_off()
                    return fig

                # Build legend labels per cluster_key from counts (prefer PASS/FAIL legends)
                legend_map = {}
                if pass_counts is not None and not pass_counts.empty:
                    for _, r in pass_counts.iterrows():
                        legend_map[f"PASS-{int(r['cluster'])}"] = _shorten(r.get("legend") or f"C{int(r['cluster'])}")
                if fail_counts is not None and not fail_counts.empty:
                    for _, r in fail_counts.iterrows():
                        legend_map[f"FAIL-{int(r['cluster'])}"] = _shorten(r.get("legend") or f"C{int(r['cluster'])}")

                handles = []
                labels = []

                # Plot grouped by cluster_key to keep colors consistent between sides
                for ck in sorted(df["cluster_key"].unique()):
                    sub = df[df["cluster_key"] == ck]
                    # PASS first: get color handle
                    sp = sub[sub["side"] == "pass"]
                    if not sp.empty:
                        sc_pass = ax.scatter(sp["x"], sp["y"], marker="o", s=28, alpha=0.85)
                        handles.append(sc_pass)
                        labels.append(legend_map.get(ck, ck))
                        # Use same color for FAIL
                        facecol = sc_pass.get_facecolor()
                        color = facecol[0] if len(facecol) else None
                        sf = sub[sub["side"] == "fail"]
                        if not sf.empty:
                            ax.scatter(sf["x"], sf["y"], marker="x", s=42, alpha=0.9, c=[color] if color is not None else None)
                    else:
                        # If no PASS points, create color from FAIL and still add a handle
                        sf = sub[sub["side"] == "fail"]
                        if not sf.empty:
                            sc_fail = ax.scatter(sf["x"], sf["y"], marker="x", s=42, alpha=0.9)
                            handles.append(sc_fail)
                            labels.append(legend_map.get(ck, ck))

                ax.set_title("Embedding space (PCA 2D)")
                ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
                ax.grid(True, alpha=0.3)

                # Legend 1: clusters (colors)
                leg1 = ax.legend(handles, labels, title="Clusters (repr. label)", loc="upper right", frameon=True)
                ax.add_artist(leg1)

                # Legend 2: sides (markers)
                side_handles = [
                    Line2D([0], [0], marker="o", linestyle="None", markersize=7, label="PASS"),
                    Line2D([0], [0], marker="x", linestyle="None", markersize=7, label="FAIL"),
                ]
                ax.legend(side_handles, [h.get_label() for h in side_handles],
                        title="Side", loc="lower right", frameon=True)

                fig.tight_layout()
                return fig

            fig_pass_pie = _pie_fig(p_counts, "PASS clusters")
            fig_fail_pie = _pie_fig(f_counts, "FAIL clusters")
            fig_scatter  = _scatter_fig(proj, p_counts, f_counts)


            return fig_pass_pie, fig_fail_pie, fig_scatter, md, md

        metrics_run_btn.click(
            _compute_all_metrics,
            inputs=[eval_tables_dd, metrics_limit,
                    cm_provider, cm_api_key, cm_base_url, cm_model,
                    cm_side_filter, cm_label_filter, cm_min_chars, cm_dedupe],
            outputs=[pass_pie, fail_pie, cluster_scatter, metrics_md, metrics_text_hidden],
        )

        export_metrics_pdf_btn.click(ui_export_pdf, inputs=[metrics_text_hidden], outputs=[metrics_pdf_file])


