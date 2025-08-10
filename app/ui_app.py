# App entry point that composes all tabs into a Gradio Blocks app.
# App entry point. Builds the full Gradio UI by composing tab builders.
from .ui_base import *
from .ui_tabs import build_tab_generate_cases, build_tab_run_cases, build_tab_evaluate_cases, build_tab_refinement, build_tab_evaluation_metrics

def build_app():
    with gr.Blocks(title="FaiLLM", theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown("# FaiLLM\nGenerate adversarial cases, run them against models, and analyze results.")
        with gr.Tabs():
            build_tab_generate_cases()
            build_tab_run_cases()
            build_tab_evaluate_cases()
            build_tab_refinement()
            build_tab_evaluation_metrics()
        gr.Markdown("Made by Sebastian Steppuhn during Hack-Nation Hackathon")
    return demo
