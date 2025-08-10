# Thin wrapper that re-exports build_app for external imports.
# Thin wrapper re-exporting build_app from ui_app.
from .ui_app import build_app

def _effective_base_url(provider: str, base_url: str | None):
    return base_url if (provider or "").lower() == "ollama" else None