from __future__ import annotations
import time
import requests
from typing import Any, Dict, Optional

class OllamaClient:
    def __init__(self, base_url: str, model: str, temperature: float=0.2, timeout: int=120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = float(temperature)
        self.timeout = int(timeout)

    def chat(self, prompt: str, *, system: Optional[str] = None) -> Dict[str, Any]:
        start = time.perf_counter()
        payload_chat = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": (([{"role":"system","content":system}] if system else []) + [{"role":"user","content":prompt}]),
        }
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload_chat, timeout=self.timeout)
            endpoint = "chat"
            if r.status_code == 404:
                endpoint = "generate"
                payload_gen = {
                    "model": self.model,
                    "prompt": (f"[SYSTEM]\n{system}\n[/SYSTEM]\n{prompt}" if system else prompt),
                    "stream": False,
                    "options": {"temperature": self.temperature},
                }
                r = requests.post(f"{self.base_url}/api/generate", json=payload_gen, timeout=self.timeout)
            latency_ms = int((time.perf_counter() - start) * 1000)
            status = r.status_code
            if status >= 400:
                return {"ok": False, "text": "", "latency_ms": latency_ms, "status": status, "error": r.text, "endpoint": endpoint}
            data = r.json()
            text = (data.get("message", {}) or {}).get("content") or data.get("response", "") or ""
            return {"ok": True, "text": text, "latency_ms": latency_ms, "status": status, "error": None, "endpoint": endpoint}
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return {"ok": False, "text": "", "latency_ms": latency_ms, "status": 0, "error": f"{type(e).__name__}: {e}", "endpoint": "chat"}
