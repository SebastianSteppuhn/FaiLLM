import time
import requests

def chat(prompt, base_url="http://localhost:11434", model="llama3.1:8b",
         system=None, temperature=0.2, timeout=120):
    base = base_url.rstrip("/")
    start = time.perf_counter()
    payload_chat = {
        "model": model, "stream": False, "options": {"temperature": temperature},
        "messages": (([{"role":"system","content":system}] if system else [])
                     + [{"role":"user","content":prompt}]),
    }
    try:
        r = requests.post(f"{base}/api/chat", json=payload_chat, timeout=timeout)
        endpoint = "chat"
        if r.status_code == 404:
            endpoint = "generate"
            payload_gen = {
                "model": model, "prompt": (f"[SYSTEM]\n{system}\n[/SYSTEM]\n{prompt}" if system else prompt),
                "stream": False, "options": {"temperature": temperature},
            }
            r = requests.post(f"{base}/api/generate", json=payload_gen, timeout=timeout)

        latency_ms = int((time.perf_counter() - start) * 1000)
        status = r.status_code
        if status >= 400:
            return {"ok": False, "text": "", "latency_ms": latency_ms, "status": status,
                    "error": r.text, "endpoint": endpoint}

        data = r.json()
        text = (data.get("message", {}) or {}).get("content") or data.get("response", "") or ""
        return {"ok": True, "text": text, "latency_ms": latency_ms, "status": status,
                "error": None, "endpoint": endpoint}
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {"ok": False, "text": "", "latency_ms": latency_ms, "status": 0,
                "error": str(e), "endpoint": "chat"}
