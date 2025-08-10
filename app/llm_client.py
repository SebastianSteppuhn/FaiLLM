# HTTP client that talks to OpenAI/Ollama depending on provider.

from __future__ import annotations
import os
import time
from typing import Any, Dict, Optional, List
import requests


class LLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        *,
        temperature: float = 0.2,
        timeout: int = 120,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.provider = (provider or "ollama").lower()
        self.model = model
        self.temperature = float(temperature)
        self.timeout = int(timeout)
        self.base_url = (base_url or "").strip() or None
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.reasoning_effort = reasoning_effort                   




    def chat(self, prompt: str, *, system: Optional[str] = None) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            if self.provider == "ollama":
                text = self._chat_ollama(prompt=prompt, system=system)
            elif self.provider == "openai":
                text = self._chat_openai(prompt=prompt, system=system)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            latency_ms = int((time.perf_counter() - start) * 1000)
            return {
                "ok": True,
                "text": text or "",
                "latency_ms": latency_ms,
                "status": 200,
                "error": None,
                "endpoint": self.provider,
            }
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return {
                "ok": False,
                "text": "",
                "latency_ms": latency_ms,
                "status": 0,
                "error": f"{type(e).__name__}: {e}",
                "endpoint": self.provider,
            }




    def _chat_ollama(self, *, prompt: str, system: Optional[str]) -> str:
        import requests

        base = (self.base_url or "http://localhost:11434").rstrip("/")


        try:
            payload = {
                "model": self.model,
                "stream": False,
                "options": {"temperature": self.temperature},
                "messages": (
                    ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": prompt}]
                ),
            }
            r = requests.post(f"{base}/api/chat", json=payload, timeout=self.timeout)
            if r.status_code == 200:
                data = r.json() or {}
                msg = data.get("message") or {}
                return (msg.get("content") or "").strip()

            if r.status_code not in (404, 405):
                self._raise_ollama(r)
        except requests.RequestException:

            pass


        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        if system:
            payload["system"] = system

        r = requests.post(f"{base}/api/generate", json=payload, timeout=self.timeout)
        if r.status_code != 200:
            self._raise_ollama(r)
        data = r.json() or {}
        return (data.get("response") or "").strip()

    @staticmethod
    def _raise_ollama(r):
        try:
            j = r.json()
        except Exception:
            j = None
        detail = None
        if isinstance(j, dict):

            detail = j.get("error") or j.get("message") or j.get("detail")
        raise RuntimeError(f"Ollama error {r.status_code}: {detail or r.text[:500]}")



    def _chat_openai(self, *, prompt: str, system: Optional[str]) -> str:
        """
        Use the official OpenAI API (api.openai.com) only.
        1) Try Responses API.
        2) Fall back to Chat Completions.
        """
        from openai import OpenAI

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})


        client = OpenAI(api_key=self.api_key)
        try:
            client = client.with_options(timeout=self.timeout)
        except Exception:
            pass


        try:
            inputs = [{"role": m["role"], "content": m["content"]} for m in messages]
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "input": inputs,
                "temperature": self.temperature,
            }
            if self.reasoning_effort:
                kwargs["reasoning"] = {"effort": self.reasoning_effort}

            resp = client.responses.create(**kwargs)
            text = getattr(resp, "output_text", None)
            if text:
                return text.strip()

            try:
                parts = []
                for item in getattr(resp, "output", []) or []:
                    if isinstance(item, dict):
                        parts.append(item.get("content", "") or "")
                text = "\n".join(p for p in parts if p).strip()
                if text:
                    return text
            except Exception:
                pass
            raise RuntimeError("Responses API returned no text")
        except Exception:

            try:
                cc = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                choice = (cc.choices or [])[0]
                content = getattr(getattr(choice, "message", None), "content", None)
                if not content and isinstance(choice, dict):
                    content = ((choice.get("message") or {}).get("content"))
                return (content or "").strip()
            except Exception as e2:
                raise RuntimeError(f"OpenAI call failed (Responses & ChatCompletions): {e2}") from e2


    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Erzeuge Embeddings für gegebene Texte.
        - provider='openai': offizielle Embeddings API
        - provider='ollama': POST {base}/api/embeddings pro Text
        """
        if not texts:
            return []
        if self.provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            try:
                client = client.with_options(timeout=self.timeout)
            except Exception:
                pass


            resp = client.embeddings.create(model=self.model, input=texts)

            return [item.embedding for item in getattr(resp, "data", [])]
        elif self.provider == "ollama":
            base = (self.base_url or "http://localhost:11434").rstrip("/")
            out: List[List[float]] = []
            for t in texts:
                r = requests.post(
                    f"{base}/api/embeddings",
                    json={"model": self.model, "prompt": t},
                    timeout=self.timeout,
                )
                if r.status_code != 200:
                    try:
                        j = r.json()
                    except Exception:
                        j = None
                    detail = None
                    if isinstance(j, dict):
                        detail = j.get("error") or j.get("message") or j.get("detail")
                    raise RuntimeError(f"Ollama embeddings error {r.status_code}: {detail or r.text[:500]}")
                data = r.json() or {}
                emb = data.get("embedding")
                if not emb:
                    raise RuntimeError("Ollama embeddings returned no 'embedding'.")
                out.append(emb)
            return out
        else:
            raise ValueError(f"Unknown provider for embeddings: {self.provider}")
