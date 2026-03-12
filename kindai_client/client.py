"""
KindAI Client — shared client for all KindPath projects.

Usage:
    from kindai_client import KindAIClient

    ai = KindAIClient()
    response = ai.ask("What is the current ν for SPY?")
    print(response)

    # Streaming
    for token in ai.stream("Explain the trade tier logic"):
        print(token, end="", flush=True)

Auto-selects backend:
  1. kindai inference server (localhost:7862) — GGUF with doctrine + logit processor
  2. Ollama (localhost:11434 / llama3.2) — free local fallback

Install into any KindPath project:
    pip install -e /Users/sam/kindai
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

import httpx

logger = logging.getLogger(__name__)

INFERENCE_SERVER = "http://localhost:7862/v1"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"
CLAUDE_MODEL = "claude-sonnet-4-6"
INFERENCE_MODEL = "kindai-doctrine"
TIMEOUT = 120.0
_DOCTRINE_PATH_DEFAULT = "/Users/sam/kindai/doctrine.md"
if not Path(_DOCTRINE_PATH_DEFAULT).exists():
    _DOCTRINE_PATH_DEFAULT = "doctrine.md" # container fallback
# Max chars of doctrine to inline into Ollama /api/generate prompt.
# /api/chat with full doctrine (~3500 tokens) hangs on Intel CPU Ollama.
_OLLAMA_DOCTRINE_CHARS = 1200


def _load_doctrine(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _claude_available() -> bool:
    import os
    key = os.getenv("ANTHROPIC_API_KEY", "")
    return bool(key and key.startswith("sk-ant-"))


def _inference_server_alive() -> bool:
    try:
        r = httpx.get(f"{INFERENCE_SERVER}/health", timeout=2.0)
        data = r.json()
        return r.status_code == 200 and data.get("model_loaded", False)
    except Exception:
        return False


def _ollama_alive() -> bool:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


class KindAIClient:
    """
    Unified KindAI client. All KindPath projects use this to talk to kindai.

    Parameters
    ----------
    doctrine_path : str | None
        Path to doctrine.md. None = use default. "" = no doctrine injection.
    system : str | None
        Additional system context appended after doctrine (e.g. project-specific role).
    """

    def __init__(
        self,
        doctrine_path: str | None = _DOCTRINE_PATH_DEFAULT,
        system: str | None = None,
    ):
        self.doctrine = _load_doctrine(doctrine_path) if doctrine_path else ""
        self.extra_system = system or ""
        self._backend: str | None = None

    @property
    def backend(self) -> str:
        if self._backend is None:
            if _claude_available():
                self._backend = "claude"
                logger.info("KindAIClient: Claude API (%s)", CLAUDE_MODEL)
            elif _inference_server_alive():
                self._backend = "inference"
                logger.info("KindAIClient: inference server (localhost:7862)")
            elif _ollama_alive():
                self._backend = "ollama"
                logger.info("KindAIClient: Ollama fallback (%s)", OLLAMA_MODEL)
            else:
                self._backend = "unavailable"
                logger.warning("KindAIClient: no backend available")
        return self._backend

    def _system_prompt(self) -> str:
        return "\n\n".join(p for p in [self.doctrine, self.extra_system] if p)

    def ask(self, prompt: str, system: str | None = None) -> str:
        """Single-turn question. Returns full response text."""
        sys = system or self._system_prompt()
        messages = []
        if sys:
            messages.append({"role": "system", "content": sys})
        messages.append({"role": "user", "content": prompt})
        return self._complete(messages)

    def chat(self, messages: list[dict]) -> str:
        """Multi-turn. messages = [{"role": ..., "content": ...}]."""
        sys = self._system_prompt()
        full = []
        if sys and (not messages or messages[0].get("role") != "system"):
            full.append({"role": "system", "content": sys})
        full.extend(messages)
        return self._complete(full)

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """Stream response tokens for a single prompt."""
        sys = system or self._system_prompt()
        messages = []
        if sys:
            messages.append({"role": "system", "content": sys})
        messages.append({"role": "user", "content": prompt})
        yield from self._stream(messages)

    def reset_backend(self) -> None:
        """Re-detect backend on next call (e.g. after starting the inference server)."""
        self._backend = None

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _complete(self, messages: list[dict]) -> str:
        b = self.backend
        if b == "claude":
            return self._claude_complete(messages)
        elif b == "inference":
            return self._inference_complete(messages)
        elif b == "ollama":
            return self._ollama_complete(messages)
        raise RuntimeError(
            "No KindAI backend available.\n"
            "  Add ANTHROPIC_API_KEY to .env (recommended)\n"
            "  Or start Ollama: brew services start ollama"
        )

    def _stream(self, messages: list[dict]) -> Iterator[str]:
        b = self.backend
        if b == "claude":
            yield from self._claude_stream(messages)
        elif b == "inference":
            yield from self._inference_stream(messages)
        elif b == "ollama":
            yield from self._ollama_stream(messages)
        else:
            raise RuntimeError("No KindAI backend available.")

    def _claude_complete(self, messages: list[dict]) -> str:
        import os
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        sys_content = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_msgs = [m for m in messages if m["role"] != "system"]
        kwargs = {"model": CLAUDE_MODEL, "max_tokens": 1024, "messages": user_msgs}
        if sys_content:
            kwargs["system"] = sys_content
        response = client.messages.create(**kwargs)
        return response.content[0].text

    def _claude_stream(self, messages: list[dict]) -> Iterator[str]:
        import os
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        sys_content = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_msgs = [m for m in messages if m["role"] != "system"]
        kwargs = {"model": CLAUDE_MODEL, "max_tokens": 1024, "messages": user_msgs}
        if sys_content:
            kwargs["system"] = sys_content
        with client.messages.stream(**kwargs) as stream:
            yield from stream.text_stream

    def _inference_complete(self, messages: list[dict]) -> str:
        payload = {"model": INFERENCE_MODEL, "messages": messages, "stream": False}
        with httpx.Client(timeout=TIMEOUT) as client:
            r = client.post(f"{INFERENCE_SERVER}/chat/completions", json=payload)
            r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _inference_stream(self, messages: list[dict]) -> Iterator[str]:
        payload = {"model": INFERENCE_MODEL, "messages": messages, "stream": True}
        with httpx.Client(timeout=TIMEOUT) as client:
            with client.stream("POST", f"{INFERENCE_SERVER}/chat/completions", json=payload) as r:
                for line in r.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError):
                            pass

    def _ollama_prompt(self, messages: list[dict]) -> str:
        """
        Convert messages to a flat prompt for /api/generate.
        Inlines a doctrine excerpt instead of passing as system message
        to avoid context overflow on Intel CPU Ollama.
        """
        doctrine_excerpt = self.doctrine[:_OLLAMA_DOCTRINE_CHARS] if self.doctrine else ""
        parts = []
        if doctrine_excerpt:
            parts.append(f"KindPath doctrine: {doctrine_excerpt}")
        if self.extra_system:
            parts.append(f"Context: {self.extra_system}")
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                continue  # already inlined above
            elif role == "user":
                parts.append(f"Q: {content}")
            elif role == "assistant":
                parts.append(f"A: {content}")
        parts.append("A:")
        return "\n\n".join(parts)

    def _ollama_complete(self, messages: list[dict]) -> str:
        prompt = self._ollama_prompt(messages)
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": 2048, "num_predict": 300, "temperature": 0.7},
        }
        with httpx.Client(timeout=TIMEOUT) as client:
            r = client.post(OLLAMA_GENERATE_URL, json=payload)
            r.raise_for_status()
        return r.json().get("response", "").strip()

    def _ollama_stream(self, messages: list[dict]) -> Iterator[str]:
        prompt = self._ollama_prompt(messages)
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {"num_ctx": 2048, "num_predict": 300, "temperature": 0.7},
        }
        with httpx.Client(timeout=TIMEOUT) as client:
            with client.stream("POST", OLLAMA_GENERATE_URL, json=payload) as r:
                for line in r.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("response", "")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            pass
