"""Thin async client around the Ollama HTTP API.

Only the two endpoints this project needs are wrapped: ``/api/chat`` for
inference and ``/api/tags`` for health/model discovery. Errors are translated
into a small hierarchy so callers can distinguish "Ollama is unreachable"
from "Ollama responded with a problem" and react accordingly (see
``service.py`` for the graceful-fallback behavior this enables).
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from .config import Settings


class OllamaError(Exception):
    """Base class for all Ollama-related failures."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama host could not be reached at all."""


class OllamaTimeoutError(OllamaError):
    """Raised when Ollama did not respond within the configured timeout."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama responded but with an error status or bad payload."""


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(
                settings.timeout, connect=settings.connect_timeout
            ),
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request and return a normalized result dict."""
        target_model = model or self._settings.model
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": options or {"num_ctx": self._settings.num_ctx},
        }
        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"could not reach Ollama at {self._settings.base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama did not respond within {self._settings.timeout:.0f}s "
                f"(model={target_model})"
            ) from exc

        if response.status_code == 404:
            raise OllamaResponseError(
                f"model '{target_model}' was not found on the Ollama host at "
                f"{self._settings.base_url}. Run `ollama pull {target_model}` there, "
                "or set OLLAMA_MODEL to a model that is already pulled."
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OllamaResponseError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:500]}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaResponseError(
                f"Ollama returned a non-JSON response: {response.text[:500]}"
            ) from exc

        message = data.get("message") or {}
        content = message.get("content")
        if not content:
            raise OllamaResponseError(
                f"Ollama response for model '{target_model}' had no message content"
            )

        total_duration_ns = data.get("total_duration") or 0
        return {
            "content": content,
            "model": data.get("model", target_model),
            "total_duration_ms": round(total_duration_ns / 1_000_000, 1),
            "eval_count": data.get("eval_count"),
            "done_reason": data.get("done_reason"),
        }

    async def list_models(self) -> list[str]:
        response = await self._client.get("/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    async def health_check(self) -> dict[str, Any]:
        """Check reachability and whether the configured model is pulled."""
        start = time.monotonic()
        try:
            models = await self.list_models()
        except httpx.ConnectError as exc:
            return {
                "reachable": False,
                "base_url": self._settings.base_url,
                "error": f"connection failed: {exc}",
            }
        except httpx.TimeoutException as exc:
            return {
                "reachable": False,
                "base_url": self._settings.base_url,
                "error": f"timed out: {exc}",
            }
        except httpx.HTTPStatusError as exc:
            return {
                "reachable": False,
                "base_url": self._settings.base_url,
                "error": f"HTTP {exc.response.status_code} from /api/tags",
            }
        latency_ms = round((time.monotonic() - start) * 1000, 1)

        configured_model = self._settings.model
        model_available = configured_model in models or any(
            m.split(":")[0] == configured_model.split(":")[0] for m in models
        )
        return {
            "reachable": True,
            "base_url": self._settings.base_url,
            "configured_model": configured_model,
            "model_available": model_available,
            "available_models": models,
            "latency_ms": latency_ms,
        }

    async def aclose(self) -> None:
        await self._client.aclose()
