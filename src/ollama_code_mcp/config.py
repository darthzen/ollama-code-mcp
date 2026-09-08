"""Environment-driven configuration for the ollama-code-mcp server."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.8:27b-mtp-q8-precise"
DEFAULT_TIMEOUT_SECONDS = 900.0  # 15 minutes: large-context refactors on a single GPU can be slow
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_NUM_CTX = 8192
DEFAULT_THINK_STYLE = "qwen"      # how the think toggle is expressed in the prompt
THINK_STYLES = ("qwen", "none")  # "qwen": /think|/no_think switch; "none": append nothing
DEFAULT_MAX_FILE_BYTES = 1_000_000  # 1 MB per file read server-side
DEFAULT_MAX_BATCH_FILES = 20
DEFAULT_TRANSPORT = "stdio"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in _TRUE_VALUES


def coerce_think_style(value: str | None, default: str = DEFAULT_THINK_STYLE) -> str:
    """Coerce an arbitrary think-style string to a known value.

    Empty/blank or unrecognized input falls back to ``default``. Used both for
    the ``OLLAMA_THINK_STYLE`` env default and for a per-call ``think_style``
    override, so a caller can pick the right dialect (``"none"`` for a
    DeepSeek/Llama model, ``"qwen"`` for a Qwen model) alongside a per-call
    ``model`` without restarting the server."""
    if not value or not value.strip():
        return default
    style = value.strip().lower()
    return style if style in THINK_STYLES else default


def _think_style() -> str:
    """OLLAMA_THINK_STYLE, coerced to a known value (unknown → default).

    Lets non-Qwen models (DeepSeek-R1 distills, Llama, …) run without the
    Qwen-only ``/think`` / ``/no_think`` switch, which is just prompt noise to
    them and can mislead a lighter model."""
    return coerce_think_style(os.environ.get("OLLAMA_THINK_STYLE"), DEFAULT_THINK_STYLE)


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for a single server process."""

    base_url: str
    model: str
    timeout: float
    connect_timeout: float
    num_ctx: int
    allowed_base_dir: str
    max_file_bytes: int
    max_batch_files: int
    default_think: bool
    think_style: str
    transport: str
    host: str
    port: int


def load_settings() -> Settings:
    """Read configuration from the environment, falling back to sane defaults.

    OLLAMA_BASE_URL must point at wherever Ollama is actually listening -- this is
    commonly a LAN address (e.g. a k3s LoadBalancer IP) rather than localhost, since
    the model runs on a dedicated GPU host.
    """
    base_url = _env_str("OLLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        base_url = f"http://{base_url}"

    allowed_base_dir = os.path.realpath(
        os.environ.get("OLLAMA_MCP_ALLOWED_DIR", os.getcwd())
    )

    return Settings(
        base_url=base_url,
        model=_env_str("OLLAMA_MODEL", DEFAULT_MODEL),
        timeout=_env_float("OLLAMA_TIMEOUT", DEFAULT_TIMEOUT_SECONDS),
        connect_timeout=_env_float(
            "OLLAMA_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT_SECONDS
        ),
        num_ctx=_env_int("OLLAMA_NUM_CTX", DEFAULT_NUM_CTX),
        allowed_base_dir=allowed_base_dir,
        max_file_bytes=_env_int("OLLAMA_MCP_MAX_FILE_BYTES", DEFAULT_MAX_FILE_BYTES),
        max_batch_files=_env_int(
            "OLLAMA_MCP_MAX_BATCH_FILES", DEFAULT_MAX_BATCH_FILES
        ),
        default_think=_env_bool("OLLAMA_MCP_DEFAULT_THINK", True),
        think_style=_think_style(),
        transport=_env_str("MCP_TRANSPORT", DEFAULT_TRANSPORT),
        host=_env_str("MCP_HOST", DEFAULT_HOST),
        port=_env_int("MCP_PORT", DEFAULT_PORT),
    )
