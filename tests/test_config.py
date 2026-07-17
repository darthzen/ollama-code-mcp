from ollama_code_mcp.config import coerce_think_style, load_settings


def test_defaults(monkeypatch):
    for key in [
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_TIMEOUT",
        "MCP_TRANSPORT",
        "OLLAMA_MCP_ALLOWED_DIR",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("OLLAMA_THINK_STYLE", raising=False)
    settings = load_settings()
    assert settings.base_url == "http://localhost:11434"
    assert settings.timeout == 900.0
    assert settings.transport == "stdio"
    assert settings.think_style == "qwen"


def test_think_style_parsing(monkeypatch):
    monkeypatch.setenv("OLLAMA_THINK_STYLE", "none")
    assert load_settings().think_style == "none"
    monkeypatch.setenv("OLLAMA_THINK_STYLE", "NONE")   # case-insensitive
    assert load_settings().think_style == "none"


def test_invalid_think_style_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_THINK_STYLE", "banana")
    assert load_settings().think_style == "qwen"


def test_coerce_think_style():
    # blank/None → the given default (lets a per-call override fall back to config)
    assert coerce_think_style("", "none") == "none"
    assert coerce_think_style(None, "qwen") == "qwen"
    # known values pass through, case-insensitively
    assert coerce_think_style("none") == "none"
    assert coerce_think_style("QWEN") == "qwen"
    # unknown → default
    assert coerce_think_style("banana", "none") == "none"


def test_base_url_scheme_is_added_when_missing(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "ollama.ash4d.com:11434")
    settings = load_settings()
    assert settings.base_url == "http://ollama.ash4d.com:11434"


def test_base_url_trailing_slash_is_stripped(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.lan:11434/")
    settings = load_settings()
    assert settings.base_url == "http://ollama.lan:11434"


def test_invalid_numeric_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_TIMEOUT", "not-a-number")
    settings = load_settings()
    assert settings.timeout == 900.0


def test_bool_env_parsing(monkeypatch):
    monkeypatch.setenv("OLLAMA_MCP_DEFAULT_THINK", "false")
    assert load_settings().default_think is False
    monkeypatch.setenv("OLLAMA_MCP_DEFAULT_THINK", "true")
    assert load_settings().default_think is True
