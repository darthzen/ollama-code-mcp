import httpx
import pytest
import respx

from ollama_code_mcp.config import Settings
from ollama_code_mcp.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)

BASE_URL = "http://ollama.lan:11434"


def make_settings(**overrides) -> Settings:
    defaults = dict(
        base_url=BASE_URL,
        model="qwen3:32b",
        timeout=5.0,
        connect_timeout=2.0,
        num_ctx=8192,
        allowed_base_dir="/tmp",
        max_file_bytes=1_000_000,
        max_batch_files=20,
        default_think=True,
        transport="stdio",
        host="0.0.0.0",
        port=8765,
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.mark.asyncio
@respx.mock
async def test_chat_success():
    respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"content": "hello world"},
                "model": "qwen3:32b",
                "total_duration": 2_000_000,
                "eval_count": 42,
            },
        )
    )
    client = OllamaClient(make_settings())
    result = await client.chat([{"role": "user", "content": "hi"}])
    assert result["content"] == "hello world"
    assert result["total_duration_ms"] == 2.0
    assert result["eval_count"] == 42
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_chat_connection_error():
    respx.post(f"{BASE_URL}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    client = OllamaClient(make_settings())
    with pytest.raises(OllamaConnectionError):
        await client.chat([{"role": "user", "content": "hi"}])
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_chat_timeout_error():
    respx.post(f"{BASE_URL}/api/chat").mock(side_effect=httpx.ReadTimeout("too slow"))
    client = OllamaClient(make_settings())
    with pytest.raises(OllamaTimeoutError):
        await client.chat([{"role": "user", "content": "hi"}])
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_chat_model_not_found():
    respx.post(f"{BASE_URL}/api/chat").mock(return_value=httpx.Response(404))
    client = OllamaClient(make_settings())
    with pytest.raises(OllamaResponseError, match="not found"):
        await client.chat([{"role": "user", "content": "hi"}])
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_health_check_reachable_and_model_available():
    respx.get(f"{BASE_URL}/api/tags").mock(
        return_value=httpx.Response(
            200, json={"models": [{"name": "qwen3:32b"}, {"name": "llama3.1:8b"}]}
        )
    )
    client = OllamaClient(make_settings())
    info = await client.health_check()
    assert info["reachable"] is True
    assert info["model_available"] is True
    assert "llama3.1:8b" in info["available_models"]
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_health_check_unreachable():
    respx.get(f"{BASE_URL}/api/tags").mock(side_effect=httpx.ConnectError("refused"))
    client = OllamaClient(make_settings())
    info = await client.health_check()
    assert info["reachable"] is False
    assert "connection failed" in info["error"]
    await client.aclose()
