import pytest

from ollama_code_mcp.config import Settings
from ollama_code_mcp.ollama_client import OllamaConnectionError
from ollama_code_mcp.service import CodeService


def make_settings(base_dir, **overrides) -> Settings:
    defaults = dict(
        base_url="http://ollama.lan:11434",
        model="qwen3:32b",
        timeout=900.0,
        connect_timeout=10.0,
        num_ctx=8192,
        allowed_base_dir=str(base_dir),
        max_file_bytes=1_000_000,
        max_batch_files=20,
        default_think=True,
        transport="stdio",
        host="0.0.0.0",
        port=8765,
    )
    defaults.update(overrides)
    return Settings(**defaults)


class FakeClient:
    """Stands in for OllamaClient with scripted responses, no network involved."""

    def __init__(self, response_content: str = "```\nfixed = True\n```", fail: Exception | None = None):
        self.response_content = response_content
        self.fail = fail
        self.calls: list[list[dict]] = []
        self.health = {"reachable": True, "base_url": "http://ollama.lan:11434", "configured_model": "qwen3:32b", "model_available": True, "available_models": ["qwen3:32b"], "latency_ms": 12.3}

    async def chat(self, messages, model=None, options=None):
        self.calls.append(messages)
        if self.fail:
            raise self.fail
        return {
            "content": self.response_content,
            "model": "qwen3:32b",
            "total_duration_ms": 123.0,
            "eval_count": 10,
        }

    async def health_check(self):
        return self.health


@pytest.mark.asyncio
async def test_generate_code_requires_instruction(tmp_path):
    service = CodeService(FakeClient(), make_settings(tmp_path))
    with pytest.raises(ValueError):
        await service.generate_code("")


@pytest.mark.asyncio
async def test_generate_code_success_includes_header(tmp_path):
    client = FakeClient(response_content="```py\nprint('hi')\n```")
    service = CodeService(client, make_settings(tmp_path))
    result = await service.generate_code("write a hello world script")
    assert "generate_code" in result
    assert "print('hi')" in result
    assert client.calls[0][1]["content"].endswith("/think")


@pytest.mark.asyncio
async def test_review_code_falls_back_gracefully_when_ollama_down(tmp_path):
    client = FakeClient(fail=OllamaConnectionError("refused"))
    service = CodeService(client, make_settings(tmp_path))
    result = await service.review_code(code="x = 1")
    assert "unavailable" in result
    assert "refused" in result


@pytest.mark.asyncio
async def test_review_code_reads_file_when_given(tmp_path):
    (tmp_path / "mod.py").write_text("def f(): return 1")
    client = FakeClient()
    service = CodeService(client, make_settings(tmp_path))
    await service.review_code(file_path="mod.py")
    assert "def f(): return 1" in client.calls[0][1]["content"]


@pytest.mark.asyncio
async def test_code_review_diff_rejects_both_inputs(tmp_path):
    service = CodeService(FakeClient(), make_settings(tmp_path))
    with pytest.raises(ValueError):
        await service.code_review_diff(diff="+x", diff_file="a.diff")


@pytest.mark.asyncio
async def test_code_review_diff_rejects_neither_input(tmp_path):
    service = CodeService(FakeClient(), make_settings(tmp_path))
    with pytest.raises(ValueError):
        await service.code_review_diff()


@pytest.mark.asyncio
async def test_batch_refactor_dry_run_does_not_write(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    client = FakeClient(response_content="```py\nx = 2\n```")
    service = CodeService(client, make_settings(tmp_path))
    result = await service.batch_refactor("*.py", "rename x", dry_run=True)
    assert "WOULD CHANGE" in result
    assert (tmp_path / "a.py").read_text() == "x = 1\n"


@pytest.mark.asyncio
async def test_batch_refactor_writes_when_not_dry_run(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    client = FakeClient(response_content="```py\nx = 2\n```")
    service = CodeService(client, make_settings(tmp_path))
    result = await service.batch_refactor("*.py", "rename x", dry_run=False)
    assert "CHANGED (written)" in result
    assert (tmp_path / "a.py").read_text().strip() == "x = 2"


@pytest.mark.asyncio
async def test_batch_refactor_no_matches(tmp_path):
    service = CodeService(FakeClient(), make_settings(tmp_path))
    result = await service.batch_refactor("*.rs", "rename x")
    assert "no files" in result


@pytest.mark.asyncio
async def test_status_reachable(tmp_path):
    service = CodeService(FakeClient(), make_settings(tmp_path))
    result = await service.status()
    assert "OK at" in result
    assert "qwen3:32b" in result


@pytest.mark.asyncio
async def test_status_unreachable(tmp_path):
    client = FakeClient()
    client.health = {"reachable": False, "base_url": "http://ollama.lan:11434", "error": "connection failed"}
    service = CodeService(client, make_settings(tmp_path))
    result = await service.status()
    assert "UNREACHABLE" in result
