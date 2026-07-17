import pytest

from ollama_code_mcp.config import Settings
from ollama_code_mcp.file_utils import (
    PathAccessError,
    glob_files,
    read_file,
    resolve_code_input,
    resolve_path,
    write_file,
)


def make_settings(base_dir, **overrides) -> Settings:
    defaults = dict(
        base_url="http://localhost:11434",
        model="qwen3:32b",
        timeout=900.0,
        connect_timeout=10.0,
        num_ctx=8192,
        allowed_base_dir=str(base_dir),
        max_file_bytes=1_000_000,
        max_batch_files=20,
        default_think=True,
        think_style="qwen",
        transport="stdio",
        host="0.0.0.0",
        port=8765,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_resolve_path_within_base_dir(tmp_path):
    settings = make_settings(tmp_path)
    (tmp_path / "foo.py").write_text("x = 1")
    resolved = resolve_path("foo.py", settings)
    assert resolved == (tmp_path / "foo.py").resolve()


def test_resolve_path_rejects_escape(tmp_path):
    settings = make_settings(tmp_path / "project")
    (tmp_path / "project").mkdir()
    with pytest.raises(PathAccessError):
        resolve_path("../secrets.txt", settings)


def test_read_file_enforces_size_cap(tmp_path):
    settings = make_settings(tmp_path, max_file_bytes=10)
    (tmp_path / "big.txt").write_text("x" * 100)
    with pytest.raises(ValueError):
        read_file("big.txt", settings)


def test_read_file_missing_raises(tmp_path):
    settings = make_settings(tmp_path)
    with pytest.raises(FileNotFoundError):
        read_file("missing.py", settings)


def test_write_then_read_roundtrip(tmp_path):
    settings = make_settings(tmp_path)
    write_file("out.py", "print('hi')", settings)
    assert read_file("out.py", settings) == "print('hi')"


def test_resolve_code_input_requires_exactly_one(tmp_path):
    settings = make_settings(tmp_path)
    with pytest.raises(ValueError):
        resolve_code_input("", "", settings)
    with pytest.raises(ValueError):
        resolve_code_input("x = 1", "file.py", settings)


def test_resolve_code_input_reads_file(tmp_path):
    settings = make_settings(tmp_path)
    (tmp_path / "file.py").write_text("x = 1")
    content, label = resolve_code_input("", "file.py", settings)
    assert content == "x = 1"
    assert label == "file.py"


def test_resolve_code_input_inline(tmp_path):
    settings = make_settings(tmp_path)
    content, label = resolve_code_input("x = 1", "", settings)
    assert content == "x = 1"
    assert label == "<inline>"


def test_glob_files_matches_and_excludes_junk_dirs(tmp_path):
    settings = make_settings(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("a")
    (tmp_path / "src" / "b.py").write_text("b")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "c.py").write_text("c")

    matches = glob_files("src/*.py", settings)
    names = {m.name for m in matches}
    assert names == {"a.py", "b.py"}


def test_glob_files_respects_size_filter(tmp_path):
    settings = make_settings(tmp_path, max_file_bytes=5)
    (tmp_path / "small.py").write_text("hi")
    (tmp_path / "large.py").write_text("x" * 100)
    matches = glob_files("*.py", settings)
    assert [m.name for m in matches] == ["small.py"]
