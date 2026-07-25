"""Server-side file access for the file-aware tool variants.

Reading files on the MCP server instead of asking Claude to paste their
contents into a tool call keeps large files out of Claude's context window
entirely -- Claude only ever sees a path and the model's output, not the
source text. Every path is resolved and confined to ``settings.allowed_base_dir``
to prevent a malicious or buggy caller from reading/writing arbitrary files
on the host.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from .config import Settings

EXCLUDED_DIR_NAMES = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
}


class PathAccessError(ValueError):
    """Raised when a requested path falls outside the allowed base directory."""


def resolve_path(path: str, settings: Settings) -> Path:
    """Resolve ``path`` (absolute or relative) and confine it to the allowed dir.

    The base is realpath'd alongside the candidate. Comparing a resolved
    candidate against an unresolved base rejects every path whenever the
    allowed dir contains a symlink -- ``~/Developer`` pointing into iCloud
    Drive, for instance, where the candidate resolves to the real
    ``~/Library/Mobile Documents/...`` location and never appears to sit
    under the configured base.
    """
    base = Path(os.path.realpath(settings.allowed_base_dir))
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = Path(os.path.realpath(candidate))
    try:
        resolved.relative_to(base)
    except ValueError:
        raise PathAccessError(
            f"path '{path}' resolves outside the allowed directory "
            f"({settings.allowed_base_dir}); set OLLAMA_MCP_ALLOWED_DIR to widen it"
        ) from None
    return resolved


def read_file(path: str, settings: Settings) -> str:
    """Read a text file server-side, subject to the configured size cap."""
    resolved = resolve_path(path, settings)
    if not resolved.is_file():
        raise FileNotFoundError(f"'{path}' does not exist or is not a regular file")
    size = resolved.stat().st_size
    if size > settings.max_file_bytes:
        raise ValueError(
            f"'{path}' is {size} bytes, exceeding the {settings.max_file_bytes}-byte "
            "limit (set OLLAMA_MCP_MAX_FILE_BYTES to raise it)"
        )
    try:
        return resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"'{path}' does not look like a UTF-8 text file") from exc


def write_file(path: str, content: str, settings: Settings) -> None:
    resolved = resolve_path(path, settings)
    resolved.write_text(content, encoding="utf-8")


def resolve_code_input(
    code: str, file_path: str, settings: Settings
) -> tuple[str, str]:
    """Resolve the (code, file_path) pair every code-taking tool accepts.

    Exactly one of ``code`` or ``file_path`` must be provided. Returns
    ``(content, source_label)`` where ``source_label`` is used purely for
    display in the tool's response.
    """
    code = (code or "").strip()
    file_path = (file_path or "").strip()
    if code and file_path:
        raise ValueError("pass either 'code' or 'file_path', not both")
    if not code and not file_path:
        raise ValueError("pass either 'code' or 'file_path'")
    if file_path:
        return read_file(file_path, settings), file_path
    return code, "<inline>"


def glob_files(
    pattern: str, settings: Settings, root_dir: str = ""
) -> list[Path]:
    """Expand a glob pattern to a sorted, filtered list of files.

    Matching is confined to ``root_dir`` (default: the allowed base dir) and
    skips common junk directories and oversized files. Callers that feed this
    into sequential LLM calls (e.g. batch_refactor) are responsible for
    applying ``settings.max_batch_files`` themselves and reporting when the
    match set was truncated.
    """
    base = resolve_path(root_dir, settings) if root_dir else Path(
        settings.allowed_base_dir
    )
    if not base.is_dir():
        raise FileNotFoundError(f"root_dir '{root_dir or '.'}' is not a directory")

    matches: list[Path] = []
    hard_ceiling = settings.max_batch_files * 10
    for candidate in sorted(base.rglob("*")):
        if not candidate.is_file():
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in candidate.relative_to(base).parts[:-1]):
            continue
        rel = candidate.relative_to(base).as_posix()
        if not fnmatch.fnmatch(rel, pattern):
            continue
        if candidate.stat().st_size > settings.max_file_bytes:
            continue
        matches.append(candidate)
        if len(matches) >= hard_ceiling:
            break
    return matches
