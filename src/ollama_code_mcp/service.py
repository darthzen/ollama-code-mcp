"""Business logic for every coding tool, independent of the MCP transport layer.

Keeping this logic out of the ``@mcp.tool()`` decorated functions in
``server.py`` means it can be unit tested directly (no MCP client/session
needed) and reused if additional transports or entry points are added later.

Every method here follows the same fallback contract: if Ollama cannot be
reached, times out, or errors, the method returns a plain string describing
the problem rather than raising -- so a caller like Claude Code sees a
normal tool result and can decide to handle the task itself instead of
retrying. Only usage errors (bad arguments, path traversal, missing files)
raise exceptions, since those indicate the caller should fix its request.
"""

from __future__ import annotations

import difflib
from typing import Any

from .config import Settings
from .file_utils import glob_files, read_file, resolve_code_input, write_file
from .ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from .prompts import (
    DIFF_REVIEW_SYSTEM,
    EXPLAIN_SYSTEM,
    FIX_SYSTEM,
    GENERATE_SYSTEM,
    REFACTOR_SYSTEM,
    REVIEW_SYSTEM,
    TEST_SYSTEM,
    build_messages,
    extract_code_block,
    split_thinking,
)

_OLLAMA_ERRORS = (OllamaConnectionError, OllamaTimeoutError, OllamaResponseError)


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _format_success(tool_name: str, result: dict[str, Any], thinking: str | None, answer: str) -> str:
    stats = f"{result['total_duration_ms']:.0f} ms"
    if result.get("eval_count"):
        stats += f", {result['eval_count']} tokens"
    header = f"[{tool_name}] via {result['model']} ({stats})"
    parts = [header, answer]
    if thinking:
        parts.append(f"--- model reasoning ---\n{thinking}")
    return "\n\n".join(parts)


def _format_unavailable(tool_name: str, base_url: str, reason: str) -> str:
    return (
        f"[{tool_name}] local Ollama at {base_url} is unavailable: {reason}\n"
        "Falling back: handle this task yourself instead of retrying this tool."
    )


class CodeService:
    def __init__(self, client: OllamaClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    def _think(self, think: bool | None) -> bool:
        return self._settings.default_think if think is None else think

    async def _run_chat(
        self, tool_name: str, system: str, user_content: str, think: bool
    ) -> str:
        messages = build_messages(system, user_content, think)
        try:
            result = await self._client.chat(messages)
        except _OLLAMA_ERRORS as exc:
            return _format_unavailable(tool_name, self._settings.base_url, str(exc))
        thinking, answer = split_thinking(result["content"])
        return _format_success(tool_name, result, thinking, answer)

    # -- core coding tools ------------------------------------------------

    async def generate_code(
        self,
        instruction: str,
        language: str = "",
        context_file: str = "",
        think: bool | None = None,
    ) -> str:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        parts = []
        if language.strip():
            parts.append(f"Target language: {language.strip()}")
        if context_file.strip():
            context_code = read_file(context_file, self._settings)
            parts.append(
                f"Existing file for style/context ({context_file}):\n"
                f"```\n{context_code}\n```"
            )
        parts.append(f"Task:\n{instruction.strip()}")
        return await self._run_chat(
            "generate_code", GENERATE_SYSTEM, "\n\n".join(parts), self._think(think)
        )

    async def review_code(
        self, code: str = "", file_path: str = "", focus: str = "", think: bool | None = None
    ) -> str:
        content, label = resolve_code_input(code, file_path, self._settings)
        parts = [f"Code to review (source: {label}):\n```\n{content}\n```"]
        if focus.strip():
            parts.append(f"Focus areas: {focus.strip()}")
        return await self._run_chat(
            "review_code", REVIEW_SYSTEM, "\n\n".join(parts), self._think(think)
        )

    async def refactor_code(
        self,
        instruction: str,
        code: str = "",
        file_path: str = "",
        think: bool | None = None,
    ) -> str:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        content, label = resolve_code_input(code, file_path, self._settings)
        user_content = (
            f"Code to refactor (source: {label}):\n```\n{content}\n```\n\n"
            f"Instruction: {instruction.strip()}"
        )
        return await self._run_chat(
            "refactor_code", REFACTOR_SYSTEM, user_content, self._think(think)
        )

    async def fix_code(
        self,
        code: str = "",
        file_path: str = "",
        error_message: str = "",
        think: bool | None = None,
    ) -> str:
        content, label = resolve_code_input(code, file_path, self._settings)
        parts = [f"Code with a bug (source: {label}):\n```\n{content}\n```"]
        if error_message.strip():
            parts.append(f"Observed error/symptom:\n{error_message.strip()}")
        return await self._run_chat(
            "fix_code", FIX_SYSTEM, "\n\n".join(parts), self._think(think)
        )

    async def write_tests(
        self,
        code: str = "",
        file_path: str = "",
        framework: str = "",
        think: bool | None = None,
    ) -> str:
        content, label = resolve_code_input(code, file_path, self._settings)
        parts = [f"Code to test (source: {label}):\n```\n{content}\n```"]
        if framework.strip():
            parts.append(f"Test framework: {framework.strip()}")
        return await self._run_chat(
            "write_tests", TEST_SYSTEM, "\n\n".join(parts), self._think(think)
        )

    async def explain_code(
        self, code: str = "", file_path: str = "", think: bool | None = None
    ) -> str:
        content, label = resolve_code_input(code, file_path, self._settings)
        user_content = f"Code to explain (source: {label}):\n```\n{content}\n```"
        return await self._run_chat(
            "explain_code", EXPLAIN_SYSTEM, user_content, self._think(think)
        )

    # -- diff and batch tools ----------------------------------------------

    async def code_review_diff(
        self,
        diff: str = "",
        diff_file: str = "",
        context: str = "",
        think: bool | None = None,
    ) -> str:
        diff = (diff or "").strip()
        diff_file = (diff_file or "").strip()
        if diff and diff_file:
            raise ValueError("pass either 'diff' or 'diff_file', not both")
        if not diff and not diff_file:
            raise ValueError("pass either 'diff' or 'diff_file'")
        if diff_file:
            diff = read_file(diff_file, self._settings)
        if not diff.strip():
            raise ValueError("diff is empty")

        parts = [f"Git diff to review:\n```diff\n{diff.strip()}\n```"]
        if context.strip():
            parts.append(f"PR context: {context.strip()}")
        return await self._run_chat(
            "code_review_diff", DIFF_REVIEW_SYSTEM, "\n\n".join(parts), self._think(think)
        )

    async def batch_refactor(
        self,
        glob_pattern: str,
        instruction: str,
        root_dir: str = "",
        dry_run: bool = True,
        think: bool | None = None,
    ) -> str:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        if not glob_pattern.strip():
            raise ValueError("glob_pattern must not be empty")

        all_matches = glob_files(glob_pattern, self._settings, root_dir)
        if not all_matches:
            return (
                f"[batch_refactor] no files under '{root_dir or '.'}' "
                f"matched pattern '{glob_pattern}'"
            )

        max_files = self._settings.max_batch_files
        files = all_matches[:max_files]
        truncated = len(all_matches) > max_files

        lines = [
            f"[batch_refactor] pattern='{glob_pattern}' dry_run={dry_run} "
            f"matched={len(all_matches)} processing={len(files)}"
        ]
        if truncated:
            lines.append(
                f"  (truncated to the first {max_files} matches; raise "
                "OLLAMA_MCP_MAX_BATCH_FILES to process more per call)"
            )

        think_resolved = self._think(think)
        for path in files:
            label = str(path)
            try:
                original = read_file(label, self._settings)
            except (FileNotFoundError, ValueError) as exc:
                lines.append(f"  - {label}: SKIPPED ({exc})")
                continue

            user_content = (
                f"Code to refactor (source: {label}):\n```\n{original}\n```\n\n"
                f"Instruction: {instruction.strip()}"
            )
            messages = build_messages(REFACTOR_SYSTEM, user_content, think_resolved)
            try:
                result = await self._client.chat(messages)
            except _OLLAMA_ERRORS as exc:
                lines.append(f"  - {label}: OLLAMA ERROR ({exc})")
                continue

            _, answer = split_thinking(result["content"])
            new_code = extract_code_block(answer)
            if new_code.strip() == original.strip():
                lines.append(f"  - {label}: no change")
                continue

            diff_preview = "\n".join(
                difflib.unified_diff(
                    original.splitlines(),
                    new_code.splitlines(),
                    fromfile=label,
                    tofile=label,
                    lineterm="",
                    n=2,
                )
            )
            if dry_run:
                lines.append(f"  - {label}: WOULD CHANGE\n{_indent(diff_preview)}")
            else:
                write_file(label, new_code, self._settings)
                lines.append(f"  - {label}: CHANGED (written)\n{_indent(diff_preview)}")

        return "\n".join(lines)

    # -- operational --------------------------------------------------------

    async def status(self) -> str:
        info = await self._client.health_check()
        if not info["reachable"]:
            return (
                f"[ollama_status] UNREACHABLE at {info['base_url']}: {info['error']}\n"
                "Coding tools will fail until this is fixed; fall back to handling "
                "tasks directly in the meantime."
            )
        model_line = (
            f"model '{info['configured_model']}' is available"
            if info["model_available"]
            else f"model '{info['configured_model']}' is NOT pulled on this host "
            f"(run `ollama pull {info['configured_model']}` there)"
        )
        models = ", ".join(info["available_models"]) or "(none)"
        return (
            f"[ollama_status] OK at {info['base_url']} ({info['latency_ms']:.0f} ms)\n"
            f"{model_line}\n"
            f"available models: {models}"
        )
