"""MCP server entry point: wires FastMCP tool definitions to CodeService.

Each tool below is a thin wrapper -- argument parsing and formatting live in
``service.py`` so they can be tested without an MCP session. Docstrings here
double as the tool descriptions Claude sees, so they explain not just what
each tool does but when file_path/diff_file inputs save context window.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .config import load_settings
from .ollama_client import OllamaClient
from .service import CodeService

settings = load_settings()
client = OllamaClient(settings)
service = CodeService(client, settings)

mcp = FastMCP(
    "ollama-code-mcp",
    host=settings.host,
    port=settings.port,
)


@mcp.tool()
async def generate_code(
    instruction: str,
    language: str = "",
    context_file: str = "",
    think: bool = True,
    model: str = "",
    think_style: str = "",
) -> str:
    """Generate new code from a natural-language instruction using the local Ollama model.

    Pass `context_file` (a server-side path) instead of pasting an existing
    file's contents to give the model style/API context without spending
    your own context window on it. Set `think=False` for faster, simpler
    generations; leave `think=True` for anything non-trivial.

    `model` overrides the server's configured Ollama model for this one call
    (e.g. a lighter model for a throwaway task); empty = the configured
    default. Pair it with `think_style` (`"none"` for non-Qwen models like
    DeepSeek/Llama, `"qwen"` for Qwen) so the think switch matches the model.
    """
    return await service.generate_code(
        instruction, language, context_file, think, model, think_style
    )


@mcp.tool()
async def review_code(
    code: str = "",
    file_path: str = "",
    focus: str = "",
    think: bool = True,
    model: str = "",
    think_style: str = "",
) -> str:
    """Review code for correctness bugs, security issues, and simplification opportunities.

    Provide exactly one of `code` (inline text) or `file_path` (a server-side
    path read directly by this tool, saving your context window). `focus`
    can narrow the review, e.g. "concurrency" or "input validation".

    `model`/`think_style` override the configured model and think dialect for
    this call (see `generate_code`).
    """
    return await service.review_code(code, file_path, focus, think, model, think_style)


@mcp.tool()
async def refactor_code(
    instruction: str,
    code: str = "",
    file_path: str = "",
    think: bool = True,
    model: str = "",
    think_style: str = "",
) -> str:
    """Refactor code according to an instruction, preserving external behavior.

    Provide exactly one of `code` or `file_path`. Returns the full refactored
    code plus a summary of changes; this tool does not write files itself --
    use `batch_refactor` if you want changes applied to disk.

    `model`/`think_style` override the configured model and think dialect for
    this call (see `generate_code`).
    """
    return await service.refactor_code(
        instruction, code, file_path, think, model, think_style
    )


@mcp.tool()
async def fix_code(
    code: str = "",
    file_path: str = "",
    error_message: str = "",
    think: bool = True,
    model: str = "",
    think_style: str = "",
) -> str:
    """Diagnose and fix a bug in code, given an optional error message or symptom.

    Provide exactly one of `code` or `file_path`. Include `error_message`
    (a stack trace, test failure, or description of wrong behavior) whenever
    you have one -- it substantially improves fix quality.

    `model`/`think_style` override the configured model and think dialect for
    this call (see `generate_code`).
    """
    return await service.fix_code(
        code, file_path, error_message, think, model, think_style
    )


@mcp.tool()
async def write_tests(
    code: str = "",
    file_path: str = "",
    framework: str = "",
    think: bool = True,
    model: str = "",
    think_style: str = "",
) -> str:
    """Write tests covering the golden path and realistic edge cases for given code.

    Provide exactly one of `code` or `file_path`. Set `framework` (e.g.
    "pytest", "jest") to match your project's conventions; otherwise the
    model infers one from the code's language.

    `model`/`think_style` override the configured model and think dialect for
    this call (see `generate_code`).
    """
    return await service.write_tests(code, file_path, framework, think, model, think_style)


@mcp.tool()
async def explain_code(
    code: str = "",
    file_path: str = "",
    think: bool = False,
    model: str = "",
    think_style: str = "",
) -> str:
    """Explain what code does, including control/data flow and non-obvious behavior.

    Provide exactly one of `code` or `file_path`. Explanations are usually
    fast enough that `think=False` (the default) is sufficient.

    `model`/`think_style` override the configured model and think dialect for
    this call (see `generate_code`).
    """
    return await service.explain_code(code, file_path, think, model, think_style)


@mcp.tool()
async def code_review_diff(
    diff: str = "",
    diff_file: str = "",
    context: str = "",
    think: bool = True,
    model: str = "",
    think_style: str = "",
) -> str:
    """Review a git diff (e.g. `git diff main...HEAD`) as if for a pull request.

    Provide exactly one of `diff` (inline diff text) or `diff_file` (a
    server-side path to a saved diff). `context` can carry the PR
    description or any background the model should know.

    `model`/`think_style` override the configured model and think dialect for
    this call (see `generate_code`).
    """
    return await service.code_review_diff(
        diff, diff_file, context, think, model, think_style
    )


@mcp.tool()
async def batch_refactor(
    glob_pattern: str,
    instruction: str,
    root_dir: str = "",
    dry_run: bool = True,
    think: bool = True,
    model: str = "",
    think_style: str = "",
) -> str:
    """Apply a refactor instruction to every file matching a glob pattern, sequentially.

    `glob_pattern` matches relative paths under `root_dir` (default: the
    server's allowed base directory), e.g. `"src/**/*.py"`. Files are
    processed one at a time against the local model. With `dry_run=True`
    (the default) nothing is written -- you get a unified diff per file to
    review first; set `dry_run=False` to write accepted changes to disk.
    Large match sets are capped (see OLLAMA_MCP_MAX_BATCH_FILES) to avoid
    runaway sequential runs.

    `model`/`think_style` override the configured model and think dialect for
    every file in this run (see `generate_code`).
    """
    return await service.batch_refactor(
        glob_pattern, instruction, root_dir, dry_run, think, model, think_style
    )


@mcp.tool()
async def ollama_status() -> str:
    """Check connectivity to the configured Ollama host and report available models.

    Call this first if other tools are failing, or proactively before a
    batch of delegated work, to confirm the local model is reachable and
    pulled before relying on it.
    """
    return await service.status()


def main() -> None:
    mcp.run(transport=settings.transport)


if __name__ == "__main__":
    main()
