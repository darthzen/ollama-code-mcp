"""Qwen3-tuned prompt templates and response post-processing.

Qwen3 models support a soft "thinking" toggle: appending the literal token
``/think`` or ``/no_think`` to the end of the user turn switches extended
chain-of-thought reasoning on or off for that turn (this is how Ollama's
Qwen3 template exposes the feature, since ``/api/chat`` does not have a
first-class ``enable_thinking`` field). When thinking is enabled, Qwen3
wraps its reasoning in a leading ``<think>...</think>`` block before the
actual answer -- ``split_thinking`` separates the two so tools can surface
the reasoning as optional context without making callers parse it out
themselves.
"""

from __future__ import annotations

import re

_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

GENERATE_SYSTEM = (
    "You are an expert software engineer generating new code from a "
    "specification. Write correct, idiomatic code that matches the style of "
    "any reference context you are given. Do not invent requirements beyond "
    "what was asked. Reply with the code in a fenced code block, followed by "
    "a short explanation only if something non-obvious needs calling out."
)

REVIEW_SYSTEM = (
    "You are an expert code reviewer. Review the given code for correctness "
    "bugs, security issues, and real simplification/efficiency opportunities. "
    "Do not nitpick style or naming unless it causes an actual problem. For "
    "each finding, state the concrete failure scenario (what input or "
    "condition triggers it), not just a vague concern. If the code has no "
    "significant issues, say so briefly instead of inventing filler feedback."
)

REFACTOR_SYSTEM = (
    "You are an expert software engineer refactoring existing code according "
    "to an instruction. Preserve external behavior unless told otherwise. "
    "Reply with the full refactored code in a single fenced code block, "
    "followed by a short list of what changed and why."
)

FIX_SYSTEM = (
    "You are an expert software engineer fixing a bug in the given code. "
    "Identify the root cause before proposing a change -- do not paper over "
    "symptoms. Reply with the corrected full code in a fenced code block, "
    "followed by a short explanation of the root cause and the fix."
)

TEST_SYSTEM = (
    "You are an expert software engineer writing tests for the given code. "
    "Cover the golden path plus realistic edge cases and failure modes. "
    "Match the idioms of the specified test framework (or infer a sensible "
    "one from the code's language if none is given). Reply with the test "
    "code in a fenced code block."
)

EXPLAIN_SYSTEM = (
    "You are an expert software engineer explaining code to another "
    "engineer. Describe what the code does, how control and data flow "
    "through it, and call out any non-obvious behavior, edge cases, or "
    "implicit assumptions. Do not simply restate the code line by line."
)

DIFF_REVIEW_SYSTEM = (
    "You are an expert code reviewer reviewing a git diff as part of a pull "
    "request. Focus on correctness bugs, security issues, and meaningful "
    "simplification opportunities introduced or exposed by this diff. Refer "
    "to specific hunks using their file path and line context. Do not "
    "comment on unchanged code unless the diff makes it newly relevant. If "
    "the diff looks safe to merge, say so briefly."
)


def build_messages(
    system: str, user_content: str, think: bool, style: str = "qwen"
) -> list[dict[str, str]]:
    """Build a chat message list, optionally with a think-mode switch.

    ``style`` selects how the think toggle is expressed:

    - ``"qwen"`` (default): append Qwen3's literal ``/think`` / ``/no_think``
      token — the only way Ollama's Qwen3 template exposes the toggle.
    - ``"none"``: append nothing. Use for models that don't understand the
      Qwen switch (DeepSeek-R1 distills, Llama, …), where a stray ``/no_think``
      is prompt noise that can mislead a lighter model into hallucinating. The
      model's own default reasoning applies; ``split_thinking`` still strips any
      ``<think>`` block it emits.
    """
    content = user_content.strip()
    if style == "qwen":
        content = f"{content}\n\n{'/think' if think else '/no_think'}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def split_thinking(text: str) -> tuple[str | None, str]:
    """Split a Qwen3 response into (reasoning, final_answer).

    Returns ``(None, text)`` when no ``<think>`` block is present, e.g. when
    ``/no_think`` was used or the model chose to skip reasoning.
    """
    match = _THINK_BLOCK_RE.search(text)
    if not match:
        return None, text.strip()
    thinking = match.group(1).strip()
    answer = (text[: match.start()] + text[match.end() :]).strip()
    return (thinking or None), answer


_CODE_BLOCK_RE = re.compile(r"```(?:[\w+-]*)\n(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str:
    """Return the contents of the largest fenced code block in ``text``.

    Falls back to the stripped full text when no fenced block is found, so
    callers always get something usable even if the model didn't format its
    answer as expected.
    """
    blocks = _CODE_BLOCK_RE.findall(text)
    if not blocks:
        return text.strip()
    return max(blocks, key=len).strip()
