from ollama_code_mcp.prompts import (
    build_messages,
    extract_code_block,
    split_thinking,
)


def test_build_messages_appends_think_switch():
    messages = build_messages("sys", "do the thing", think=True)
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1]["content"].endswith("/think")


def test_build_messages_appends_no_think_switch():
    messages = build_messages("sys", "do the thing", think=False)
    assert messages[1]["content"].endswith("/no_think")


def test_build_messages_qwen_style_is_default():
    messages = build_messages("sys", "do the thing", think=True, style="qwen")
    assert messages[1]["content"].endswith("/think")


def test_build_messages_none_style_omits_switch():
    for think in (True, False):
        messages = build_messages("sys", "do the thing", think=think, style="none")
        assert messages[1]["content"] == "do the thing"
        assert "/think" not in messages[1]["content"]
        assert "/no_think" not in messages[1]["content"]


def test_split_thinking_extracts_reasoning_block():
    text = "<think>step by step reasoning</think>final answer here"
    thinking, answer = split_thinking(text)
    assert thinking == "step by step reasoning"
    assert answer == "final answer here"


def test_split_thinking_handles_no_think_block():
    thinking, answer = split_thinking("just the answer")
    assert thinking is None
    assert answer == "just the answer"


def test_extract_code_block_picks_largest_fenced_block():
    text = (
        "Here is a short snippet:\n```py\nx = 1\n```\n"
        "And the full solution:\n```py\ndef f():\n    return 42\n```\n"
    )
    code = extract_code_block(text)
    assert "def f():" in code
    assert "x = 1" not in code


def test_extract_code_block_falls_back_to_full_text():
    assert extract_code_block("no fences here") == "no fences here"
