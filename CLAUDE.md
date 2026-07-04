# Routing guidance: local Ollama vs. cloud Claude

This project exposes coding tools backed by a local/LAN Ollama instance
(Qwen3) via MCP. Use this file to decide when to delegate a task to those
tools versus doing it yourself.

## Delegate to `ollama-code-mcp` when the task is...

- **Mechanical or repetitive.** Writing a first-draft test suite, generating
  boilerplate (CRUD handlers, DTOs, config parsing), or applying the same
  mechanical edit across many files (`batch_refactor`).
- **Context-window-heavy but low-ambiguity.** Explaining an unfamiliar file,
  reviewing a large diff for obvious issues (`code_review_diff`), or
  reviewing/refactoring a big file you'd otherwise have to paste in full.
  Use `file_path` / `diff_file` / `glob_pattern` instead of pasting content
  so the file never enters your own context window.
- **A useful first pass before your own review.** Run `review_code` or
  `code_review_diff` to surface candidate issues, then apply your own
  judgment to what it finds rather than trusting it blindly -- local models
  are meaningfully weaker than you at subtle correctness reasoning.
- **Not security- or architecture-sensitive.** Straightforward fixes,
  explanations, and tests for code where being wrong is cheap to catch and
  correct.

## Keep in the cloud (do it yourself) when the task is...

- **Architecturally significant** -- API design, cross-module refactors that
  require holding the whole system in mind, anything where a subtly wrong
  answer is expensive to unwind.
- **Security-sensitive** -- auth, crypto, permission checks, anything
  touching secrets or user data. Don't delegate generation or fixes here;
  you can delegate a *review pass* as a second opinion, but verify its
  findings yourself.
- **Ambiguous or under-specified.** If the task needs clarifying questions
  or multi-turn negotiation with the user, handle it yourself -- the Ollama
  tools are single-shot, not conversational.
- **Small and fast.** A one-line fix or a quick rename isn't worth a
  network round-trip to another host; just make the edit.
- **Already failing.** If a tool call reports Ollama is unreachable, timed
  out, or the model isn't pulled, don't retry it -- handle the task
  yourself. Call `ollama_status` once to confirm before writing off local
  delegation for the rest of the session.

## Using the tools well

- Pass `file_path` (or `diff_file`, or a `glob_pattern` for `batch_refactor`)
  instead of inline `code`/`diff` whenever the content already exists on
  disk -- this is the entire point of the file-aware variants.
- `refactor_code` and the single-file tools never write to disk; they just
  return text for you to apply. Only `batch_refactor` can write files, and
  only with `dry_run=False` -- always look at the `dry_run=True` diff output
  first, especially for anything non-trivial.
- Leave `think=True` (the default for review/refactor/fix/test tools) for
  anything where getting it right matters; use `think=False` for quick,
  low-stakes generations or explanations where latency matters more.
- Requests can take up to 15 minutes (`OLLAMA_TIMEOUT`) on modest hardware.
  Don't assume a slow response means something is broken -- but if you hit
  the timeout, consider whether the task should be split into smaller
  pieces before retrying.
- `batch_refactor` caps how many files it processes per call
  (`OLLAMA_MCP_MAX_BATCH_FILES`, default 20). If a glob matches more than
  that, the response tells you it was truncated -- narrow the pattern or
  make additional calls rather than assuming everything was covered.
