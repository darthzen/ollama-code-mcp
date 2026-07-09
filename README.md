# ollama-code-mcp

An [MCP](https://modelcontextprotocol.io) server that lets Claude Code delegate
coding tasks to a local (or LAN) [Ollama](https://ollama.com) instance running
a Qwen3 model. Point it at a GPU box on your network -- a Tesla V100 running
`qwen3-coder:30b`, for example -- and Claude Code can hand off boilerplate
generation, test writing, diff review, and batch refactors to it instead of
spending cloud tokens and context window on them.

See [`CLAUDE.md`](./CLAUDE.md) for the routing guidance Claude Code reads to
decide what to delegate versus what to keep in the cloud.

## Why

- **Saves Claude's context window.** The file-aware tool variants take a
  `file_path` (or `diff_file`, or a `glob_pattern` for batches) and read the
  content server-side, so Claude never has to paste large files into a tool
  call just to hand them off.
- **Uses idle GPU capacity.** If you already run Ollama on a home GPU box or
  in a k3s cluster, this turns that capacity into a first-class Claude Code
  tool instead of a chat window you have to copy-paste into.
- **Fails safe.** If Ollama is unreachable, times out, or the model isn't
  pulled, tools return a clear, non-fatal message telling Claude to just
  handle the task itself rather than getting stuck retrying.

## Tools

| Tool | Purpose | Inputs |
|---|---|---|
| `generate_code` | Generate new code from an instruction | `instruction`, `language`, `context_file` |
| `review_code` | Review code for bugs, security issues, simplifications | `code` or `file_path`, `focus` |
| `refactor_code` | Refactor code per an instruction, behavior-preserving | `instruction`, `code` or `file_path` |
| `fix_code` | Diagnose and fix a bug | `code` or `file_path`, `error_message` |
| `write_tests` | Write tests for given code | `code` or `file_path`, `framework` |
| `explain_code` | Explain what code does | `code` or `file_path` |
| `code_review_diff` | Review a git diff, PR-review style | `diff` or `diff_file`, `context` |
| `batch_refactor` | Apply an instruction across files matching a glob, sequentially | `glob_pattern`, `instruction`, `root_dir`, `dry_run` |
| `ollama_status` | Health check: reachability, configured model, available models | -- |

Every tool (except `ollama_status`) accepts a `think: bool` parameter. This
toggles Qwen3's extended reasoning by appending `/think` or `/no_think` to
the prompt (see [Think mode](#think-mode) below). All coding tools default to
`think=True`; `explain_code` defaults to `False` since explanations are
usually fast enough without it.

`code` / `file_path` (and `diff` / `diff_file`) pairs are mutually exclusive
-- pass exactly one. File paths are resolved and confined to
`OLLAMA_MCP_ALLOWED_DIR` (see [Configuration](#configuration)); attempts to
read or write outside it are rejected.

`refactor_code` returns the refactored code as text -- it does not touch
disk. `batch_refactor` is the one tool that can write files, and only when
called with `dry_run=False`; by default it returns a unified diff per file
so you can review before applying.

## Installation

Requires Python 3.10+.

```bash
git clone git@github.com:darthzen/ollama-code-mcp.git
cd ollama-code-mcp
pip install -e .
```

Or with `uv`:

```bash
uv pip install -e .
```

### Register with Claude Code

Add to your Claude Code MCP config (`claude mcp add` or the `mcpServers`
block in your settings), pointing `OLLAMA_BASE_URL` at wherever Ollama
actually listens -- commonly a LAN address, not localhost, since the model
runs on a dedicated GPU host:

```json
{
  "mcpServers": {
    "ollama-code": {
      "command": "ollama-code-mcp",
      "env": {
        "OLLAMA_BASE_URL": "http://ollama.ash4d.com:11434",
        "OLLAMA_MODEL": "qwen3-coder:30b",
        "OLLAMA_MCP_ALLOWED_DIR": "/Users/you/code"
      }
    }
  }
}
```

Or run it straight from the repo without installing:

```json
{
  "mcpServers": {
    "ollama-code": {
      "command": "/path/to/ollama-code-mcp/.venv/bin/python",
      "args": ["-m", "ollama_code_mcp.server"],
      "env": { "OLLAMA_BASE_URL": "http://ollama.ash4d.com:11434" }
    }
  }
}
```

`OLLAMA_MCP_ALLOWED_DIR` should be set to the project root (or a parent of
every project) you want file-aware tools to be able to read/write. It
defaults to the server's current working directory.

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where Ollama listens. LAN addresses and bare `host:port` (scheme added automatically) are supported. |
| `OLLAMA_MODEL` | `qwen3-coder` | Model tag to use, as shown by `ollama list` on the target host. |
| `OLLAMA_TIMEOUT` | `900` | Request timeout in seconds. Defaults to 15 minutes to allow large generations/refactors on modest hardware. |
| `OLLAMA_CONNECT_TIMEOUT` | `10` | TCP connect timeout in seconds. |
| `OLLAMA_NUM_CTX` | `8192` | Context window passed to Ollama's `options.num_ctx`. |
| `OLLAMA_MCP_ALLOWED_DIR` | server CWD | Base directory that file-aware tools are confined to. |
| `OLLAMA_MCP_MAX_FILE_BYTES` | `1000000` | Per-file size cap for server-side reads. |
| `OLLAMA_MCP_MAX_BATCH_FILES` | `20` | Max files processed per `batch_refactor` call. |
| `OLLAMA_MCP_DEFAULT_THINK` | `true` | Default value for each tool's `think` parameter when the caller omits it. |
| `MCP_TRANSPORT` | `stdio` | `stdio` (spawned locally by Claude Code), `sse`, or `streamable-http` (for remote/k8s deployment). |
| `MCP_HOST` | `0.0.0.0` | Bind host for `sse`/`streamable-http` transports. |
| `MCP_PORT` | `8765` | Bind port for `sse`/`streamable-http` transports. |

## Think mode

Qwen3 exposes a soft toggle for its extended chain-of-thought reasoning:
appending `/think` or `/no_think` to the end of a prompt turns it on or off
for that turn. This server does that automatically based on each tool's
`think` parameter, and separates the model's `<think>...</think>` block from
its final answer in the response, so you get:

```
[review_code] via qwen3-coder (4213 ms, 812 tokens)

<the actual review>

--- model reasoning ---
<the model's chain of thought, if think=True>
```

Use `think=True` (the default for most tools) for review, refactor, fix, and
test-writing tasks where reasoning quality matters. Use `think=False` for
quick, low-stakes generations or explanations where latency matters more.

## Running standalone

```bash
OLLAMA_BASE_URL=http://ollama.ash4d.com:11434 ollama-code-mcp
```

By default this speaks MCP over stdio, which is what Claude Code expects
when it spawns the process itself. To run it as a long-lived network
service instead (for the Docker/k8s deployment below), set
`MCP_TRANSPORT=streamable-http`.

## Docker

```bash
docker build -t ollama-code-mcp .
docker run --rm -p 8765:8765 \
  -e OLLAMA_BASE_URL=http://ollama.ash4d.com:11434 \
  -e MCP_TRANSPORT=streamable-http \
  -v /path/to/your/code:/workspace \
  -e OLLAMA_MCP_ALLOWED_DIR=/workspace \
  ollama-code-mcp
```

Claude Code would then connect to it as a remote MCP server at
`http://<host>:8765/mcp`.

## Kubernetes / k3s

Manifests are in [`k8s/`](./k8s). They assume Ollama is already running in
the same cluster (e.g. via the `ollama-helm` chart with a `LoadBalancer`
service on port 11434, as in `ollama-current-values.yaml`), and reach it over
the cluster-internal service DNS name rather than a LAN IP:

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

`k8s/configmap.yaml` ships pointing `OLLAMA_BASE_URL` at
`http://ollama.ash4d.com:11434`; edit it if your Ollama lives elsewhere
(e.g. the cluster-internal `http://ollama.ollama.svc.cluster.local:11434`
when both run in the same cluster). Adjust the
image reference in `k8s/deployment.yaml` to wherever you push the built
image. The manifests expose the server via `streamable-http` on a `ClusterIP`
service fronted by an Ingress at `https://mcp-ollama.ash4d.com/mcp`
(`k8s/ingress.yaml`). TLS is required because Claude's custom-connector UI
only accepts https URLs; the host resolves to a private IP, so provision the
cert via cert-manager DNS-01 or an existing wildcard secret (see the comments
in `ingress.yaml`). Register the connector in Claude (desktop or Claude Code)
as `https://mcp-ollama.ash4d.com/mcp`.

## Security notes

- File-aware tools are confined to `OLLAMA_MCP_ALLOWED_DIR` via path
  resolution + containment checks -- paths that resolve outside it (e.g.
  `../../etc/passwd`) are rejected.
- `batch_refactor` writes are opt-in (`dry_run=False`) and capped in count
  (`OLLAMA_MCP_MAX_BATCH_FILES`) and per-file size
  (`OLLAMA_MCP_MAX_FILE_BYTES`).
- This server has no authentication of its own. If you deploy it with a
  network transport (`sse`/`streamable-http`), keep it on a trusted LAN or
  put it behind a network policy / VPN -- do not expose it to the public
  internet.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests mock all Ollama HTTP calls (via `respx`) and use `tmp_path` for file
operations, so they run offline and don't need a real Ollama instance.

## License

MIT -- see [LICENSE](./LICENSE).
