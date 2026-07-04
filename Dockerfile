FROM registry.suse.com/bci/python:3.11

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 1000 mcp \
    && mkdir -p /workspace \
    && chown -R mcp:mcp /workspace
USER mcp

WORKDIR /workspace

ENV MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8765 \
    OLLAMA_MCP_ALLOWED_DIR=/workspace

EXPOSE 8765

ENTRYPOINT ["ollama-code-mcp"]
