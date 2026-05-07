# llama-index-tools-goodmem

[GoodMem](https://goodmem.ai) is a server-side memory layer for AI agents with semantic storage, retrieval, and LLM-powered summarization. This package exposes GoodMem's API as a [LlamaIndex](https://docs.llamaindex.ai) `BaseToolSpec`, so any LlamaIndex agent can store, search, and manage long-term memories.

## Install

```bash
pip install -e .          # editable install from this repo
# or, with dev/test deps:
pip install -e ".[dev]"
```

## Quick start

```python
from llama_index.tools.goodmem import GoodMemToolSpec

tool_spec = GoodMemToolSpec(
    api_key="gm_xxx",
    base_url="https://api.goodmem.ai",
    verify_ssl=True,    # set False for a self-signed dev cert
)

# Use with a LlamaIndex agent
tools = tool_spec.to_tool_list()
```

### Constructor parameters

| Parameter    | Type  | Required | Description                                              |
|--------------|-------|----------|----------------------------------------------------------|
| `api_key`    | str   | yes      | GoodMem API key (sent as `X-API-Key`)                    |
| `base_url`   | str   | yes      | Base URL of your GoodMem server                          |
| `verify_ssl` | bool  | no       | Verify TLS certificates. Default `True`.                 |
| `timeout`    | float | no       | Per-request timeout in seconds. Default `120` (LLM-summary retrievals can take tens of seconds). |

## Tools

The tool spec exposes 11 sync/async tool pairs. The names below are the sync entry points; each has an async counterpart prefixed with `a` (e.g. `aretrieve_memories`).

| Tool                 | Purpose                                                                 |
|----------------------|-------------------------------------------------------------------------|
| `list_embedders`     | List server-managed embedder models.                                    |
| `list_spaces`        | List spaces.                                                            |
| `get_space`          | Fetch one space by ID.                                                  |
| `create_space`       | Create a new space (or reuse one with the same name).                   |
| `update_space`       | Rename a space, toggle `publicRead`, or merge/replace labels.           |
| `delete_space`       | Delete a space (cascades to its memories).                              |
| `create_memory`      | Store a text or file payload as a memory.                               |
| `list_memories`      | Paginate memories in a space (with optional status / metadata filters). |
| `retrieve_memories`  | Semantic search; optional rerank, threshold, LLM summary, chrono-resort. |
| `get_memory`         | Fetch one memory and (optionally) its original content.                 |
| `delete_memory`      | Delete a memory.                                                        |

`retrieve_memories` returns `List[llama_index.core.schema.Document]`. Each `Document.text` is a matched chunk; `Document.metadata` includes `chunkId`, `memoryId`, `relevanceScore`, `resultSetId`, `query`, plus `abstractReply` if an `llm_id` was passed.

### Post-processor knobs (`retrieve_memories`)

| Argument                | Type            | Notes                                                          |
|-------------------------|-----------------|----------------------------------------------------------------|
| `reranker_id`           | UUID            | Reranks results by direct query–chunk scoring.                 |
| `llm_id`                | UUID            | Generates an `abstractReply` summary across the result set.    |
| `relevance_threshold`   | float `0..1`    | Drops results below this score.                                |
| `llm_temperature`       | float `0..2`    | LLM creativity (sent on the wire as `llm_temp`).               |
| `chronological_resort`  | bool            | Re-sort the post-processor output by creation time.            |

## Examples

```python
# Create a space
space = tool_spec.create_space(name="my-research", embedder_id="<uuid>")

# Add memories
tool_spec.create_memory(space_id=space["spaceId"], text_content="...")
tool_spec.create_memory(space_id=space["spaceId"], file_path="/path/to/doc.pdf")

# Search
docs = tool_spec.retrieve_memories(
    query="What's the conclusion?",
    space_ids=[space["spaceId"]],
    max_results=5,
    llm_id="<llm-uuid>",
    llm_temperature=0.2,
)
for d in docs:
    print(d.metadata["relevanceScore"], d.text[:120])

# LLM summary (when llm_id is set)
print(docs[0].metadata["abstractReply"]["text"])
```

## Use inside a LlamaIndex agent

```python
from llama_index.tools.goodmem import GoodMemToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

tool_spec = GoodMemToolSpec(api_key="gm_xxx", base_url="https://api.goodmem.ai")
agent = ReActAgent.from_tools(tool_spec.to_tool_list(), llm=OpenAI(model="gpt-4o-mini"))

response = agent.chat("Remember that the project deadline is May 14.")
```

## Testing

### Unit tests (no live server)

```bash
pip install -e ".[dev]"
pytest tests/test_tools_goodmem.py -v
```

### Live e2e smoke test

Exercises every one of the 11 tools and every post-processor knob against a running GoodMem server. Mirrors the §7 smoke-test plan in the GoodMem build guide.

```bash
GOODMEM_API_KEY=gm_xxx \
GOODMEM_BASE_URL=https://localhost:8080 \
GOODMEM_EMBEDDER_ID=<embedder-uuid> \
GOODMEM_RERANKER_ID=<reranker-uuid> \
GOODMEM_LLM_ID=<llm-uuid> \
GOODMEM_VERIFY_SSL=false \
pytest tests/test_tools_goodmem_e2e.py -v
```

`GOODMEM_RERANKER_ID` and `GOODMEM_LLM_ID` are optional — the variants that need them are auto-skipped if unset. The full 18-step smoke test passes when all three are configured.

## License

MIT
