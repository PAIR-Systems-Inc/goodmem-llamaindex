# GoodMem for LlamaIndex

Use [GoodMem](https://goodmem.ai) as a persistent document and retrieval service in LlamaIndex. GoodMem handles chunking, embeddings and optional reranking; the integration returns native `NodeWithScore` objects for query engines and agents.

```bash
pip install 'llamaindex-goodmem>=0.2.0'
```

The import namespace is `llama_index.tools.goodmem`. Set `GOODMEM_BASE_URL` to your server’s REST root and `GOODMEM_API_KEY` to its API key.

## Store and retrieve Documents

Use an existing GoodMem space configured with an embedder:

```python
import os

from goodmem import Goodmem
from llama_index.core.schema import Document
from llama_index.tools.goodmem import (
    GoodMemDocumentIngestor,
    GoodMemRetriever,
    wait_for_memories,
)

with Goodmem(base_url=os.environ["GOODMEM_BASE_URL"],
             api_key=os.environ["GOODMEM_API_KEY"]) as client:
    space_id = os.environ["GOODMEM_SPACE_ID"]
    ids = GoodMemDocumentIngestor(client=client, space_id=space_id).add_documents([
        Document(text="The returns period is 30 days.",
                 metadata={"source": "https://example.com/returns"})
    ])
    wait_for_memories(client, ids, timeout=120)

    retriever = GoodMemRetriever(client=client, space_ids=[space_id])
    for result in retriever.retrieve("How long do I have to return an item?"):
        print(result.score, result.text, result.metadata.get("source"))
```

`add_documents` returns accepted IDs without waiting. Waiting is explicit and checks only those IDs. Ordinary empty searches return immediately.

## Connect an agent

Use LlamaIndex’s own tool wrapper. Give each collection a useful name and description:

```python
from llama_index.core.tools import RetrieverTool

retriever = GoodMemRetriever(space_ids=[space_id])  # uses environment settings
search = RetrieverTool.from_defaults(
    retriever,
    name="returns_policy",
    description="Search the company's returns and refund policies.",
)
# Pass search to a workflow-based ReActAgent or FunctionAgent.
```

The model supplies the query; the application configures spaces, filters and reranking. The same retriever works with `RetrieverQueryEngine` and standard LlamaIndex callbacks.

Pass `filters=MetadataFilters(...)` for supported scalar comparisons, membership tests and nested conditions. Pass `reranker_id=...` to rerank on the server without an LLM. Sources, custom metadata and Document metadata exclusions survive storage. Framework scores rank higher as more relevant; `raw_score` preserves the original value.

## Async and administrative tools

`aretrieve` and `aadd_documents` use the SDK’s `AsyncGoodmem` directly. Inject `async_client` to share its connection pool; caller-owned clients remain open.

`GoodMemToolSpec` supplies optional space and memory management tools. Its retrieval result includes chunks, statuses and a `partial` flag. File uploads require an explicitly configured directory. Prefer a scoped retriever tool when an agent only needs search.

See [usage and migration](https://github.com/PAIR-Systems-Inc/goodmem-llamaindex/blob/v0.2.0/docs/usage.md) for async examples, supported filters and diagnostics, and the [changelog](https://github.com/PAIR-Systems-Inc/goodmem-llamaindex/blob/v0.2.0/CHANGELOG.md) for the changes from 0.1. Run `pip install -e '.[dev]'` and `pytest` for the test suite. Live tests are opt-in and clean up their own spaces.
