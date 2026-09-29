# GoodMem for LlamaIndex

Use [GoodMem](https://goodmem.ai) as a persistent document and retrieval service in LlamaIndex. GoodMem handles chunking, embeddings and optional reranking; the integration returns native `NodeWithScore` objects for query engines and agents.

```bash
pip install 'goodmem-llamaindex>=0.2.3'
```

This package was previously published as `llamaindex-goodmem` (last version on that name: 0.2.2). It moved into the PAIR Systems PyPI organisation under the `goodmem-<framework>` naming used by goodmem-adk and goodmem-semantic-kernel. The import name is unchanged. Both distributions ship the same `llama_index.tools.goodmem` files and would overwrite each other, so remove the old one first:

```bash
pip uninstall -y llamaindex-goodmem && pip install goodmem-llamaindex
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
import os

from llama_index.core.tools import RetrieverTool
from llama_index.tools.goodmem import GoodMemRetriever

space_id = os.environ["GOODMEM_SPACE_ID"]
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

Retrieval does not raise on problems the server reports in its results; it returns what the server sent. HTTP errors, such as an unknown space, still raise the SDK's exception. If a configured reranker fails (for example, its ID does not exist), GoodMem reports `NOT_FOUND` and `RERANKING_FAILED` and still returns the vector-search hits. The retriever returns those hits as vector scores, negated like any vector score, with `score_kind="negative_inner_product"`, `partial=True` and the server's `statuses` on each result, and logs a WARNING. A reported problem with no hits returns an empty list, a `UserWarning` and a WARNING log line naming the statuses.

## Async and administrative tools

`aretrieve` and `aadd_documents` use the SDK’s `AsyncGoodmem` directly. Inject `async_client` to share its connection pool; caller-owned clients remain open.

`GoodMemToolSpec` supplies optional space and memory management tools. Its retrieval result includes chunks with the server's raw scores, `statuses`, a `partial` flag and `score_kind` (`reranker`, or `negative_inner_product` for vector scores, including when reranking failed). File uploads require an explicitly configured directory. Prefer a scoped retriever tool when an agent only needs search.

Every GoodMem ID you or a model pass in (space, memory, embedder, reranker or LLM) must be a UUID. Anything else raises `ValueError` before a request is sent, because the SDK puts IDs into URL paths, where a value such as `../spaces/<id>` would reach a different resource.

See [usage and migration](https://github.com/PAIR-Systems-Inc/goodmem-llamaindex/blob/v0.2.3/docs/usage.md) for async examples, supported filters and diagnostics, and the [changelog](https://github.com/PAIR-Systems-Inc/goodmem-llamaindex/blob/v0.2.3/CHANGELOG.md) for the changes from 0.1. Run `pip install -e '.[dev]'`, then `pytest`, `ruff check llama_index tests` and `ruff format --check llama_index tests`, as CI does. Live tests are opt-in and clean up their own spaces.
