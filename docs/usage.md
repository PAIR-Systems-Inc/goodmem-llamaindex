# Usage and migration

## Native retrieval

`GoodMemRetriever(space_ids=[...])` is a `BaseRetriever`. It supports `retrieve`, `aretrieve`, callbacks and native wrappers such as `RetrieverTool.from_defaults` and `RetrieverQueryEngine.from_args`.

Use `top_k` for the returned chunk count and `fetch_k` for the candidate count. Reranking defaults to four times `top_k` candidates and returns at most `top_k`. It requires `reranker_id`, not an LLM.

Results are `GoodMemNodeWithScore` objects, a `NodeWithScore` subclass. Their `score` follows LlamaIndex's higher-is-better convention for fusion and similarity postprocessors. GoodMem's vector scores are negative inner products, so the retriever negates them. Successful reranker scores are unchanged. `result.raw_score` preserves the server value, `result.score_kind` identifies `negative_inner_product` or `reranker`, and `result.statuses` holds retrieval diagnostics. These query-dependent fields live on the result wrapper, outside the `TextNode` hash, so multi-query fusion combines repeated chunks correctly. Scores are not normalized to 0–1 or calibrated across models; choose similarity thresholds for your model. A known reranker failure raises before fallback vector scores can be treated as reranker scores. Administrative retrieval keeps the original server scores.

Stored metadata is copied onto each `TextNode`. A missing `source` falls back to the original content reference when available. `_goodmem` contains server IDs and original memory/chunk metadata maps. This namespace is excluded from LLM context. Original maps preserve collisions rather than losing user fields. A SOURCE relationship points to the GoodMem memory; the node ID is the chunk ID.

Document ingestion persists `excluded_llm_metadata_keys` and `excluded_embed_metadata_keys` under the reserved `_llamaindex_goodmem` metadata key. Retrieval restores them and hides integration metadata from LLM/embedding content. The fields remain available programmatically; exclusions control formatting, not server access. Existing memories without these settings have no original exclusions to restore. The ingestor rejects a user field with the reserved name rather than overwriting it.

Known failure statuses raise `GoodMemRetrievalError`. Unknown codes are exposed as `UNKNOWN` and do not abort retrieval. The administrative tool retains chunks and sets `partial=true` for non-informational diagnostics. Consumers should inspect `statuses` when partial is true. The SDK owns stream parsing, HTTP errors and unknown-code deserialization.

## Metadata filtering

```python
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

filters = MetadataFilters(filters=[
    MetadataFilter(key="tenant", value="acme"),
    MetadataFilter(key="year", operator=">=", value=2025),
])
retriever = GoodMemRetriever(space_ids=[space_id], filters=filters)
```

Supported operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `nin`. Supported conditions: nested AND, OR and single-child NOT. Keys are simple top-level field names; values are strings or finite numbers. Unsupported operators raise before a request. An empty filter applies no constraint. Values are escaped for the server’s expression grammar.

Missing and null fields follow LlamaIndex's matching behavior: `!=`, `nin` and negated equality include them; positive comparisons do not. Translation uses explicit null checks so SQL's three-valued logic cannot silently change negated filter results.

For other paths and operations, pass an application-owned native GoodMem expression with `filter=...`. When both are supplied, they are combined with AND. Do not build raw expressions by concatenating untrusted input.

## Native async

```python
from goodmem import AsyncGoodmem
from llama_index.tools.goodmem import GoodMemDocumentIngestor, await_memories

async with AsyncGoodmem(base_url=base_url, api_key=api_key) as client:
    ids = await GoodMemDocumentIngestor(
        async_client=client, space_id=space_id
    ).aadd_documents(documents)
    await await_memories(client, ids, timeout=120)
    nodes = await GoodMemRetriever(
        async_client=client, space_ids=[space_id]
    ).aretrieve("What changed?")
```

Inject `client` for synchronous calls and `async_client` for asynchronous calls. When either is injected, calls in the other mode require its corresponding client too. Environment variables and scalar connection options cannot substitute for a missing injected client. This applies to retrieval, ingestion and administrative tools. Without injected clients, configure `base_url`, `api_key`, `timeout` and `verify_ssl`, or the two GoodMem environment variables. `verify_ssl` can be a CA-bundle path for a local certificate. SDK HTTP timeouts are separate from the indexing polling budget.

## Ingestion and recovery

Document UUIDs become memory IDs. Non-UUID Document IDs map deterministically within the destination space. Ingestion creates memories; it does not update existing records or download URLs in metadata. Documents are sent as whole text so GoodMem owns chunking and embeddings.

`add_documents` and `aadd_documents` do not wait by default. They batch up to 100 Documents per request. An HTTP failure before any confirmed write keeps its SDK exception type. A partial write raises `GoodMemIngestionError` with `created_memory_ids`; confirmed writes are not rolled back. You may wait for these IDs without uploading again.

`wait_for_memories` and `await_memories` batch status lookups and remove completed IDs from subsequent rounds. `GoodMemIndexingError.pending_memory_ids` identifies memories to inspect or await again. A polling timeout does not imply a write failed. The administrative create-memory tool waits for its own memory by default, using its configured `indexing_timeout`.

If that tool's wait fails, it returns an accepted receipt with `memory_id`, `space_id`, `accepted=true`, `indexing.status="unconfirmed"`, the waiting error and recovery guidance. This is also what an agent sees. Call `get_memory` with the accepted ID to check its current state; do not upload again. The receipt omits the creation response's stale `processing_status`. A successful wait returns `processing_status="COMPLETED"`; `wait=False` returns the creation response immediately. Initial creation errors still raise, and the explicit wait helpers retain their exception-based API.

## Administrative changes from 0.1

Use SDK objects directly for advanced server configuration. `GoodMemToolSpec` retains eleven basic operations, plus an optional file-upload tool:

- `create_space(name, embedder_id)` always creates; duplicate names raise `ConflictError`. Chunking customization belongs in the SDK.
- `update_space(space_id, name, labels)` renames or replaces labels; `public_read` is removed.
- Listing returns lists of SDK dictionaries with snake_case fields. Spaces and memories are paginated up to `max_items`.
- `create_memory` accepts text and metadata. `upload_memory` exists in the tool list only when `upload_directory` is configured. Paths and resolved symlinks must stay inside that directory.
- `get_memory` returns memory metadata plus readable `content`, or a content notice/error. Both SDK errors and HTTP transport errors during content download preserve the metadata. The initial metadata lookup's errors still raise. Binary content is downloaded through the SDK, not handed to the model as base64.
- `retrieve_memories` returns `chunks`, `statuses`, `partial` and `abstract_reply`; it has no `wait_for_indexing`. Use `GoodMemRetriever` when you need LlamaIndex nodes.

Construct agents from `llama_index.core.agent.workflow` and use `await agent.run(...)`. The older `ReActAgent.from_tools(...).chat(...)` pattern is not available in the tested LlamaIndex 0.14.24.
