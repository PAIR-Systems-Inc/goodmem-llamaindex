# Changelog

## 0.2.1 — 2026-09-25

Security fix: IDs can no longer reach a different resource through the URL path.

- Refuse any space, memory, embedder, reranker or LLM ID that is not a UUID in the standard 8-4-4-4-12 hexadecimal form, with a `ValueError` naming the argument, before any request is sent. The official SDK interpolates IDs into URL paths unescaped and httpx resolves dot segments, so in 0.2.0 `delete_memory("../spaces/<id>")` sent `DELETE /v1/spaces/<id>` and reported success. `get_space`, `update_space`, `delete_space`, `list_memories`, `get_memory` and `delete_memory` (sync and async) were affected; `<id>?x=1` and `<id>#frag` also altered the request.
- Apply the same check to IDs sent in request bodies: `create_space`, `create_memory`, `upload_memory`, `retrieve_memories`, `GoodMemRetriever`, `GoodMemDocumentIngestor`, `wait_for_memories` and `await_memories`. An empty `reranker_id` or `llm_id` is now refused instead of being ignored; pass `None` to omit it.
- Normalize accepted UUIDs to lowercase. A space ID configured in uppercase now derives the same memory IDs for non-UUID Document IDs as its lowercase form, and waiting on uppercase memory IDs matches the server's lowercase results.
- Mark ID arguments in agent tool schemas with the JSON-schema `uuid` format.

## 0.2.0 — 2026-09-14

A clean API update using the official GoodMem SDK, with native LlamaIndex retrieval and Document ingestion.

- Add `GoodMemRetriever`, returning `NodeWithScore` objects with stable chunk IDs, source relationships and stored metadata. Supports callbacks, query engines, scoped `RetrieverTool`s, metadata filters and LLM-free server reranking.
- Add `GoodMemDocumentIngestor.add_documents` / `aadd_documents`, returning accepted memory IDs. `wait_for_memories` / `await_memories` explicitly poll batches of pending IDs under a caller-selected timeout.
- Replace the duplicate HTTP client and NDJSON parser with the official SDK. Retrieval and ingestion have native async paths.
- Keep useful chunks and diagnostics in administrative retrieval results, including a `partial` flag. Unknown future status codes do not abort retrieval; known failures raise in the standard retriever.
- Preserve source and user metadata. Return readable text from get-memory instead of attempting to parse every content response as JSON. Preserve metadata when content cannot be downloaded.
- Follow space and memory pagination through the SDK. Space creation now always creates; duplicate names return the SDK conflict error.
- Remove `wait_for_indexing`, `public_read`, name-based space reuse and agent-facing chunking configuration. File upload is a separate, opt-in tool restricted to a configured directory.
- Update examples to LlamaIndex’s workflow-based agent API. Replace internal-method mocks with SDK-over-HTTP regression tests and opt-in live tests.
- Keep injected connections authoritative in both sync and async calls. Calling the other mode requires its corresponding injected client; environment settings cannot silently select another server or credential.
- Convert vector scores to LlamaIndex's higher-is-better direction. `GoodMemNodeWithScore` retains raw scores, their kind and statuses outside node metadata, keeping identity stable across queries. Reranker scores are unchanged. Cover all four query-fusion modes with multiple queries and the standard similarity postprocessor.
- Return accepted memory IDs and recovery guidance to agents when indexing confirmation fails after a successful write. Initial write errors and explicit wait-helper exceptions keep their normal behavior.
- Retain memory metadata after content transport failures with caller-supplied HTTP clients. Declare the HTTP dependency used by this fallback.
- Persist and restore Document LLM and embedding metadata exclusions through ingestion and retrieval, including standard `RetrieverTool` rendering.
- Match LlamaIndex's missing/null behavior for inequality, excluded membership and negated comparisons through explicit null checks.

Tool results use SDK snake_case field names. Retrieval returns a compact dictionary instead of `List[Document]`; use `GoodMemRetriever` for framework retrieval. See `docs/usage.md`.

## 0.1.0

Initial `GoodMemToolSpec` integration with synchronous and asynchronous administrative tools.
