# Changelog

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
