"""LlamaIndex retrieval with server-side embeddings, filters and reranking."""

from collections.abc import Iterable
from typing import Any, Literal

from llama_index.core.callbacks import CallbackManager
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import (
    NodeRelationship,
    NodeWithScore,
    QueryBundle,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import MetadataFilters
from pydantic import Field

from goodmem.models.retrieve_memory_event import RetrieveMemoryEvent
from goodmem.models.space_key import SpaceKey

from ._connection import Connection
from ._metadata import _DOCUMENT_METADATA, stored_metadata
from .filters import filter_expression

# Only failures whose meaning this integration understands can abort retrieval.
# A newer SDK may recognize a new informational code before this package does.
_KNOWN_FAILURES = frozenset(
    {
        "INVALID_ARGUMENT",
        "NOT_FOUND",
        "PERMISSION_DENIED",
        "FAILED_PRECONDITION",
        "EMBEDDER_FAILED",
        "EMBEDDER_UNAVAILABLE",
        "EMBEDDER_TIMEOUT",
        "VECTOR_SEARCH_FAILED",
        "VECTOR_SEARCH_PARTIAL",
        "VECTOR_SEARCH_TIMEOUT",
        "SPACE_INACCESSIBLE",
        "SPACE_NOT_FOUND",
        "SPACE_NO_EMBEDDERS",
        "CHUNK_NOT_FOUND",
        "MEMORY_LOAD_FAILED",
        "MEMORY_CONTENT_UNAVAILABLE",
        "RERANKING_FAILED",
        "SUMMARIZATION_FAILED",
        "SUMMARIZATION_TIMEOUT",
        "RATE_LIMITED",
        "RESOURCE_EXHAUSTED",
        "CONFIGURATION_ERROR",
    }
)


class GoodMemRetrievalError(RuntimeError):
    """A known retrieval failure, with the original server diagnostics."""

    def __init__(self, statuses):
        self.statuses = statuses
        super().__init__("; ".join(f"{s['code']}: {s['message']}" for s in statuses))


class GoodMemNodeWithScore(NodeWithScore):
    """A native scored node with retrieval diagnostics outside its content identity.

    LlamaIndex hashes TextNode metadata for fusion and deduplication. Query-dependent
    raw scores and statuses belong on this result wrapper, never on the TextNode.
    """

    node: TextNode
    raw_score: float
    score_kind: Literal["negative_inner_product", "reranker"] = "negative_inner_product"
    statuses: list[dict[str, Any]] = Field(default_factory=list)


def informational(status):
    """Identify notices that do not imply incomplete retrieval."""
    details = status.details or {}
    return status.code == "LLM_CAPABILITY_INFERRED" or (
        status.code == "FEATURE_DISABLED"
        and details.get("feature") == "summarization"
        and details.get("required_param") == "llm_id"
    )


def retrieval_result(
    events: Iterable[RetrieveMemoryEvent], *, strict: bool = False
) -> dict[str, Any]:
    """Join complete SDK events, preserving sources and future status codes.

    Returns:
        Nodes, original statuses, partial flag and optional abstract reply.
        Unknown codes become UNKNOWN through the Python SDK's None fallback.
    """
    events = list(events)
    statuses = [
        e.status.model_dump(mode="json", exclude_none=True) | {"code": e.status.code or "UNKNOWN"}
        for e in events
        if e.status
    ]
    failures = [e.status for e in events if e.status and not informational(e.status)]
    if strict and any(s.code in _KNOWN_FAILURES for s in failures):
        raise GoodMemRetrievalError(
            [
                s.model_dump(mode="json", exclude_none=True) | {"code": s.code or "UNKNOWN"}
                for s in failures
            ]
        )
    memories = {
        e.memory_definition.memory_id: e.memory_definition for e in events if e.memory_definition
    }
    nodes, seen = [], set()
    for event in events:
        item = event.retrieved_item
        if item is None or item.chunk is None:
            continue
        hit, chunk = item.chunk, item.chunk.chunk
        if chunk.chunk_id in seen or not chunk.chunk_text:
            continue
        seen.add(chunk.chunk_id)
        memory = memories.get(chunk.memory_id)
        memory_metadata, exclusions = stored_metadata(memory.metadata if memory else None)
        chunk_metadata = dict(getattr(chunk, "metadata", None) or {})
        metadata = memory_metadata | chunk_metadata
        if memory and memory.original_content_ref:
            metadata.setdefault("source", memory.original_content_ref)
        metadata["_goodmem"] = {
            "memory_id": chunk.memory_id,
            "chunk_id": chunk.chunk_id,
            "space_id": memory.space_id if memory else None,
            "memory_metadata": memory_metadata,
            "chunk_metadata": chunk_metadata,
        }
        nodes.append(
            GoodMemNodeWithScore(
                node=TextNode(
                    id_=chunk.chunk_id,
                    text=chunk.chunk_text,
                    metadata=metadata,
                    excluded_llm_metadata_keys=["_goodmem", _DOCUMENT_METADATA]
                    + exclusions["excluded_llm_metadata_keys"],
                    excluded_embed_metadata_keys=["_goodmem", _DOCUMENT_METADATA]
                    + exclusions["excluded_embed_metadata_keys"],
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(
                            node_id=chunk.memory_id, metadata=memory_metadata
                        )
                    },
                ),
                score=hit.relevance_score,
                raw_score=hit.relevance_score,
                statuses=statuses,
            )
        )
    abstract = next(
        (
            e.abstract_reply.model_dump(mode="json", exclude_none=True)
            for e in reversed(events)
            if e.abstract_reply
        ),
        None,
    )
    return {
        "nodes": nodes,
        "statuses": statuses,
        "partial": bool(failures),
        "abstract_reply": abstract,
    }


class GoodMemRetriever(BaseRetriever):
    """Retrieve NodeWithScore objects through the official sync/async SDK.

    Scores follow LlamaIndex's higher-is-better convention. Raw server scores
    and their kind remain on the GoodMemNodeWithScore wrapper, outside node identity.

    Args:
        space_ids: Application-configured spaces to search.
        top_k: Maximum returned chunks.
        fetch_k: Candidate count; defaults to four times top_k when reranking.
        reranker_id: Optional server reranker. No LLM is needed.
        filters: LlamaIndex MetadataFilters for supported comparisons.
        filter: Explicit native GoodMem filter expression. Combined with filters.
        callback_manager: Standard LlamaIndex retrieval callbacks.
        connection: SDK clients or connection options accepted by Connection.
    """

    def __init__(
        self,
        *,
        space_ids: list[str],
        top_k: int = 5,
        fetch_k: int | None = None,
        reranker_id: str | None = None,
        filters: MetadataFilters | None = None,
        filter: str | None = None,
        callback_manager: CallbackManager | None = None,
        **connection: Any,
    ) -> None:
        super().__init__(callback_manager=callback_manager)
        if not space_ids or any(not s.strip() for s in space_ids):
            raise ValueError("At least one nonempty space ID is required")
        if top_k < 1 or (fetch_k is not None and fetch_k < top_k):
            raise ValueError("top_k must be positive and fetch_k must be at least top_k")
        self._connection = Connection(**connection)
        self.space_ids = tuple(space_ids)
        self.top_k, self.fetch_k, self.reranker_id = top_k, fetch_k, reranker_id
        expressions = [e for e in (filter_expression(filters), filter) if e]
        self.filter = " AND ".join(f"({e})" for e in expressions) or None

    def _request(self, query):
        if not query.strip():
            raise ValueError("Query must not be empty")
        options = {
            "message": query,
            "stream": False,
            "fetch_memory": True,
            "fetch_memory_content": False,
            "space_keys": [SpaceKey(space_id=s, filter=self.filter) for s in self.space_ids],
            "requested_size": self.fetch_k or self.top_k * (4 if self.reranker_id else 1),
        }
        if self.reranker_id:
            options.update(
                reranker_id=self.reranker_id, max_results=self.top_k, chronological_resort=False
            )
        return options

    def _nodes(self, events):
        """Adapt vector scores for LlamaIndex without changing reranker scores."""
        nodes = retrieval_result(events, strict=True)["nodes"][: self.top_k]
        for node in nodes:
            node.score_kind = "reranker" if self.reranker_id else "negative_inner_product"
            if not self.reranker_id:
                node.score = -node.raw_score
        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        with self._connection.sync() as client:
            events = client.memories.retrieve(**self._request(query_bundle.query_str))
        return self._nodes(events)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        async with self._connection.async_() as client:
            events = await client.memories.retrieve(**self._request(query_bundle.query_str))
        return self._nodes(events)
