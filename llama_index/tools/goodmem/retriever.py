"""LlamaIndex retrieval with server-side embeddings, filters and reranking."""

import logging
import warnings
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
from ._ids import require_uuid
from ._metadata import _DOCUMENT_METADATA, stored_metadata
from .filters import filter_expression

logger = logging.getLogger(__name__)

# Q1 of the GoodMem retrieval status contract: these codes report an optional
# feature the caller did not configure, never a missing part of the result.
_NOISE_CODES = frozenset({"LLM_CAPABILITY_INFERRED", "FEATURE_DISABLED"})


class GoodMemRetrievalError(RuntimeError):
    """Retained for compatibility; retrieval no longer raises it.

    Since 0.2.2 a problem reported by the server is returned as a partial result
    with its statuses: hits the server sent are kept, and an empty result warns.
    """

    def __init__(self, statuses):
        self.statuses = statuses
        super().__init__("; ".join(f"{s['code']}: {s['message']}" for s in statuses))


class GoodMemNodeWithScore(NodeWithScore):
    """A native scored node with retrieval diagnostics outside its content identity.

    LlamaIndex hashes TextNode metadata for fusion and deduplication. Query-dependent
    raw scores, statuses and the partial flag belong on this result wrapper, never
    on the TextNode.
    """

    node: TextNode
    raw_score: float
    score_kind: Literal["negative_inner_product", "reranker"] = "negative_inner_product"
    statuses: list[dict[str, Any]] = Field(default_factory=list)
    partial: bool = False


def informational(status):
    """Identify notices that do not imply incomplete retrieval, by code alone."""
    return status.code in _NOISE_CODES


def reranking_failed(statuses) -> bool:
    """Whether the server reported that a requested reranker did not run.

    RERANKING_FAILED says so directly. A NOT_FOUND naming the reranker (server
    v1.0.320: details {"reranker_id": ...}, "Reranker not found") means the same,
    even alone. The server then returns the vector-stage hits instead.
    """
    for status in statuses:
        details = status.details or {}
        if status.code == "RERANKING_FAILED":
            return True
        if status.code == "NOT_FOUND" and (
            "reranker_id" in details
            or "rerankerId" in details
            or "reranker" in (status.message or "").lower()
        ):
            return True
    return False


def problem_summary(statuses: list[dict[str, Any]]) -> str:
    """Name the reported problems, leaving out informational notices."""
    return "; ".join(
        f"{s['code']}: {s['message']}" if s.get("message") else s["code"]
        for s in statuses
        if s["code"] not in _NOISE_CODES
    )


def retrieval_result(
    events: Iterable[RetrieveMemoryEvent], *, reranker_requested: bool = False
) -> dict[str, Any]:
    """Join complete SDK events, preserving sources and future status codes.

    Never raises on statuses: hits the server returned are always kept, and
    ``partial`` reports any status that is not informational. Scores are
    reranker scores only if a reranker was requested and the server did not
    report that it failed; this is decided after the whole stream is read,
    because a failure status can follow the hits. Vector scores are negative
    inner products, so node scores negate them to LlamaIndex's higher-is-better
    direction; ``raw_score`` keeps the server value.

    Returns:
        Nodes, original statuses, partial flag, score kind and optional abstract
        reply. Unknown codes become UNKNOWN through the Python SDK's None fallback.
    """
    events = list(events)
    statuses = [
        e.status.model_dump(mode="json", exclude_none=True) | {"code": e.status.code or "UNKNOWN"}
        for e in events
        if e.status
    ]
    partial = any(e.status and not informational(e.status) for e in events)
    reranked = reranker_requested and not reranking_failed([e.status for e in events if e.status])
    score_kind = "reranker" if reranked else "negative_inner_product"
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
                score=hit.relevance_score if reranked else -hit.relevance_score,
                raw_score=hit.relevance_score,
                score_kind=score_kind,
                statuses=statuses,
                partial=partial,
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
        "partial": partial,
        "score_kind": score_kind,
        "abstract_reply": abstract,
    }


class GoodMemRetriever(BaseRetriever):
    """Retrieve NodeWithScore objects through the official sync/async SDK.

    Scores follow LlamaIndex's higher-is-better convention. Raw server scores,
    their kind, statuses and a partial flag remain on the GoodMemNodeWithScore
    wrapper, outside node identity. Retrieval does not raise on server statuses:
    when a configured reranker fails, the server's vector hits are returned as
    vector scores with partial=True. A reported problem with no hits returns an
    empty list, a UserWarning and a WARNING log line naming the statuses.

    Args:
        space_ids: Application-configured space UUIDs to search.
        top_k: Maximum returned chunks.
        fetch_k: Candidate count; defaults to four times top_k when reranking.
        reranker_id: Optional server reranker UUID. No LLM is needed.
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
        if not space_ids:
            raise ValueError("At least one space ID is required")
        if top_k < 1 or (fetch_k is not None and fetch_k < top_k):
            raise ValueError("top_k must be positive and fetch_k must be at least top_k")
        self._connection = Connection(**connection)
        self.space_ids = tuple(require_uuid(s, f"space_ids[{i}]") for i, s in enumerate(space_ids))
        if reranker_id is not None:
            reranker_id = require_uuid(reranker_id, "reranker_id")
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
        """Return the server's hits, reporting problems instead of raising."""
        result = retrieval_result(events, reranker_requested=bool(self.reranker_id))
        nodes = result["nodes"][: self.top_k]
        if result["partial"]:
            problems = problem_summary(result["statuses"])
            logger.warning(
                "GoodMem reported a problem during retrieval (%d result(s) returned): %s",
                len(nodes),
                problems,
            )
            if not nodes:
                # A bare list has no slot for the partial flag; make the empty result visible.
                warnings.warn(
                    f"GoodMem returned no results and reported a problem: {problems}",
                    UserWarning,
                    stacklevel=2,
                )
        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        with self._connection.sync() as client:
            events = client.memories.retrieve(**self._request(query_bundle.query_str))
        return self._nodes(events)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        async with self._connection.async_() as client:
            events = await client.memories.retrieve(**self._request(query_bundle.query_str))
        return self._nodes(events)
