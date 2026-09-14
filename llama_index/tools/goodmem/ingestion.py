"""Write LlamaIndex Documents and explicitly wait for known memory IDs."""

import asyncio
import time
import uuid
from collections.abc import Iterable
from typing import Any

from llama_index.core.schema import Document, MetadataMode

from goodmem import AsyncGoodmem, Goodmem, MemoryCreationRequest

from ._connection import Connection
from ._metadata import document_metadata


class GoodMemIngestionError(RuntimeError):
    """A partial write; accepted IDs remain available for recovery."""

    def __init__(self, message, *, created_memory_ids):
        self.created_memory_ids = list(created_memory_ids)
        super().__init__(message)


class GoodMemIndexingError(RuntimeError):
    """Indexing failed or timed out; resume waiting without uploading again."""

    def __init__(self, message, *, pending_memory_ids):
        self.pending_memory_ids = list(pending_memory_ids)
        super().__init__(message)


def _check_batch(results, batch, pending):
    errors, returned = [], set()
    for result in results:
        if not result.success or result.memory is None:
            errors.append(result.error.message if result.error else "Missing memory result")
            continue
        memory = result.memory
        returned.add(memory.memory_id)
        if memory.processing_status == "COMPLETED":
            pending.pop(memory.memory_id, None)
        elif memory.processing_status not in {"PENDING", "PROCESSING"}:
            errors.append(f"Memory {memory.memory_id}: indexing {memory.processing_status}")
    if errors or not set(batch).issubset(returned):
        raise GoodMemIndexingError(
            "; ".join(errors) or "Missing indexing status", pending_memory_ids=pending
        )


def _validate_wait(timeout, poll_interval):
    if timeout < 0 or poll_interval <= 0:
        raise ValueError("timeout must be nonnegative and poll_interval must be positive")


def wait_for_memories(
    client: Goodmem, memory_ids: Iterable[str], timeout: float, *, poll_interval: float = 1
) -> None:
    """Poll batches of known IDs; completed memories are removed from later polls.

    Args:
        client: Caller-owned synchronous SDK client.
        memory_ids: Accepted memory IDs to await.
        timeout: Total polling budget in seconds; HTTP timeout is configured separately.
        poll_interval: Delay between rounds of status requests.

    Returns:
        None once all requested memories have completed indexing.
    """
    _validate_wait(timeout, poll_interval)
    pending, deadline = dict.fromkeys(memory_ids), time.monotonic() + timeout
    first = True
    while pending:
        previous = list(pending)
        for offset in range(0, len(previous), 100):
            if not first and time.monotonic() >= deadline:
                raise GoodMemIndexingError("Indexing wait timed out", pending_memory_ids=pending)
            first = False
            batch = previous[offset : offset + 100]
            try:
                response = client.memories.batch_get(memory_ids=batch)
            except Exception as exc:
                raise GoodMemIndexingError(str(exc), pending_memory_ids=pending) from exc
            _check_batch(response.results, batch, pending)
        if pending:
            time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))


async def await_memories(
    client: AsyncGoodmem, memory_ids: Iterable[str], timeout: float, *, poll_interval: float = 1
) -> None:
    """Native async counterpart of wait_for_memories, with the same total budget."""
    _validate_wait(timeout, poll_interval)
    pending, deadline = dict.fromkeys(memory_ids), time.monotonic() + timeout
    first = True
    while pending:
        previous = list(pending)
        for offset in range(0, len(previous), 100):
            if not first and time.monotonic() >= deadline:
                raise GoodMemIndexingError("Indexing wait timed out", pending_memory_ids=pending)
            first = False
            batch = previous[offset : offset + 100]
            try:
                response = await client.memories.batch_get(memory_ids=batch)
            except Exception as exc:
                raise GoodMemIndexingError(str(exc), pending_memory_ids=pending) from exc
            _check_batch(response.results, batch, pending)
        if pending:
            await asyncio.sleep(min(poll_interval, max(0, deadline - time.monotonic())))


class GoodMemDocumentIngestor:
    """Batch Documents into a configured space using the official SDK.

    Args:
        space_id: Target space; GoodMem owns its chunking and embeddings.
        batch_size: Maximum Documents in one write request, from 1 to 100.
        connection: SDK clients or connection options accepted by Connection.

    Document UUIDs become memory IDs. Other IDs map deterministically within the
    space. Duplicate writes raise a conflict rather than updating or silently duplicating a document.
    """

    def __init__(self, *, space_id: str, batch_size: int = 100, **connection: Any) -> None:
        if not space_id or not 1 <= batch_size <= 100:
            raise ValueError("space_id is required and batch_size must be from 1 to 100")
        self.space_id, self.batch_size = space_id, batch_size
        self._connection = Connection(**connection)

    def _requests(self, documents):
        requests = []
        for document in documents:
            if not isinstance(document, Document):
                raise TypeError("add_documents requires LlamaIndex Documents")
            text = document.get_content(MetadataMode.NONE)
            if not text.strip():
                raise ValueError("Documents must contain nonempty text")
            try:
                memory_id = str(uuid.UUID(document.id_))
            except ValueError:
                memory_id = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"goodmem:{self.space_id}:{document.id_}")
                )
            requests.append(
                MemoryCreationRequest(
                    memory_id=memory_id,
                    space_id=self.space_id,
                    original_content=text,
                    content_type="text/plain",
                    metadata=document_metadata(document),
                    original_content_ref=document.metadata.get("source"),
                )
            )
        return requests

    def _accepted(self, response, batch, ids):
        ids.extend(r.memory.memory_id for r in response.results if r.success and r.memory)
        if len(response.results) != len(batch) or any(
            not r.success or not r.memory for r in response.results
        ):
            errors = "; ".join(r.error.message for r in response.results if r.error)
            raise GoodMemIngestionError(
                errors or "Incomplete batch response", created_memory_ids=ids
            )

    def add_documents(self, documents: Iterable[Document]) -> list[str]:
        """Return accepted memory IDs immediately; use wait_for_memories to await indexing."""
        requests, ids = self._requests(documents), []
        if not requests:
            return []
        with self._connection.sync() as client:
            for offset in range(0, len(requests), self.batch_size):
                batch = requests[offset : offset + self.batch_size]
                try:
                    response = client.memories.batch_create(requests=batch)
                except Exception as exc:
                    if not ids:
                        raise
                    raise GoodMemIngestionError(str(exc), created_memory_ids=ids) from exc
                self._accepted(response, batch, ids)
        return ids

    async def aadd_documents(self, documents: Iterable[Document]) -> list[str]:
        """Native async Document ingestion, returning accepted IDs without polling."""
        requests, ids = self._requests(documents), []
        if not requests:
            return []
        async with self._connection.async_() as client:
            for offset in range(0, len(requests), self.batch_size):
                batch = requests[offset : offset + self.batch_size]
                try:
                    response = await client.memories.batch_create(requests=batch)
                except Exception as exc:
                    if not ids:
                        raise
                    raise GoodMemIngestionError(str(exc), created_memory_ids=ids) from exc
                self._accepted(response, batch, ids)
        return ids
