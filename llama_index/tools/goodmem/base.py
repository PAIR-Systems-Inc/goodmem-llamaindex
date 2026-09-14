"""Optional administrative tools backed by the official GoodMem SDK."""

from email.message import Message
from pathlib import Path
from typing import Any

import httpx
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from goodmem.errors import GoodMemError

from ._connection import Connection
from .ingestion import GoodMemIndexingError, await_memories, wait_for_memories
from .retriever import GoodMemRetriever, retrieval_result


def _dump(model):
    return model.model_dump(mode="json", exclude_none=True)


def _unconfirmed_write(memory, error):
    """Return an accepted write receipt without claiming a current indexing state."""
    return {
        "memory_id": memory.memory_id,
        "space_id": memory.space_id,
        "accepted": True,
        "indexing": {"status": "unconfirmed", "error": str(error)},
        "next_step": (
            "Call get_memory with this memory_id to check indexing. "
            "Creation succeeded; do not upload this memory again."
        ),
    }


def _content(data, content_type):
    mime = Message()
    mime["content-type"] = content_type
    if mime.get_content_maintype() == "text" or mime.get_content_subtype() in {"json", "xml"}:
        try:
            return {"content": data.decode(mime.get_content_charset() or "utf-8")}
        except (UnicodeError, LookupError):
            return {"content_notice": "Original content could not be decoded as text"}
    return {"content_notice": "Binary content omitted; use the SDK to download original bytes"}


def _compact(events, limit):
    result = retrieval_result(events)
    result["chunks"] = [
        {
            "text": n.text,
            "source": n.metadata.get("source"),
            "score": n.score,
            "memory_id": n.metadata["_goodmem"]["memory_id"],
            "chunk_id": n.node_id,
            "space_id": n.metadata["_goodmem"]["space_id"],
        }
        for n in result.pop("nodes")[:limit]
    ]
    return result


class GoodMemToolSpec(BaseToolSpec):
    """Manage spaces and memories with synchronous and native async agent tools.

    Prefer GoodMemRetriever with LlamaIndex's RetrieverTool for scoped RAG search.
    These administrative tools expose resource IDs to the model.

    Args:
        upload_directory: Optional directory the upload tool may read. Uploads are
            absent from the tool list unless this directory is configured.
        indexing_timeout: Per-write readiness budget for the create-memory tool.
        connection: SDK clients or connection options accepted by Connection.
    """

    spec_functions = [
        (name, "a" + name)
        for name in (
            "list_embedders",
            "list_spaces",
            "get_space",
            "create_space",
            "update_space",
            "delete_space",
            "create_memory",
            "list_memories",
            "retrieve_memories",
            "get_memory",
            "delete_memory",
        )
    ]

    def __init__(self, *, upload_directory=None, indexing_timeout=60, **connection):
        if indexing_timeout < 0:
            raise ValueError("indexing_timeout must be nonnegative")
        self._connection = Connection(**connection)
        self._indexing_timeout = indexing_timeout
        self._upload_directory = (
            Path(upload_directory).resolve(strict=True) if upload_directory else None
        )
        if self._upload_directory and not self._upload_directory.is_dir():
            raise ValueError("upload_directory must be a directory")
        self.spec_functions = list(type(self).spec_functions)
        if self._upload_directory:
            self.spec_functions.append(("upload_memory", "aupload_memory"))

    def list_embedders(self, max_items: int = 100) -> list[dict[str, Any]]:
        """List configured embedding models, traversing pages up to max_items."""
        with self._connection.sync() as client:
            return [_dump(m) for m in client.embedders.list()[:max_items]]

    async def alist_embedders(self, max_items: int = 100) -> list[dict[str, Any]]:
        """List embedding models asynchronously."""
        async with self._connection.async_() as client:
            return [_dump(m) for m in (await client.embedders.list())[:max_items]]

    def list_spaces(
        self, name_filter: str | None = None, max_items: int = 100
    ) -> list[dict[str, Any]]:
        """List spaces, optionally filtering by name, across pages up to max_items."""
        with self._connection.sync() as client:
            return [
                _dump(s) for s in client.spaces.list(name_filter=name_filter, max_items=max_items)
            ]

    async def alist_spaces(
        self, name_filter: str | None = None, max_items: int = 100
    ) -> list[dict[str, Any]]:
        """List matching spaces asynchronously."""
        async with self._connection.async_() as client:
            return [
                _dump(s)
                async for s in await client.spaces.list(
                    name_filter=name_filter, max_items=max_items
                )
            ]

    def get_space(self, space_id: str) -> dict[str, Any]:
        """Fetch a space's metadata and configuration by UUID."""
        with self._connection.sync() as client:
            return _dump(client.spaces.get(id=space_id))

    async def aget_space(self, space_id: str) -> dict[str, Any]:
        """Fetch a space asynchronously."""
        async with self._connection.async_() as client:
            return _dump(await client.spaces.get(id=space_id))

    def create_space(self, name: str, embedder_id: str) -> dict[str, Any]:
        """Create a new space with server defaults; duplicate names raise a conflict."""
        with self._connection.sync() as client:
            return _dump(
                client.spaces.create(name=name, space_embedders=[{"embedder_id": embedder_id}])
            )

    async def acreate_space(self, name: str, embedder_id: str) -> dict[str, Any]:
        """Create a space asynchronously, using server chunking defaults."""
        async with self._connection.async_() as client:
            return _dump(
                await client.spaces.create(
                    name=name, space_embedders=[{"embedder_id": embedder_id}]
                )
            )

    def update_space(
        self, space_id: str, name: str | None = None, labels: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Rename a space or replace its labels."""
        with self._connection.sync() as client:
            return _dump(
                client.spaces.update(
                    id=space_id,
                    request={
                        k: v
                        for k, v in {"name": name, "replace_labels": labels}.items()
                        if v is not None
                    },
                )
            )

    async def aupdate_space(
        self, space_id: str, name: str | None = None, labels: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Update a space asynchronously."""
        async with self._connection.async_() as client:
            return _dump(
                await client.spaces.update(
                    id=space_id,
                    request={
                        k: v
                        for k, v in {"name": name, "replace_labels": labels}.items()
                        if v is not None
                    },
                )
            )

    def delete_space(self, space_id: str) -> dict[str, str]:
        """Delete a space and all of its memories."""
        with self._connection.sync() as client:
            client.spaces.delete(id=space_id)
        return {"space_id": space_id}

    async def adelete_space(self, space_id: str) -> dict[str, str]:
        """Delete a space and its memories asynchronously."""
        async with self._connection.async_() as client:
            await client.spaces.delete(id=space_id)
        return {"space_id": space_id}

    def create_memory(
        self,
        space_id: str,
        text_content: str,
        metadata: dict[str, Any] | None = None,
        wait: bool = True,
    ) -> dict[str, Any]:
        """Store a note and await its indexing; return its accepted ID if waiting fails.

        An unconfirmed indexing result means creation succeeded. Use get_memory
        with the returned memory_id to check status without uploading again.
        """
        with self._connection.sync() as client:
            memory = client.memories.create(
                space_id=space_id, original_content=text_content, metadata=metadata
            )
            if wait:
                try:
                    wait_for_memories(client, [memory.memory_id], self._indexing_timeout)
                except GoodMemIndexingError as exc:
                    return _unconfirmed_write(memory, exc)
                memory.processing_status = "COMPLETED"
            return _dump(memory)

    async def acreate_memory(
        self,
        space_id: str,
        text_content: str,
        metadata: dict[str, Any] | None = None,
        wait: bool = True,
    ) -> dict[str, Any]:
        """Store a note asynchronously, retaining its accepted ID if indexing waiting fails."""
        async with self._connection.async_() as client:
            memory = await client.memories.create(
                space_id=space_id, original_content=text_content, metadata=metadata
            )
            if wait:
                try:
                    await await_memories(client, [memory.memory_id], self._indexing_timeout)
                except GoodMemIndexingError as exc:
                    return _unconfirmed_write(memory, exc)
                memory.processing_status = "COMPLETED"
            return _dump(memory)

    def _upload_path(self, file_path):
        if self._upload_directory is None:
            raise ValueError("File uploads require an upload_directory")
        path = (self._upload_directory / file_path).resolve(strict=True)
        if not path.is_relative_to(self._upload_directory) or not path.is_file():
            raise ValueError("File must be inside the configured upload_directory")
        return str(path)

    def upload_memory(
        self, space_id: str, file_path: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Upload a file from the application-approved directory; returns an accepted memory ID."""
        path = self._upload_path(file_path)
        with self._connection.sync() as client:
            return _dump(
                client.memories.create(space_id=space_id, file_path=path, metadata=metadata)
            )

    async def aupload_memory(
        self, space_id: str, file_path: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Upload an approved file asynchronously, without waiting for indexing."""
        path = self._upload_path(file_path)
        async with self._connection.async_() as client:
            return _dump(
                await client.memories.create(space_id=space_id, file_path=path, metadata=metadata)
            )

    def list_memories(
        self, space_id: str, filter_expression: str | None = None, max_items: int = 100
    ) -> list[dict[str, Any]]:
        """List memory metadata across pages, optionally using a native GoodMem filter."""
        with self._connection.sync() as client:
            return [
                _dump(m)
                for m in client.memories.list(
                    space_id=space_id, filter=filter_expression, max_items=max_items
                )
            ]

    async def alist_memories(
        self, space_id: str, filter_expression: str | None = None, max_items: int = 100
    ) -> list[dict[str, Any]]:
        """List matching memories asynchronously."""
        async with self._connection.async_() as client:
            return [
                _dump(m)
                async for m in await client.memories.list(
                    space_id=space_id, filter=filter_expression, max_items=max_items
                )
            ]

    def _search_options(
        self, query, space_ids, max_results, fetch_k, reranker_id, llm_id, filter_expression
    ):
        retriever = GoodMemRetriever(
            space_ids=space_ids,
            top_k=max_results,
            fetch_k=fetch_k,
            reranker_id=reranker_id,
            filter=filter_expression,
        )
        options = retriever._request(query)
        if llm_id:
            options.update(llm_id=llm_id, max_results=max_results)
        return options

    def retrieve_memories(
        self,
        query: str,
        space_ids: list[str],
        max_results: int = 5,
        fetch_k: int | None = None,
        reranker_id: str | None = None,
        llm_id: str | None = None,
        filter_expression: str | None = None,
    ) -> dict[str, Any]:
        """Search once, returning compact chunks, sources, statuses and a partial flag.

        Reranking needs no LLM. partial=true means diagnostics need attention;
        useful chunks are retained even when a post-processing stage fails.
        """
        options = self._search_options(
            query, space_ids, max_results, fetch_k, reranker_id, llm_id, filter_expression
        )
        with self._connection.sync() as client:
            return _compact(client.memories.retrieve(**options), max_results)

    async def aretrieve_memories(
        self,
        query: str,
        space_ids: list[str],
        max_results: int = 5,
        fetch_k: int | None = None,
        reranker_id: str | None = None,
        llm_id: str | None = None,
        filter_expression: str | None = None,
    ) -> dict[str, Any]:
        """Search asynchronously with the same compact diagnostic result."""
        options = self._search_options(
            query, space_ids, max_results, fetch_k, reranker_id, llm_id, filter_expression
        )
        async with self._connection.async_() as client:
            return _compact(await client.memories.retrieve(**options), max_results)

    def get_memory(self, memory_id: str, include_content: bool = True) -> dict[str, Any]:
        """Fetch memory metadata and readable original text; retain metadata if content is unavailable."""
        with self._connection.sync() as client:
            memory = client.memories.get(id=memory_id)
            result = {"memory": _dump(memory)}
            if include_content:
                try:
                    result.update(
                        _content(client.memories.content(id=memory_id), memory.content_type)
                    )
                except (GoodMemError, httpx.RequestError) as exc:
                    result["content_error"] = str(exc)
            return result

    async def aget_memory(self, memory_id: str, include_content: bool = True) -> dict[str, Any]:
        """Fetch memory metadata and readable content asynchronously."""
        async with self._connection.async_() as client:
            memory = await client.memories.get(id=memory_id)
            result = {"memory": _dump(memory)}
            if include_content:
                try:
                    result.update(
                        _content(await client.memories.content(id=memory_id), memory.content_type)
                    )
                except (GoodMemError, httpx.RequestError) as exc:
                    result["content_error"] = str(exc)
            return result

    def delete_memory(self, memory_id: str) -> dict[str, str]:
        """Delete a memory and its indexed chunks."""
        with self._connection.sync() as client:
            client.memories.delete(id=memory_id)
        return {"memory_id": memory_id}

    async def adelete_memory(self, memory_id: str) -> dict[str, str]:
        """Delete a memory asynchronously."""
        async with self._connection.async_() as client:
            await client.memories.delete(id=memory_id)
        return {"memory_id": memory_id}
