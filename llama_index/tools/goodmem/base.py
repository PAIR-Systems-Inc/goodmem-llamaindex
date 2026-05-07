"""GoodMem Tool Spec for LlamaIndex.

Provides tools for vector-based memory storage and semantic retrieval
using the GoodMem API. Exposes operations as LlamaIndex tools that can
be used with any LlamaIndex agent.
"""

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

# MIME type mapping for file uploads
_MIME_TYPES: Dict[str, str] = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "txt": "text/plain",
    "html": "text/html",
    "md": "text/markdown",
    "csv": "text/csv",
    "json": "application/json",
    "xml": "application/xml",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "ppt": "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _get_mime_type(extension: str) -> Optional[str]:
    """Get MIME type from file extension."""
    return _MIME_TYPES.get(extension.lower().lstrip("."))


def _build_memory_request_body(
    space_id: str,
    text_content: Optional[str] = None,
    file_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the JSON request body for creating a memory.

    Raises:
        FileNotFoundError: If file_path is given but does not exist.
        ValueError: If neither text_content nor file_path is provided.
    """
    request_body: Dict[str, Any] = {"spaceId": space_id}

    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lstrip(".")
        mime_type = _get_mime_type(extension) or "application/octet-stream"

        with open(path, "rb") as f:
            file_bytes = f.read()

        if mime_type.startswith("text/"):
            request_body["contentType"] = mime_type
            request_body["originalContent"] = file_bytes.decode("utf-8")
        else:
            request_body["contentType"] = mime_type
            request_body["originalContentB64"] = base64.b64encode(
                file_bytes
            ).decode("ascii")
    elif text_content:
        request_body["contentType"] = "text/plain"
        request_body["originalContent"] = text_content
    else:
        raise ValueError(
            "No content provided. Please provide a file_path or text_content."
        )

    if metadata and isinstance(metadata, dict) and len(metadata) > 0:
        request_body["metadata"] = metadata

    return request_body


def _build_retrieve_request_body(
    query: str,
    space_ids: List[str],
    max_results: int = 5,
    include_memory_definition: bool = True,
    reranker_id: Optional[str] = None,
    llm_id: Optional[str] = None,
    relevance_threshold: Optional[float] = None,
    llm_temperature: Optional[float] = None,
    chronological_resort: bool = False,
) -> Dict[str, Any]:
    """Build the JSON request body for memory retrieval.

    Raises:
        ValueError: If space_ids is empty.
    """
    if not space_ids:
        raise ValueError("At least one space_id must be provided.")

    space_keys = [{"spaceId": sid} for sid in space_ids if sid]

    request_body: Dict[str, Any] = {
        "message": query,
        "spaceKeys": space_keys,
        "requestedSize": max_results,
        "fetchMemory": include_memory_definition,
    }

    # Add post-processor config if reranker or LLM is specified
    if reranker_id or llm_id:
        config: Dict[str, Any] = {}
        if reranker_id:
            config["reranker_id"] = reranker_id
        if llm_id:
            config["llm_id"] = llm_id
        if relevance_threshold is not None:
            config["relevance_threshold"] = relevance_threshold
        if llm_temperature is not None:
            config["llm_temp"] = llm_temperature
        if max_results:
            config["max_results"] = max_results
        if chronological_resort:
            config["chronological_resort"] = True

        request_body["postProcessor"] = {
            "name": "com.goodmem.retrieval.postprocess.ChatPostProcessorFactory",
            "config": config,
        }

    return request_body


def _parse_ndjson_response(response_text: str) -> Dict[str, Any]:
    """Parse an ndjson/SSE retrieval response into structured data.

    Returns:
        Dict with keys: results (list of chunk dicts), memories,
        result_set_id, and optional abstract_reply.
    """
    results: List[Dict[str, Any]] = []
    memories: List[Dict[str, Any]] = []
    result_set_id = ""
    abstract_reply: Optional[Dict[str, Any]] = None

    lines = response_text.strip().split("\n")

    for line in lines:
        json_str = line.strip()
        if not json_str:
            continue

        # Handle SSE format
        if json_str.startswith("data:"):
            json_str = json_str[5:].strip()
        if json_str.startswith("event:") or json_str == "":
            continue

        try:
            item = json.loads(json_str)

            if "resultSetBoundary" in item:
                result_set_id = item["resultSetBoundary"].get(
                    "resultSetId", ""
                )
            elif "memoryDefinition" in item:
                memories.append(item["memoryDefinition"])
            elif "abstractReply" in item:
                abstract_reply = item["abstractReply"]
            elif "retrievedItem" in item:
                chunk_data = (
                    item["retrievedItem"].get("chunk", {}).get("chunk", {})
                )
                results.append(
                    {
                        "chunkId": chunk_data.get("chunkId"),
                        "chunkText": chunk_data.get("chunkText"),
                        "memoryId": chunk_data.get("memoryId"),
                        "relevanceScore": item["retrievedItem"]
                        .get("chunk", {})
                        .get("relevanceScore"),
                        "memoryIndex": item["retrievedItem"]
                        .get("chunk", {})
                        .get("memoryIndex"),
                    }
                )
        except json.JSONDecodeError:
            continue

    parsed: Dict[str, Any] = {
        "results": results,
        "memories": memories,
        "result_set_id": result_set_id,
    }
    if abstract_reply:
        parsed["abstract_reply"] = abstract_reply
    return parsed


def _chunks_to_documents(
    results: List[Dict[str, Any]],
    query: str,
    result_set_id: str,
    memories: List[Dict[str, Any]],
    abstract_reply: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Convert parsed retrieval chunks into LlamaIndex Document objects."""
    # Index memory definitions by ID for fast lookup
    memory_map: Dict[str, Dict[str, Any]] = {}
    for mem in memories:
        mid = mem.get("memoryId")
        if mid:
            memory_map[mid] = mem

    documents: List[Document] = []
    for chunk in results:
        meta: Dict[str, Any] = {
            "chunkId": chunk.get("chunkId"),
            "memoryId": chunk.get("memoryId"),
            "relevanceScore": chunk.get("relevanceScore"),
            "memoryIndex": chunk.get("memoryIndex"),
            "resultSetId": result_set_id,
            "query": query,
        }
        # Attach memory-level metadata if available
        mem_def = memory_map.get(chunk.get("memoryId", ""))
        if mem_def:
            meta["spaceId"] = mem_def.get("spaceId")
            meta["contentType"] = mem_def.get("contentType")
            meta["processingStatus"] = mem_def.get("processingStatus")

        if abstract_reply:
            meta["abstractReply"] = abstract_reply

        documents.append(
            Document(
                text=chunk.get("chunkText", ""),
                metadata=meta,
            )
        )
    return documents


class GoodMemToolSpec(BaseToolSpec):
    """GoodMem tool spec for LlamaIndex.

    Provides tools for creating spaces, storing memories (text and files),
    retrieving memories via semantic search, fetching individual memories,
    and deleting memories using the GoodMem API.

    Args:
        api_key: GoodMem API key for authentication (X-API-Key).
        base_url: Base URL of the GoodMem API server
            (e.g., https://api.goodmem.ai or http://localhost:8080).
        verify_ssl: Whether to verify SSL certificates. Defaults to True.
    """

    spec_functions = [
        ("list_embedders", "alist_embedders"),
        ("list_spaces", "alist_spaces"),
        ("get_space", "aget_space"),
        ("create_space", "acreate_space"),
        ("update_space", "aupdate_space"),
        ("delete_space", "adelete_space"),
        ("create_memory", "acreate_memory"),
        ("list_memories", "alist_memories"),
        ("retrieve_memories", "aretrieve_memories"),
        ("get_memory", "aget_memory"),
        ("delete_memory", "adelete_memory"),
    ]

    def __init__(
        self,
        api_key: str,
        base_url: str,
        verify_ssl: bool = True,
        timeout: float = 120.0,
    ) -> None:
        """Initialize with GoodMem API credentials.

        Args:
            api_key: GoodMem API key (X-API-Key).
            base_url: Base URL of the GoodMem API server.
            verify_ssl: Whether to verify SSL certificates. Defaults to True.
            timeout: Per-request timeout in seconds. Defaults to 120 because
                retrieve with an LLM post-processor can run for tens of
                seconds while the server generates an abstractReply.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.verify_ssl = verify_ssl
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        """Return common request headers."""
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # Sync HTTP helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make a synchronous HTTP request to the GoodMem API."""
        url = f"{self.base_url}{path}"
        headers = self._headers()
        if extra_headers:
            headers.update(extra_headers)
        with httpx.Client(verify=self.verify_ssl, timeout=self.timeout) as client:
            response = client.request(
                method=method,
                url=url,
                headers=headers,
                json=json_body,
                params=query_params,
            )
        response.raise_for_status()
        return response

    # ------------------------------------------------------------------
    # Async HTTP helpers
    # ------------------------------------------------------------------

    async def _arequest(
        self,
        method: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make an asynchronous HTTP request to the GoodMem API."""
        url = f"{self.base_url}{path}"
        headers = self._headers()
        if extra_headers:
            headers.update(extra_headers)
        async with httpx.AsyncClient(verify=self.verify_ssl, timeout=self.timeout) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=json_body,
                params=query_params,
            )
        response.raise_for_status()
        return response

    # ------------------------------------------------------------------
    # list_spaces
    # ------------------------------------------------------------------

    def list_spaces(self) -> List[Dict[str, Any]]:
        """List all available spaces in GoodMem.

        Returns a list of spaces, each containing spaceId, name, and
        configuration details. Useful for discovering existing spaces
        before creating memories or performing retrieval.

        Returns:
            List of space objects with their IDs, names, and configurations.
        """
        response = self._request("GET", "/v1/spaces")
        body = response.json()
        return body if isinstance(body, list) else body.get("spaces", [])

    async def alist_spaces(self) -> List[Dict[str, Any]]:
        """Async version of list_spaces."""
        response = await self._arequest("GET", "/v1/spaces")
        body = response.json()
        return body if isinstance(body, list) else body.get("spaces", [])

    # ------------------------------------------------------------------
    # list_embedders
    # ------------------------------------------------------------------

    def list_embedders(self) -> List[Dict[str, Any]]:
        """List all available embedder models in GoodMem.

        Returns a list of embedder models that can be used when creating
        spaces. Each embedder converts text into vector representations
        for similarity search.

        Returns:
            List of embedder objects with their IDs, names, and model identifiers.
        """
        response = self._request("GET", "/v1/embedders")
        body = response.json()
        return body if isinstance(body, list) else body.get("embedders", [])

    async def alist_embedders(self) -> List[Dict[str, Any]]:
        """Async version of list_embedders."""
        response = await self._arequest("GET", "/v1/embedders")
        body = response.json()
        return body if isinstance(body, list) else body.get("embedders", [])

    # ------------------------------------------------------------------
    # get_space
    # ------------------------------------------------------------------

    def get_space(self, space_id: str) -> Dict[str, Any]:
        """Fetch a single space by its ID.

        Args:
            space_id: The UUID of the space to fetch.

        Returns:
            The space metadata dict (spaceId, name, embedderIds, labels, ...).

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        response = self._request("GET", f"/v1/spaces/{space_id}")
        return response.json()

    async def aget_space(self, space_id: str) -> Dict[str, Any]:
        """Async version of get_space."""
        response = await self._arequest("GET", f"/v1/spaces/{space_id}")
        return response.json()

    # ------------------------------------------------------------------
    # create_space
    # ------------------------------------------------------------------

    def create_space(
        self,
        name: str,
        embedder_id: str,
        chunk_size: int = 256,
        chunk_overlap: int = 25,
        keep_strategy: str = "KEEP_END",
        length_measurement: str = "CHARACTER_COUNT",
    ) -> Dict[str, Any]:
        """Create a new space or reuse an existing one in GoodMem.

        A space is a logical container for organizing related memories,
        configured with embedders that convert text to vector embeddings.
        If a space with the given name already exists, its ID is returned
        instead of creating a duplicate.

        Args:
            name: A unique name for the space. If a space with this name
                already exists, its ID will be returned.
            embedder_id: The ID of the embedder model to use for this space.
                Use list_embedders() to discover available embedders.
            chunk_size: Number of characters per chunk when splitting
                documents. Defaults to 256.
            chunk_overlap: Number of overlapping characters between
                consecutive chunks. Defaults to 25.
            keep_strategy: Where to attach the separator when splitting.
                One of KEEP_END, KEEP_START, or DISCARD. Defaults to KEEP_END.
            length_measurement: How chunk size is measured. One of
                CHARACTER_COUNT or TOKEN_COUNT. Defaults to CHARACTER_COUNT.

        Returns:
            Dict with spaceId, name, and whether the space was reused
            or newly created.
        """
        # Check if a space with the same name already exists
        try:
            spaces = self.list_spaces()
            for space in spaces:
                if space.get("name") == name:
                    return {
                        "spaceId": space.get("spaceId"),
                        "name": space.get("name"),
                        "embedderId": embedder_id,
                        "message": "Space already exists, reusing existing space",
                        "reused": True,
                    }
        except Exception:
            pass  # If listing fails, proceed to create

        request_body = {
            "name": name,
            "spaceEmbedders": [
                {"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}
            ],
            "defaultChunkingConfig": {
                "recursive": {
                    "chunkSize": chunk_size,
                    "chunkOverlap": chunk_overlap,
                    "separators": ["\n\n", "\n", ". ", " ", ""],
                    "keepStrategy": keep_strategy,
                    "separatorIsRegex": False,
                    "lengthMeasurement": length_measurement,
                }
            },
        }

        response = self._request("POST", "/v1/spaces", json_body=request_body)
        body = response.json()
        return {
            "spaceId": body.get("spaceId"),
            "name": body.get("name"),
            "embedderId": embedder_id,
            "chunkingConfig": request_body["defaultChunkingConfig"],
            "message": "Space created successfully",
            "reused": False,
        }

    async def acreate_space(
        self,
        name: str,
        embedder_id: str,
        chunk_size: int = 256,
        chunk_overlap: int = 25,
        keep_strategy: str = "KEEP_END",
        length_measurement: str = "CHARACTER_COUNT",
    ) -> Dict[str, Any]:
        """Async version of create_space."""
        try:
            spaces = await self.alist_spaces()
            for space in spaces:
                if space.get("name") == name:
                    return {
                        "spaceId": space.get("spaceId"),
                        "name": space.get("name"),
                        "embedderId": embedder_id,
                        "message": "Space already exists, reusing existing space",
                        "reused": True,
                    }
        except Exception:
            pass

        request_body = {
            "name": name,
            "spaceEmbedders": [
                {"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}
            ],
            "defaultChunkingConfig": {
                "recursive": {
                    "chunkSize": chunk_size,
                    "chunkOverlap": chunk_overlap,
                    "separators": ["\n\n", "\n", ". ", " ", ""],
                    "keepStrategy": keep_strategy,
                    "separatorIsRegex": False,
                    "lengthMeasurement": length_measurement,
                }
            },
        }

        response = await self._arequest(
            "POST", "/v1/spaces", json_body=request_body
        )
        body = response.json()
        return {
            "spaceId": body.get("spaceId"),
            "name": body.get("name"),
            "embedderId": embedder_id,
            "chunkingConfig": request_body["defaultChunkingConfig"],
            "message": "Space created successfully",
            "reused": False,
        }

    # ------------------------------------------------------------------
    # update_space
    # ------------------------------------------------------------------

    def update_space(
        self,
        space_id: str,
        name: Optional[str] = None,
        public_read: Optional[bool] = None,
        replace_labels: Optional[Dict[str, str]] = None,
        merge_labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Update a space's name, public-read flag, or labels.

        Only fields explicitly passed (non-None) are sent. ``replace_labels``
        replaces the entire label set; ``merge_labels`` merges into the
        existing labels. They should not be combined.

        Args:
            space_id: The UUID of the space to update.
            name: New space name.
            public_read: Whether the space is publicly readable.
            replace_labels: Replace the full label dict.
            merge_labels: Merge these labels into the existing label dict.

        Returns:
            Updated space metadata dict.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        request_body: Dict[str, Any] = {}
        if name is not None:
            request_body["name"] = name
        if public_read is not None:
            request_body["publicRead"] = public_read
        if replace_labels is not None:
            request_body["replaceLabels"] = replace_labels
        if merge_labels is not None:
            request_body["mergeLabels"] = merge_labels

        response = self._request(
            "PUT", f"/v1/spaces/{space_id}", json_body=request_body
        )
        return response.json()

    async def aupdate_space(
        self,
        space_id: str,
        name: Optional[str] = None,
        public_read: Optional[bool] = None,
        replace_labels: Optional[Dict[str, str]] = None,
        merge_labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Async version of update_space."""
        request_body: Dict[str, Any] = {}
        if name is not None:
            request_body["name"] = name
        if public_read is not None:
            request_body["publicRead"] = public_read
        if replace_labels is not None:
            request_body["replaceLabels"] = replace_labels
        if merge_labels is not None:
            request_body["mergeLabels"] = merge_labels

        response = await self._arequest(
            "PUT", f"/v1/spaces/{space_id}", json_body=request_body
        )
        return response.json()

    # ------------------------------------------------------------------
    # delete_space
    # ------------------------------------------------------------------

    def delete_space(self, space_id: str) -> Dict[str, Any]:
        """Delete a space and cascade-delete all of its memories.

        Args:
            space_id: The UUID of the space to delete.

        Returns:
            Dict with spaceId and confirmation message.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        self._request("DELETE", f"/v1/spaces/{space_id}")
        return {
            "spaceId": space_id,
            "message": "Space deleted successfully",
        }

    async def adelete_space(self, space_id: str) -> Dict[str, Any]:
        """Async version of delete_space."""
        await self._arequest("DELETE", f"/v1/spaces/{space_id}")
        return {
            "spaceId": space_id,
            "message": "Space deleted successfully",
        }

    # ------------------------------------------------------------------
    # create_memory
    # ------------------------------------------------------------------

    def create_memory(
        self,
        space_id: str,
        text_content: Optional[str] = None,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a document as a new memory in a space.

        The memory is processed asynchronously -- chunked into searchable
        pieces and embedded into vectors. Accepts a file path or plain text.
        If both are provided, the file takes priority.

        Args:
            space_id: The ID of the space to store the memory in.
                Use list_spaces() to discover available spaces.
            text_content: Plain text content to store as memory.
            file_path: Path to a file to store as memory (PDF, DOCX,
                image, etc.). Content type is auto-detected from the
                file extension.
            metadata: Optional key-value metadata as a dict.

        Returns:
            Dict with memoryId, spaceId, processing status, and content type.

        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError: If neither text_content nor file_path is provided.
            httpx.HTTPStatusError: If the API request fails.
        """
        request_body = _build_memory_request_body(
            space_id, text_content, file_path, metadata
        )

        response = self._request("POST", "/v1/memories", json_body=request_body)
        body = response.json()
        return {
            "memoryId": body.get("memoryId"),
            "spaceId": body.get("spaceId"),
            "status": body.get("processingStatus", "PENDING"),
            "contentType": request_body["contentType"],
            "fileName": Path(file_path).name if file_path else None,
        }

    async def acreate_memory(
        self,
        space_id: str,
        text_content: Optional[str] = None,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async version of create_memory."""
        request_body = _build_memory_request_body(
            space_id, text_content, file_path, metadata
        )

        response = await self._arequest(
            "POST", "/v1/memories", json_body=request_body
        )
        body = response.json()
        return {
            "memoryId": body.get("memoryId"),
            "spaceId": body.get("spaceId"),
            "status": body.get("processingStatus", "PENDING"),
            "contentType": request_body["contentType"],
            "fileName": Path(file_path).name if file_path else None,
        }

    # ------------------------------------------------------------------
    # list_memories
    # ------------------------------------------------------------------

    def list_memories(
        self,
        space_id: str,
        max_results: Optional[int] = None,
        next_token: Optional[str] = None,
        status_filter: Optional[str] = None,
        include_content: bool = False,
        filter_expression: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Paginate memories in a space.

        Args:
            space_id: The UUID of the space whose memories to list.
            max_results: Maximum memories to return per page.
            next_token: Pagination cursor returned by a previous call.
            status_filter: One of PENDING, PROCESSING, COMPLETED, FAILED.
            include_content: Whether to include the original content blob.
            filter_expression: Server-side metadata filter expression.

        Returns:
            Dict with keys: spaceId, memories (list), totalMemories (int),
            nextToken (str | None).

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        params: Dict[str, Any] = {}
        if max_results is not None:
            params["maxResults"] = max_results
        if next_token is not None:
            params["nextToken"] = next_token
        if status_filter is not None:
            params["statusFilter"] = status_filter
        if include_content:
            params["includeContent"] = True
        if filter_expression is not None:
            params["filter"] = filter_expression

        response = self._request(
            "GET", f"/v1/spaces/{space_id}/memories", query_params=params
        )
        body = response.json()
        memories = (
            body if isinstance(body, list) else body.get("memories", [])
        )
        next_tok = (
            body.get("nextToken") if isinstance(body, dict) else None
        )
        return {
            "spaceId": space_id,
            "memories": memories,
            "totalMemories": len(memories),
            "nextToken": next_tok,
        }

    async def alist_memories(
        self,
        space_id: str,
        max_results: Optional[int] = None,
        next_token: Optional[str] = None,
        status_filter: Optional[str] = None,
        include_content: bool = False,
        filter_expression: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of list_memories."""
        params: Dict[str, Any] = {}
        if max_results is not None:
            params["maxResults"] = max_results
        if next_token is not None:
            params["nextToken"] = next_token
        if status_filter is not None:
            params["statusFilter"] = status_filter
        if include_content:
            params["includeContent"] = True
        if filter_expression is not None:
            params["filter"] = filter_expression

        response = await self._arequest(
            "GET", f"/v1/spaces/{space_id}/memories", query_params=params
        )
        body = response.json()
        memories = (
            body if isinstance(body, list) else body.get("memories", [])
        )
        next_tok = (
            body.get("nextToken") if isinstance(body, dict) else None
        )
        return {
            "spaceId": space_id,
            "memories": memories,
            "totalMemories": len(memories),
            "nextToken": next_tok,
        }

    # ------------------------------------------------------------------
    # retrieve_memories
    # ------------------------------------------------------------------

    def retrieve_memories(
        self,
        query: str,
        space_ids: List[str],
        max_results: int = 5,
        include_memory_definition: bool = True,
        wait_for_indexing: bool = True,
        reranker_id: Optional[str] = None,
        llm_id: Optional[str] = None,
        relevance_threshold: Optional[float] = None,
        llm_temperature: Optional[float] = None,
        chronological_resort: bool = False,
    ) -> List[Document]:
        """Perform similarity-based semantic retrieval across one or more spaces.

        Returns matching chunks as LlamaIndex Document objects ranked by
        relevance. Each Document's text contains the chunk content and its
        metadata includes chunkId, memoryId, relevanceScore, and other
        retrieval details. When wait_for_indexing is True, retries for
        up to 60 seconds when no results are found (useful when memories
        were just added and may still be processing).

        Args:
            query: A natural language query used to find semantically
                similar memory chunks.
            space_ids: List of space IDs to search across. Use
                list_spaces() to discover available spaces.
            max_results: Maximum number of results to return. Defaults to 5.
            include_memory_definition: Whether to fetch the full memory
                metadata alongside the matched chunks. Defaults to True.
            wait_for_indexing: Retry for up to 60 seconds when no results
                are found. Defaults to True.
            reranker_id: Optional reranker model ID to improve result ordering.
            llm_id: Optional LLM ID to generate contextual responses
                alongside retrieved chunks.
            relevance_threshold: Minimum score (0-1) for including results.
                Only used when reranker_id or llm_id is set.
            llm_temperature: Creativity setting for LLM generation (0-2).
                Only used when llm_id is set.
            chronological_resort: Reorder results by creation time instead
                of relevance score. Defaults to False.

        Returns:
            List of Document objects, one per matched chunk. Each document's
            metadata contains chunkId, memoryId, relevanceScore, resultSetId,
            and query.

        Raises:
            ValueError: If space_ids is empty.
            httpx.HTTPStatusError: If the API request fails.
        """
        request_body = _build_retrieve_request_body(
            query=query,
            space_ids=space_ids,
            max_results=max_results,
            include_memory_definition=include_memory_definition,
            reranker_id=reranker_id,
            llm_id=llm_id,
            relevance_threshold=relevance_threshold,
            llm_temperature=llm_temperature,
            chronological_resort=chronological_resort,
        )

        max_wait_ms = 60000
        poll_interval_ms = 5000
        should_wait = wait_for_indexing
        start_time = time.time() * 1000

        while True:
            retrieve_headers = {"Accept": "application/x-ndjson"}
            response = self._request(
                "POST",
                "/v1/memories:retrieve",
                json_body=request_body,
                extra_headers=retrieve_headers,
            )

            parsed = _parse_ndjson_response(response.text)

            if len(parsed["results"]) > 0 or not should_wait:
                return _chunks_to_documents(
                    results=parsed["results"],
                    query=query,
                    result_set_id=parsed["result_set_id"],
                    memories=parsed["memories"],
                    abstract_reply=parsed.get("abstract_reply"),
                )

            elapsed = time.time() * 1000 - start_time
            if elapsed >= max_wait_ms:
                return _chunks_to_documents(
                    results=parsed["results"],
                    query=query,
                    result_set_id=parsed["result_set_id"],
                    memories=parsed["memories"],
                    abstract_reply=parsed.get("abstract_reply"),
                )

            time.sleep(poll_interval_ms / 1000)

    async def aretrieve_memories(
        self,
        query: str,
        space_ids: List[str],
        max_results: int = 5,
        include_memory_definition: bool = True,
        wait_for_indexing: bool = True,
        reranker_id: Optional[str] = None,
        llm_id: Optional[str] = None,
        relevance_threshold: Optional[float] = None,
        llm_temperature: Optional[float] = None,
        chronological_resort: bool = False,
    ) -> List[Document]:
        """Async version of retrieve_memories."""
        request_body = _build_retrieve_request_body(
            query=query,
            space_ids=space_ids,
            max_results=max_results,
            include_memory_definition=include_memory_definition,
            reranker_id=reranker_id,
            llm_id=llm_id,
            relevance_threshold=relevance_threshold,
            llm_temperature=llm_temperature,
            chronological_resort=chronological_resort,
        )

        max_wait_ms = 60000
        poll_interval_ms = 5000
        should_wait = wait_for_indexing
        start_time = time.time() * 1000

        while True:
            retrieve_headers = {"Accept": "application/x-ndjson"}
            response = await self._arequest(
                "POST",
                "/v1/memories:retrieve",
                json_body=request_body,
                extra_headers=retrieve_headers,
            )

            parsed = _parse_ndjson_response(response.text)

            if len(parsed["results"]) > 0 or not should_wait:
                return _chunks_to_documents(
                    results=parsed["results"],
                    query=query,
                    result_set_id=parsed["result_set_id"],
                    memories=parsed["memories"],
                    abstract_reply=parsed.get("abstract_reply"),
                )

            elapsed = time.time() * 1000 - start_time
            if elapsed >= max_wait_ms:
                return _chunks_to_documents(
                    results=parsed["results"],
                    query=query,
                    result_set_id=parsed["result_set_id"],
                    memories=parsed["memories"],
                    abstract_reply=parsed.get("abstract_reply"),
                )

            await asyncio.sleep(poll_interval_ms / 1000)

    # ------------------------------------------------------------------
    # get_memory
    # ------------------------------------------------------------------

    def get_memory(
        self,
        memory_id: str,
        include_content: bool = True,
    ) -> Dict[str, Any]:
        """Fetch a specific memory record by its ID.

        Returns memory metadata, processing status, and optionally the
        original content.

        Args:
            memory_id: The UUID of the memory to fetch (returned by
                create_memory).
            include_content: Whether to fetch the original document content
                in addition to metadata. Defaults to True.

        Returns:
            Dict with memory metadata and optionally the original content.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        response = self._request("GET", f"/v1/memories/{memory_id}")
        result: Dict[str, Any] = {"memory": response.json()}

        if include_content:
            content_response = self._request(
                "GET", f"/v1/memories/{memory_id}/content"
            )
            result["content"] = content_response.json()

        return result

    async def aget_memory(
        self,
        memory_id: str,
        include_content: bool = True,
    ) -> Dict[str, Any]:
        """Async version of get_memory."""
        response = await self._arequest("GET", f"/v1/memories/{memory_id}")
        result: Dict[str, Any] = {"memory": response.json()}

        if include_content:
            content_response = await self._arequest(
                "GET", f"/v1/memories/{memory_id}/content"
            )
            result["content"] = content_response.json()

        return result

    # ------------------------------------------------------------------
    # delete_memory
    # ------------------------------------------------------------------

    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Permanently delete a memory and its associated chunks and vector embeddings.

        Args:
            memory_id: The UUID of the memory to delete (returned by
                create_memory).

        Returns:
            Dict with memoryId and confirmation message.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        self._request("DELETE", f"/v1/memories/{memory_id}")
        return {
            "memoryId": memory_id,
            "message": "Memory deleted successfully",
        }

    async def adelete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Async version of delete_memory."""
        await self._arequest("DELETE", f"/v1/memories/{memory_id}")
        return {
            "memoryId": memory_id,
            "message": "Memory deleted successfully",
        }
