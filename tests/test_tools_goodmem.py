"""Unit tests for GoodMem LlamaIndex tool spec.

Tests use mocked HTTP responses so they run without a live GoodMem server.
Follows LlamaIndex testing conventions: pytest functions, unittest.mock,
BaseToolSpec MRO check, and httpx-based mocking.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.goodmem import GoodMemToolSpec
from llama_index.tools.goodmem.base import (
    _build_memory_request_body,
    _build_retrieve_request_body,
    _chunks_to_documents,
    _get_mime_type,
    _parse_ndjson_response,
)


# ---------------------------------------------------------------------------
# Class & inheritance
# ---------------------------------------------------------------------------


def test_class():
    """GoodMemToolSpec inherits from BaseToolSpec (standard LlamaIndex check)."""
    names_of_base_classes = [b.__name__ for b in GoodMemToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    """All expected sync/async pairs are declared in spec_functions."""
    expected = [
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
    assert GoodMemToolSpec.spec_functions == expected


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_response(json_data=None, text=None, status_code=200):
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if json_data is not None:
        resp.json.return_value = json_data
    if text is not None:
        resp.text = text
    return resp


@pytest.fixture()
def tool():
    """Return a GoodMemToolSpec with _request mocked."""
    spec = GoodMemToolSpec(
        api_key="test-api-key",
        base_url="https://api.goodmem.test",
        verify_ssl=False,
    )
    spec._request = MagicMock()
    spec._arequest = AsyncMock()
    return spec


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_initialization():
    """Verify constructor stores credentials."""
    spec = GoodMemToolSpec(
        api_key="my-key",
        base_url="https://api.goodmem.test/",
        verify_ssl=False,
    )
    assert spec.api_key == "my-key"
    assert spec.base_url == "https://api.goodmem.test"  # trailing slash stripped
    assert spec.verify_ssl is False


def test_headers():
    """Headers include the API key and JSON content type."""
    spec = GoodMemToolSpec(api_key="k", base_url="https://x")
    headers = spec._headers()
    assert headers["X-API-Key"] == "k"
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "application/json"


# ---------------------------------------------------------------------------
# to_tool_list (LlamaIndex integration point)
# ---------------------------------------------------------------------------


def test_to_tool_list():
    """to_tool_list produces FunctionTools for every spec function pair."""
    spec = GoodMemToolSpec(api_key="k", base_url="https://x")
    tools = spec.to_tool_list()
    assert len(tools) == len(GoodMemToolSpec.spec_functions)
    tool_names = {t.metadata.name for t in tools}
    expected_names = {pair[0] for pair in GoodMemToolSpec.spec_functions}
    assert tool_names == expected_names


# ---------------------------------------------------------------------------
# list_embedders
# ---------------------------------------------------------------------------


def test_list_embedders(tool):
    """list_embedders returns parsed embedder list."""
    tool._request.return_value = _mock_response(
        json_data={"embedders": [{"embedderId": "emb-1", "name": "text-embedding-3-small"}]}
    )

    result = tool.list_embedders()

    assert len(result) == 1
    assert result[0]["embedderId"] == "emb-1"
    tool._request.assert_called_once_with("GET", "/v1/embedders")


def test_list_embedders_flat_response(tool):
    """list_embedders handles a flat list response (no wrapper object)."""
    tool._request.return_value = _mock_response(json_data=[{"embedderId": "emb-2"}])

    result = tool.list_embedders()
    assert result == [{"embedderId": "emb-2"}]


@pytest.mark.asyncio
async def test_alist_embedders(tool):
    """Async list_embedders returns parsed embedder list."""
    tool._arequest.return_value = _mock_response(
        json_data={"embedders": [{"embedderId": "emb-1"}]}
    )

    result = await tool.alist_embedders()

    assert len(result) == 1
    assert result[0]["embedderId"] == "emb-1"
    tool._arequest.assert_called_once_with("GET", "/v1/embedders")


# ---------------------------------------------------------------------------
# list_spaces
# ---------------------------------------------------------------------------


def test_list_spaces(tool):
    """list_spaces returns parsed space list."""
    tool._request.return_value = _mock_response(
        json_data={"spaces": [{"spaceId": "sp-1", "name": "My Space"}]}
    )

    result = tool.list_spaces()

    assert len(result) == 1
    assert result[0]["spaceId"] == "sp-1"


def test_list_spaces_flat_response(tool):
    """list_spaces handles a flat list response."""
    tool._request.return_value = _mock_response(
        json_data=[{"spaceId": "sp-2", "name": "Flat"}]
    )

    result = tool.list_spaces()
    assert result[0]["name"] == "Flat"


@pytest.mark.asyncio
async def test_alist_spaces(tool):
    """Async list_spaces returns parsed space list."""
    tool._arequest.return_value = _mock_response(
        json_data={"spaces": [{"spaceId": "sp-1", "name": "My Space"}]}
    )

    result = await tool.alist_spaces()
    assert result[0]["spaceId"] == "sp-1"


# ---------------------------------------------------------------------------
# create_space
# ---------------------------------------------------------------------------


def test_create_space_new(tool):
    """create_space creates a new space when none exists with the same name."""
    list_resp = _mock_response(json_data={"spaces": []})
    create_resp = _mock_response(json_data={"spaceId": "sp-new", "name": "Test Space"})

    tool._request.side_effect = [list_resp, create_resp]

    result = tool.create_space(name="Test Space", embedder_id="emb-1")

    assert result["spaceId"] == "sp-new"
    assert result["reused"] is False
    assert result["message"] == "Space created successfully"


def test_create_space_reuse(tool):
    """create_space reuses existing space if name matches."""
    tool._request.return_value = _mock_response(
        json_data={"spaces": [{"spaceId": "sp-existing", "name": "Reuse Me"}]}
    )

    result = tool.create_space(name="Reuse Me", embedder_id="emb-1")

    assert result["spaceId"] == "sp-existing"
    assert result["reused"] is True


def test_create_space_post_failure(tool):
    """create_space raises on HTTP failure (exceptions propagate)."""
    list_resp = _mock_response(json_data={"spaces": []})
    tool._request.side_effect = [
        list_resp,
        httpx.HTTPStatusError("Server Error", request=MagicMock(), response=MagicMock()),
    ]

    with pytest.raises(httpx.HTTPStatusError):
        tool.create_space(name="Fail Space", embedder_id="emb-1")


@pytest.mark.asyncio
async def test_acreate_space_new(tool):
    """Async create_space creates a new space."""
    list_resp = _mock_response(json_data={"spaces": []})
    create_resp = _mock_response(json_data={"spaceId": "sp-new", "name": "Async Space"})

    tool._arequest.side_effect = [list_resp, create_resp]

    result = await tool.acreate_space(name="Async Space", embedder_id="emb-1")

    assert result["spaceId"] == "sp-new"
    assert result["reused"] is False


# ---------------------------------------------------------------------------
# create_memory – text
# ---------------------------------------------------------------------------


def test_create_memory_text(tool):
    """create_memory with plain text content."""
    tool._request.return_value = _mock_response(
        json_data={"memoryId": "mem-1", "spaceId": "sp-1", "processingStatus": "PENDING"}
    )

    result = tool.create_memory(space_id="sp-1", text_content="Hello world")

    assert result["memoryId"] == "mem-1"
    assert result["contentType"] == "text/plain"
    assert result["fileName"] is None

    # Verify the request body
    call_args = tool._request.call_args
    body = call_args[1]["json_body"]
    assert body["spaceId"] == "sp-1"
    assert body["originalContent"] == "Hello world"
    assert body["contentType"] == "text/plain"


def test_create_memory_no_content(tool):
    """create_memory raises ValueError when no content is provided."""
    with pytest.raises(ValueError, match="No content provided"):
        tool.create_memory(space_id="sp-1")


def test_create_memory_with_metadata(tool):
    """create_memory includes metadata in request body."""
    tool._request.return_value = _mock_response(
        json_data={"memoryId": "mem-2", "spaceId": "sp-1"}
    )

    tool.create_memory(
        space_id="sp-1",
        text_content="Test",
        metadata={"author": "pytest"},
    )

    body = tool._request.call_args[1]["json_body"]
    assert body["metadata"] == {"author": "pytest"}


# ---------------------------------------------------------------------------
# create_memory – file
# ---------------------------------------------------------------------------


def test_create_memory_file_text(tool, tmp_path):
    """create_memory with a text file reads content as UTF-8."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("File content here")

    tool._request.return_value = _mock_response(
        json_data={"memoryId": "mem-f1", "spaceId": "sp-1"}
    )

    result = tool.create_memory(space_id="sp-1", file_path=str(txt_file))

    assert result["contentType"] == "text/plain"
    assert result["fileName"] == "notes.txt"

    body = tool._request.call_args[1]["json_body"]
    assert body["originalContent"] == "File content here"
    assert "originalContentB64" not in body


def test_create_memory_file_pdf(tool, tmp_path):
    """create_memory with a PDF file sends base64-encoded content."""
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake content")

    tool._request.return_value = _mock_response(
        json_data={"memoryId": "mem-f2", "spaceId": "sp-1"}
    )

    result = tool.create_memory(space_id="sp-1", file_path=str(pdf_file))

    assert result["contentType"] == "application/pdf"

    body = tool._request.call_args[1]["json_body"]
    assert "originalContentB64" in body
    assert "originalContent" not in body


def test_create_memory_file_not_found(tool):
    """create_memory raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        tool.create_memory(space_id="sp-1", file_path="/nonexistent/file.pdf")


@pytest.mark.asyncio
async def test_acreate_memory_text(tool):
    """Async create_memory with plain text content."""
    tool._arequest.return_value = _mock_response(
        json_data={"memoryId": "mem-a1", "spaceId": "sp-1", "processingStatus": "PENDING"}
    )

    result = await tool.acreate_memory(space_id="sp-1", text_content="Async hello")

    assert result["memoryId"] == "mem-a1"
    assert result["contentType"] == "text/plain"


# ---------------------------------------------------------------------------
# get_memory
# ---------------------------------------------------------------------------


def test_get_memory(tool):
    """get_memory returns memory metadata and content."""
    meta_resp = _mock_response(
        json_data={"memoryId": "mem-1", "spaceId": "sp-1", "processingStatus": "COMPLETED"}
    )
    content_resp = _mock_response(json_data={"originalContent": "Hello"})

    tool._request.side_effect = [meta_resp, content_resp]

    result = tool.get_memory(memory_id="mem-1", include_content=True)

    assert result["memory"]["memoryId"] == "mem-1"
    assert result["content"]["originalContent"] == "Hello"


def test_get_memory_without_content(tool):
    """get_memory skips content fetch when include_content=False."""
    tool._request.return_value = _mock_response(json_data={"memoryId": "mem-1"})

    result = tool.get_memory(memory_id="mem-1", include_content=False)

    assert "content" not in result
    assert tool._request.call_count == 1


def test_get_memory_not_found(tool):
    """get_memory raises on 404."""
    tool._request.side_effect = httpx.HTTPStatusError(
        "Not Found", request=MagicMock(), response=MagicMock()
    )

    with pytest.raises(httpx.HTTPStatusError):
        tool.get_memory(memory_id="bad-id")


@pytest.mark.asyncio
async def test_aget_memory(tool):
    """Async get_memory returns memory metadata and content."""
    meta_resp = _mock_response(json_data={"memoryId": "mem-1"})
    content_resp = _mock_response(json_data={"originalContent": "Async"})

    tool._arequest.side_effect = [meta_resp, content_resp]

    result = await tool.aget_memory(memory_id="mem-1", include_content=True)

    assert result["memory"]["memoryId"] == "mem-1"
    assert result["content"]["originalContent"] == "Async"


# ---------------------------------------------------------------------------
# delete_memory
# ---------------------------------------------------------------------------


def test_delete_memory(tool):
    """delete_memory returns confirmation dict."""
    tool._request.return_value = _mock_response()

    result = tool.delete_memory(memory_id="mem-del")

    assert result["memoryId"] == "mem-del"
    assert result["message"] == "Memory deleted successfully"
    tool._request.assert_called_once_with("DELETE", "/v1/memories/mem-del")


def test_delete_memory_failure(tool):
    """delete_memory raises on HTTP failure."""
    tool._request.side_effect = httpx.HTTPStatusError(
        "Not Found", request=MagicMock(), response=MagicMock()
    )

    with pytest.raises(httpx.HTTPStatusError):
        tool.delete_memory(memory_id="bad-id")


@pytest.mark.asyncio
async def test_adelete_memory(tool):
    """Async delete_memory returns confirmation dict."""
    tool._arequest.return_value = _mock_response()

    result = await tool.adelete_memory(memory_id="mem-del")

    assert result["memoryId"] == "mem-del"
    tool._arequest.assert_called_once_with("DELETE", "/v1/memories/mem-del")


# ---------------------------------------------------------------------------
# retrieve_memories – returns List[Document]
# ---------------------------------------------------------------------------


def _make_ndjson_response(*items):
    """Helper: build an ndjson text body from a list of dicts."""
    return "\n".join(json.dumps(item) for item in items)


def test_retrieve_memories_returns_documents(tool):
    """retrieve_memories returns a list of LlamaIndex Document objects."""
    ndjson = _make_ndjson_response(
        {"resultSetBoundary": {"resultSetId": "rs-1"}},
        {
            "retrievedItem": {
                "chunk": {
                    "chunk": {
                        "chunkId": "c-1",
                        "chunkText": "AI in healthcare",
                        "memoryId": "mem-1",
                    },
                    "relevanceScore": 0.95,
                    "memoryIndex": 0,
                }
            }
        },
        {
            "memoryDefinition": {
                "memoryId": "mem-1",
                "spaceId": "sp-1",
                "contentType": "text/plain",
            }
        },
    )
    tool._request.return_value = _mock_response(text=ndjson)

    docs = tool.retrieve_memories(
        query="AI healthcare",
        space_ids=["sp-1"],
        max_results=5,
        wait_for_indexing=False,
    )

    assert isinstance(docs, list)
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].text == "AI in healthcare"
    assert docs[0].metadata["chunkId"] == "c-1"
    assert docs[0].metadata["memoryId"] == "mem-1"
    assert docs[0].metadata["relevanceScore"] == 0.95
    assert docs[0].metadata["resultSetId"] == "rs-1"
    assert docs[0].metadata["query"] == "AI healthcare"
    assert docs[0].metadata["spaceId"] == "sp-1"


def test_retrieve_memories_with_abstract_reply(tool):
    """retrieve_memories attaches abstractReply to document metadata."""
    ndjson = _make_ndjson_response(
        {"resultSetBoundary": {"resultSetId": "rs-2"}},
        {
            "retrievedItem": {
                "chunk": {
                    "chunk": {
                        "chunkId": "c-2",
                        "chunkText": "chunk text",
                        "memoryId": "mem-2",
                    },
                    "relevanceScore": 0.9,
                }
            }
        },
        {"abstractReply": {"text": "Summary answer", "model": "gpt-4"}},
    )
    tool._request.return_value = _mock_response(text=ndjson)

    docs = tool.retrieve_memories(
        query="question",
        space_ids=["sp-1"],
        wait_for_indexing=False,
    )

    assert len(docs) == 1
    assert docs[0].metadata["abstractReply"]["text"] == "Summary answer"


def test_retrieve_memories_empty_space_ids(tool):
    """retrieve_memories raises ValueError on empty space_ids."""
    with pytest.raises(ValueError, match="At least one space_id"):
        tool.retrieve_memories(query="test", space_ids=[])


def test_retrieve_memories_no_results_no_wait(tool):
    """retrieve_memories returns empty list when no results and wait is off."""
    ndjson = _make_ndjson_response(
        {"resultSetBoundary": {"resultSetId": "rs-3"}},
    )
    tool._request.return_value = _mock_response(text=ndjson)

    docs = tool.retrieve_memories(
        query="nothing here",
        space_ids=["sp-1"],
        wait_for_indexing=False,
    )

    assert docs == []


def test_retrieve_memories_sse_format(tool):
    """retrieve_memories handles SSE-prefixed lines (data: ...)."""
    sse_body = (
        'data: {"resultSetBoundary": {"resultSetId": "rs-sse"}}\n'
        'data: {"retrievedItem": {"chunk": {"chunk": '
        '{"chunkId": "c-sse", "chunkText": "sse chunk", "memoryId": "mem-sse"}, '
        '"relevanceScore": 0.88}}}\n'
    )
    tool._request.return_value = _mock_response(text=sse_body)

    docs = tool.retrieve_memories(
        query="sse test",
        space_ids=["sp-1"],
        wait_for_indexing=False,
    )

    assert len(docs) == 1
    assert docs[0].text == "sse chunk"


def test_retrieve_memories_http_error(tool):
    """retrieve_memories raises on HTTP failure."""
    tool._request.side_effect = httpx.HTTPStatusError(
        "Connection refused", request=MagicMock(), response=MagicMock()
    )

    with pytest.raises(httpx.HTTPStatusError):
        tool.retrieve_memories(
            query="q",
            space_ids=["sp-1"],
            wait_for_indexing=False,
        )


@pytest.mark.asyncio
async def test_aretrieve_memories(tool):
    """Async retrieve_memories returns Document objects."""
    ndjson = _make_ndjson_response(
        {"resultSetBoundary": {"resultSetId": "rs-a"}},
        {
            "retrievedItem": {
                "chunk": {
                    "chunk": {
                        "chunkId": "c-a",
                        "chunkText": "async chunk",
                        "memoryId": "mem-a",
                    },
                    "relevanceScore": 0.92,
                }
            }
        },
    )
    tool._arequest.return_value = _mock_response(text=ndjson)

    docs = await tool.aretrieve_memories(
        query="async query",
        space_ids=["sp-1"],
        wait_for_indexing=False,
    )

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].text == "async chunk"


# ---------------------------------------------------------------------------
# get_space
# ---------------------------------------------------------------------------


def test_get_space(tool):
    """get_space returns the parsed space dict."""
    tool._request.return_value = _mock_response(
        json_data={"spaceId": "sp-1", "name": "Hello"}
    )

    result = tool.get_space(space_id="sp-1")

    assert result["spaceId"] == "sp-1"
    assert result["name"] == "Hello"
    tool._request.assert_called_once_with("GET", "/v1/spaces/sp-1")


@pytest.mark.asyncio
async def test_aget_space(tool):
    """Async get_space returns the parsed space dict."""
    tool._arequest.return_value = _mock_response(
        json_data={"spaceId": "sp-1", "name": "Async"}
    )

    result = await tool.aget_space(space_id="sp-1")
    assert result["spaceId"] == "sp-1"


# ---------------------------------------------------------------------------
# update_space
# ---------------------------------------------------------------------------


def test_update_space_name_only(tool):
    """update_space sends only the fields explicitly passed."""
    tool._request.return_value = _mock_response(
        json_data={"spaceId": "sp-1", "name": "renamed"}
    )

    result = tool.update_space(space_id="sp-1", name="renamed")

    assert result["name"] == "renamed"
    body = tool._request.call_args[1]["json_body"]
    assert body == {"name": "renamed"}


def test_update_space_all_fields(tool):
    """update_space passes publicRead and labels through correctly."""
    tool._request.return_value = _mock_response(
        json_data={"spaceId": "sp-1"}
    )

    tool.update_space(
        space_id="sp-1",
        name="x",
        public_read=True,
        replace_labels={"a": "b"},
        merge_labels={"c": "d"},
    )

    body = tool._request.call_args[1]["json_body"]
    assert body["name"] == "x"
    assert body["publicRead"] is True
    assert body["replaceLabels"] == {"a": "b"}
    assert body["mergeLabels"] == {"c": "d"}


def test_update_space_no_fields(tool):
    """update_space with no fields sends an empty body (server may reject)."""
    tool._request.return_value = _mock_response(json_data={"spaceId": "sp-1"})

    tool.update_space(space_id="sp-1")
    body = tool._request.call_args[1]["json_body"]
    assert body == {}


@pytest.mark.asyncio
async def test_aupdate_space(tool):
    """Async update_space passes through fields."""
    tool._arequest.return_value = _mock_response(
        json_data={"spaceId": "sp-1", "name": "y"}
    )

    result = await tool.aupdate_space(space_id="sp-1", name="y")
    assert result["name"] == "y"


# ---------------------------------------------------------------------------
# delete_space
# ---------------------------------------------------------------------------


def test_delete_space(tool):
    """delete_space returns confirmation dict."""
    tool._request.return_value = _mock_response()

    result = tool.delete_space(space_id="sp-del")

    assert result["spaceId"] == "sp-del"
    assert result["message"] == "Space deleted successfully"
    tool._request.assert_called_once_with("DELETE", "/v1/spaces/sp-del")


def test_delete_space_failure(tool):
    """delete_space raises on HTTP failure."""
    tool._request.side_effect = httpx.HTTPStatusError(
        "Not Found", request=MagicMock(), response=MagicMock()
    )

    with pytest.raises(httpx.HTTPStatusError):
        tool.delete_space(space_id="bad-id")


@pytest.mark.asyncio
async def test_adelete_space(tool):
    """Async delete_space returns confirmation."""
    tool._arequest.return_value = _mock_response()

    result = await tool.adelete_space(space_id="sp-del")
    assert result["spaceId"] == "sp-del"


# ---------------------------------------------------------------------------
# list_memories
# ---------------------------------------------------------------------------


def test_list_memories_basic(tool):
    """list_memories returns a paginated dict with memory list."""
    tool._request.return_value = _mock_response(
        json_data={
            "memories": [{"memoryId": "m-1"}, {"memoryId": "m-2"}],
            "nextToken": "tok-1",
        }
    )

    result = tool.list_memories(space_id="sp-1")

    assert result["spaceId"] == "sp-1"
    assert result["totalMemories"] == 2
    assert result["memories"][0]["memoryId"] == "m-1"
    assert result["nextToken"] == "tok-1"
    call = tool._request.call_args
    assert call[0] == ("GET", "/v1/spaces/sp-1/memories")
    # No filters → empty params
    assert call[1]["query_params"] == {}


def test_list_memories_with_filters(tool):
    """list_memories forwards every filter as a query param."""
    tool._request.return_value = _mock_response(
        json_data={"memories": [], "nextToken": None}
    )

    tool.list_memories(
        space_id="sp-1",
        max_results=10,
        next_token="abc",
        status_filter="COMPLETED",
        include_content=True,
        filter_expression="key=value",
    )

    params = tool._request.call_args[1]["query_params"]
    assert params == {
        "maxResults": 10,
        "nextToken": "abc",
        "statusFilter": "COMPLETED",
        "includeContent": True,
        "filter": "key=value",
    }


def test_list_memories_flat_response(tool):
    """list_memories handles a flat-list response shape."""
    tool._request.return_value = _mock_response(
        json_data=[{"memoryId": "m-3"}]
    )

    result = tool.list_memories(space_id="sp-1")
    assert result["totalMemories"] == 1
    assert result["nextToken"] is None


@pytest.mark.asyncio
async def test_alist_memories(tool):
    """Async list_memories returns memories."""
    tool._arequest.return_value = _mock_response(
        json_data={"memories": [{"memoryId": "m-a"}], "nextToken": None}
    )

    result = await tool.alist_memories(space_id="sp-1")
    assert result["totalMemories"] == 1


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def test_mime_type_mapping():
    """_get_mime_type resolves common extensions."""
    assert _get_mime_type("pdf") == "application/pdf"
    assert _get_mime_type(".png") == "image/png"
    assert _get_mime_type("JPEG") == "image/jpeg"
    assert _get_mime_type("txt") == "text/plain"
    assert _get_mime_type("unknown") is None


def test_build_memory_request_body_text():
    """_build_memory_request_body creates text body correctly."""
    body = _build_memory_request_body("sp-1", text_content="hello")
    assert body["spaceId"] == "sp-1"
    assert body["contentType"] == "text/plain"
    assert body["originalContent"] == "hello"


def test_build_memory_request_body_no_content():
    """_build_memory_request_body raises ValueError with no content."""
    with pytest.raises(ValueError, match="No content provided"):
        _build_memory_request_body("sp-1")


def test_build_memory_request_body_missing_file():
    """_build_memory_request_body raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        _build_memory_request_body("sp-1", file_path="/no/such/file.pdf")


def test_build_retrieve_request_body():
    """_build_retrieve_request_body creates correct body."""
    body = _build_retrieve_request_body("query", ["sp-1"], max_results=3)
    assert body["message"] == "query"
    assert body["spaceKeys"] == [{"spaceId": "sp-1"}]
    assert body["requestedSize"] == 3


def test_build_retrieve_request_body_empty_spaces():
    """_build_retrieve_request_body raises ValueError."""
    with pytest.raises(ValueError, match="At least one space_id"):
        _build_retrieve_request_body("query", [])


def test_build_retrieve_request_body_with_post_processor():
    """_build_retrieve_request_body includes postProcessor config."""
    body = _build_retrieve_request_body(
        "q", ["sp-1"],
        reranker_id="rerank-1",
        llm_id="llm-1",
        relevance_threshold=0.5,
        llm_temperature=0.7,
    )
    assert "postProcessor" in body
    config = body["postProcessor"]["config"]
    assert config["reranker_id"] == "rerank-1"
    assert config["llm_id"] == "llm-1"
    assert config["relevance_threshold"] == 0.5
    assert config["llm_temp"] == 0.7


def test_parse_ndjson_response():
    """_parse_ndjson_response parses all item types."""
    ndjson = _make_ndjson_response(
        {"resultSetBoundary": {"resultSetId": "rs-1"}},
        {"retrievedItem": {"chunk": {"chunk": {"chunkId": "c-1", "chunkText": "t"}, "relevanceScore": 0.9}}},
        {"memoryDefinition": {"memoryId": "mem-1"}},
        {"abstractReply": {"text": "answer"}},
    )
    parsed = _parse_ndjson_response(ndjson)
    assert parsed["result_set_id"] == "rs-1"
    assert len(parsed["results"]) == 1
    assert len(parsed["memories"]) == 1
    assert parsed["abstract_reply"]["text"] == "answer"


def test_chunks_to_documents():
    """_chunks_to_documents converts chunks to Document objects."""
    docs = _chunks_to_documents(
        results=[{"chunkId": "c-1", "chunkText": "hello", "memoryId": "mem-1", "relevanceScore": 0.9}],
        query="test",
        result_set_id="rs-1",
        memories=[{"memoryId": "mem-1", "spaceId": "sp-1"}],
    )
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].text == "hello"
    assert docs[0].metadata["spaceId"] == "sp-1"
    assert docs[0].metadata["resultSetId"] == "rs-1"
