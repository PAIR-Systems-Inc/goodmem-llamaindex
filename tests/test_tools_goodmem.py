"""Framework and wire-contract regression cases for the 0.2 integration."""

import copy
import json
import uuid

import httpx
import pytest
from goodmem.errors import AuthenticationError, ConflictError
from llama_index.core.callbacks import CallbackManager, CBEventType, LlamaDebugHandler
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import Document, NodeRelationship, NodeWithScore
from llama_index.core.tools import RetrieverTool
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters

from llama_index.tools.goodmem import (
    GoodMemDocumentIngestor,
    GoodMemIndexingError,
    GoodMemIngestionError,
    GoodMemRetrievalError,
    GoodMemRetriever,
    GoodMemToolSpec,
    await_memories,
    wait_for_memories,
)
from llama_index.tools.goodmem.filters import filter_expression

from .conftest import CHUNK, MEMORY, SPACE, ndjson


def events(*extra):
    return ndjson(CHUNK, *extra, {"memoryDefinition": MEMORY})


@pytest.mark.parametrize("async_mode", [False, True])
async def test_native_retriever_join_scores_and_framework_sources(wire, async_mode):
    handler = LlamaDebugHandler(print_trace_on_end=False)
    retriever = GoodMemRetriever(
        client=wire.sdk,
        async_client=wire.asdk,
        space_ids=["space-1"],
        callback_manager=CallbackManager([handler]),
    )
    wire.responses.append(events(CHUNK))
    nodes = await retriever.aretrieve("evidence") if async_mode else retriever.retrieve("evidence")
    assert len(nodes) == 1 and isinstance(nodes[0], NodeWithScore)
    assert nodes[0].score == 0.82
    assert nodes[0].raw_score == -0.82
    assert nodes[0].node_id == "chunk-1"
    assert nodes[0].metadata["source"] == "https://example.org/docs"
    assert nodes[0].node.relationships[NodeRelationship.SOURCE].node_id == "memory-1"
    assert len(handler.get_event_pairs(CBEventType.RETRIEVE)) == 1
    assert not wire.http.is_closed and not wire.ahttp.is_closed


@pytest.mark.parametrize("async_mode", [False, True])
async def test_unknown_status_between_valid_chunks_does_not_abort(wire, async_mode):
    second = copy.deepcopy(CHUNK)
    second["retrievedItem"]["chunk"]["chunk"]["chunkId"] = "chunk-2"
    wire.responses.append(
        events({"status": {"code": "FUTURE_SERVER_NOTICE", "message": "New notice"}}, second)
    )
    retriever = GoodMemRetriever(client=wire.sdk, async_client=wire.asdk, space_ids=["space-1"])
    nodes = await retriever.aretrieve("evidence") if async_mode else retriever.retrieve("evidence")
    assert [n.node_id for n in nodes] == ["chunk-1", "chunk-2"]
    assert nodes[0].statuses[0]["code"] == "UNKNOWN"


@pytest.mark.parametrize(
    "code", ["RERANKING_FAILED", "VECTOR_SEARCH_PARTIAL", "SUMMARIZATION_FAILED"]
)
def test_known_failure_raises_for_retriever_but_tool_retains_chunks(wire, code):
    failure = {"status": {"code": code, "message": "A requested stage failed"}}
    wire.responses.extend([events(failure), events(failure)])
    with pytest.raises(GoodMemRetrievalError, match=code):
        GoodMemRetriever(client=wire.sdk, space_ids=["space-1"]).retrieve("evidence")
    result = GoodMemToolSpec(client=wire.sdk).retrieve_memories("evidence", ["space-1"])
    assert result["partial"] is True and result["chunks"][0]["text"] == "Retrieved evidence"
    assert result["statuses"][0]["code"] == code


@pytest.mark.parametrize("async_mode", [False, True])
async def test_empty_search_is_one_request(wire, async_mode):
    wire.responses.append(ndjson())
    retriever = GoodMemRetriever(client=wire.sdk, async_client=wire.asdk, space_ids=["space-1"])
    nodes = await retriever.aretrieve("nothing") if async_mode else retriever.retrieve("nothing")
    assert nodes == [] and len(wire.requests) == 1


def test_reranking_fetches_extra_candidates_without_llm(wire):
    wire.responses.append(
        events(
            {
                "status": {
                    "code": "FEATURE_DISABLED",
                    "message": "Summarization disabled",
                    "details": {"feature": "summarization", "required_param": "llm_id"},
                }
            }
        )
    )
    retriever = GoodMemRetriever(
        client=wire.sdk, space_ids=["space-1"], reranker_id="reranker-1", top_k=5
    )
    assert retriever.retrieve("evidence")
    body = json.loads(wire.requests[0].content)
    assert body["requestedSize"] == 20
    assert body["postProcessor"]["config"]["reranker_id"] == "reranker-1"
    assert body["postProcessor"]["config"]["max_results"] == 5
    assert not body["postProcessor"]["config"].get("llm_id")


def test_native_tool_scopes_and_query_engine(wire):
    retriever = GoodMemRetriever(
        client=wire.sdk,
        space_ids=["space-1"],
        filters=MetadataFilters(filters=[MetadataFilter(key="tenant", value="O'Brien")]),
    )
    tool = RetrieverTool.from_defaults(
        retriever, name="policies", description="Search policy documents"
    )
    assert set(tool.metadata.get_parameters_dict()["properties"]) == {"input"}
    wire.responses.extend([events(), events()])
    output = tool.call(input="policy")
    assert "https://example.org/docs" in output.content
    assert "_goodmem" not in output.content
    assert isinstance(output.raw_output[0], NodeWithScore)
    result = RetrieverQueryEngine.from_args(retriever, llm=MockLLM()).query("policy")
    assert result.source_nodes[0].node_id == "chunk-1"
    body = json.loads(wire.requests[0].content)
    assert [s["spaceId"] for s in body["spaceKeys"]] == ["space-1"]
    assert "tenant" in body["spaceKeys"][0]["filter"]


def test_metadata_collisions_preserve_original_maps(wire):
    memory = copy.deepcopy(MEMORY)
    memory["metadata"]["_goodmem"] = "user value"
    memory["metadata"]["shared"] = "memory"
    chunk = copy.deepcopy(CHUNK)
    chunk["retrievedItem"]["chunk"]["chunk"]["metadata"] = {"shared": "chunk"}
    wire.responses.append(ndjson({"memoryDefinition": memory}, chunk))
    node = GoodMemRetriever(client=wire.sdk, space_ids=["space-1"]).retrieve("evidence")[0]
    assert node.metadata["shared"] == "chunk"
    assert node.metadata["_goodmem"]["memory_metadata"]["_goodmem"] == "user value"
    assert node.metadata["_goodmem"]["memory_metadata"]["shared"] == "memory"


def test_source_reference_without_metadata(wire):
    memory = MEMORY | {"metadata": {}, "originalContentRef": "https://example.org/reference"}
    wire.responses.append(ndjson(CHUNK, {"memoryDefinition": memory}))
    assert (
        GoodMemRetriever(client=wire.sdk, space_ids=["space-1"])
        .retrieve("evidence")[0]
        .metadata["source"]
        .endswith("/reference")
    )


@pytest.mark.parametrize("async_mode", [False, True])
async def test_readable_text_and_content_error(wire, async_mode):
    spec = GoodMemToolSpec(client=wire.sdk, async_client=wire.asdk)
    wire.responses.extend(
        [httpx.Response(200, json=MEMORY), httpx.Response(200, text="Stored note")]
    )
    result = await spec.aget_memory("memory-1") if async_mode else spec.get_memory("memory-1")
    assert result["content"] == "Stored note"
    wire.responses.extend(
        [
            httpx.Response(200, json=MEMORY),
            httpx.Response(404, json={"message": "Content unavailable"}),
        ]
    )
    result = await spec.aget_memory("memory-1") if async_mode else spec.get_memory("memory-1")
    assert result["memory"]["memory_id"] == "memory-1" and result["content_error"]


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ConnectError])
async def test_custom_client_transport_failure_retains_memory_metadata(
    wire, async_mode, error_type
):
    wire.responses.extend(
        [
            httpx.Response(200, json=MEMORY),
            error_type("Content transport unavailable"),
        ]
    )
    spec = GoodMemToolSpec(client=wire.sdk, async_client=wire.asdk)
    result = await spec.aget_memory("memory-1") if async_mode else spec.get_memory("memory-1")
    assert result["memory"]["memory_id"] == "memory-1"
    assert result["memory"]["metadata"] == MEMORY["metadata"]
    assert result["content_error"] == "Content transport unavailable"
    assert "content" not in result
    assert len(wire.requests) == 2


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("transport_failure", [False, True])
async def test_get_memory_initial_lookup_failure_is_not_hidden(wire, async_mode, transport_failure):
    wire.responses.append(
        httpx.ReadTimeout("Metadata unavailable")
        if transport_failure
        else httpx.Response(401, json={"message": "Unauthorized"})
    )
    spec = GoodMemToolSpec(client=wire.sdk, async_client=wire.asdk)
    with pytest.raises(httpx.ReadTimeout if transport_failure else AuthenticationError):
        if async_mode:
            await spec.aget_memory("memory-1")
        else:
            spec.get_memory("memory-1")
    assert len(wire.requests) == 1


@pytest.mark.parametrize("async_mode", [False, True])
async def test_list_spaces_follows_pages(wire, async_mode):
    spec = GoodMemToolSpec(client=wire.sdk, async_client=wire.asdk)
    wire.responses.extend(
        [
            httpx.Response(200, json={"spaces": [SPACE], "nextToken": "page2"}),
            httpx.Response(200, json={"spaces": [SPACE | {"spaceId": "space-2"}]}),
        ]
    )
    spaces = await spec.alist_spaces() if async_mode else spec.list_spaces()
    assert [s["space_id"] for s in spaces] == ["space-1", "space-2"]
    assert wire.requests[1].url.params["next_token"] == "page2"


def test_duplicate_space_preserves_sdk_conflict(wire):
    wire.responses.append(httpx.Response(409, json={"message": "Duplicate name"}))
    with pytest.raises(ConflictError):
        GoodMemToolSpec(client=wire.sdk).create_space("Docs", "embedder-1")
    assert len(wire.requests) == 1 and wire.requests[0].method == "POST"


def test_tool_schema_removes_obsolete_options_and_file_access(wire, tmp_path):
    tools = {t.metadata.name: t for t in GoodMemToolSpec(client=wire.sdk).to_tool_list()}
    assert len(tools) == 11 and "upload_memory" not in tools
    assert "public_read" not in tools["update_space"].metadata.get_parameters_dict()["properties"]
    assert (
        "wait_for_indexing"
        not in tools["retrieve_memories"].metadata.get_parameters_dict()["properties"]
    )
    assert set(tools["create_space"].metadata.get_parameters_dict()["properties"]) == {
        "name",
        "embedder_id",
    }
    allowed = tmp_path / "uploads"
    allowed.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (allowed / "link.txt").symlink_to(outside)
    spec = GoodMemToolSpec(client=wire.sdk, upload_directory=allowed)
    for name in ["../outside.txt", "link.txt", str(outside)]:
        with pytest.raises(ValueError, match="inside"):
            spec.upload_memory("space-1", name)
    assert len(spec.to_tool_list()) == 12


@pytest.mark.parametrize("async_mode", [False, True])
async def test_document_ingestion_roundtrips_metadata_without_polling(wire, async_mode):
    doc = Document(
        id_=str(uuid.uuid4()),
        text="Whole document",
        metadata={"source": "https://example.org/doc", "author": "Ada"},
    )
    response = MEMORY | {"memoryId": doc.id_, "processingStatus": "PENDING"}
    wire.responses.append(
        httpx.Response(200, json={"results": [{"success": True, "memory": response}]})
    )
    ingestor = GoodMemDocumentIngestor(client=wire.sdk, async_client=wire.asdk, space_id="space-1")
    ids = await ingestor.aadd_documents([doc]) if async_mode else ingestor.add_documents([doc])
    assert ids == [doc.id_] and len(wire.requests) == 1
    request = json.loads(wire.requests[0].content)["requests"][0]
    assert all(request["metadata"][key] == value for key, value in doc.metadata.items())
    assert request["originalContent"] == "Whole document"
    assert request["originalContentRef"] == doc.metadata["source"]


def test_partial_write_retains_receipts_and_initial_http_error_type(wire):
    docs = [Document(text="first"), Document(text="second")]
    wire.responses.append(httpx.Response(401, json={"message": "Unauthorized"}))
    ingestor = GoodMemDocumentIngestor(client=wire.sdk, space_id="space-1", batch_size=1)
    with pytest.raises(AuthenticationError):
        ingestor.add_documents(docs)
    wire.responses.extend(
        [
            httpx.Response(200, json={"results": [{"success": True, "memory": MEMORY}]}),
            httpx.Response(401, json={"message": "Unauthorized"}),
        ]
    )
    with pytest.raises(GoodMemIngestionError) as error:
        ingestor.add_documents(docs)
    assert error.value.created_memory_ids == ["memory-1"]
    assert isinstance(error.value.__cause__, AuthenticationError)


@pytest.mark.parametrize("async_mode", [False, True])
async def test_batch_wait_only_requests_pending_ids(wire, async_mode):
    pending = MEMORY | {"memoryId": "memory-2", "processingStatus": "PROCESSING"}
    wire.responses.extend(
        [
            httpx.Response(
                200,
                json={
                    "results": [
                        {"success": True, "memory": MEMORY},
                        {"success": True, "memory": pending},
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "results": [
                        {"success": True, "memory": pending | {"processingStatus": "COMPLETED"}}
                    ]
                },
            ),
        ]
    )
    if async_mode:
        await await_memories(wire.asdk, ["memory-1", "memory-2"], 1, poll_interval=0.001)
    else:
        wait_for_memories(wire.sdk, ["memory-1", "memory-2"], 1, poll_interval=0.001)
    assert json.loads(wire.requests[1].content)["memoryIds"] == ["memory-2"]
    assert all(r.url.path.endswith(":batchGet") for r in wire.requests)


def test_timeout_exposes_pending_ids(wire):
    wire.responses.append(
        httpx.Response(
            200,
            json={
                "results": [
                    {"success": True, "memory": MEMORY | {"processingStatus": "PROCESSING"}}
                ]
            },
        )
    )
    with pytest.raises(GoodMemIndexingError) as error:
        wait_for_memories(wire.sdk, ["memory-1"], 0)
    assert error.value.pending_memory_ids == ["memory-1"]


@pytest.mark.parametrize(
    "operator", [FilterOperator.ANY, FilterOperator.CONTAINS, FilterOperator.IS_EMPTY]
)
def test_unsupported_filters_fail_before_http(operator):
    with pytest.raises(ValueError, match="Unsupported"):
        GoodMemRetriever(
            space_ids=["space-1"],
            filters=MetadataFilters(
                filters=[MetadataFilter(key="tags", value="a", operator=operator)]
            ),
        )


def test_empty_filters_and_nested_numeric_filters():
    assert filter_expression(MetadataFilters(filters=[])) is None
    expr = filter_expression(
        MetadataFilters(
            filters=[
                MetadataFilter(key="age", operator=">=", value=18),
                MetadataFilters(
                    condition="or",
                    filters=[
                        MetadataFilter(key="name", value="O'Brien"),
                        MetadataFilter(key="name", value="Ada"),
                    ],
                ),
            ]
        )
    )
    assert "NUMERIC) >= 18" in expr and " OR " in expr and "O\\'Brien" in expr


def test_filter_key_injection_is_rejected():
    with pytest.raises(ValueError, match="top-level"):
        filter_expression(MetadataFilter(key="x') OR TRUE", value="a"))


@pytest.mark.parametrize("async_mode", [False, True])
async def test_raw_tool_unknown_status_is_partial_with_chunks(wire, async_mode):
    wire.responses.append(
        events({"status": {"code": "NEW_SERVER_NOTICE", "message": "A new notice"}})
    )
    spec = GoodMemToolSpec(client=wire.sdk, async_client=wire.asdk)
    result = (
        await spec.aretrieve_memories("evidence", ["space-1"])
        if async_mode
        else spec.retrieve_memories("evidence", ["space-1"])
    )
    assert result["chunks"][0]["text"] == "Retrieved evidence"
    assert result["partial"] and result["statuses"][0]["code"] == "UNKNOWN"


@pytest.mark.parametrize("async_mode", [False, True])
async def test_malformed_stream_is_not_a_successful_empty_search(wire, async_mode):
    from goodmem.errors import GoodMemError

    wire.responses.append(
        httpx.Response(200, text="not JSON", headers={"content-type": "application/x-ndjson"})
    )
    retriever = GoodMemRetriever(client=wire.sdk, async_client=wire.asdk, space_ids=["space-1"])
    with pytest.raises(GoodMemError):
        if async_mode:
            await retriever.aretrieve("evidence")
        else:
            retriever.retrieve("evidence")


@pytest.mark.parametrize("async_mode", [False, True])
async def test_update_space_uses_sdk_request_shape(wire, async_mode):
    wire.responses.append(httpx.Response(200, json=SPACE))
    spec = GoodMemToolSpec(client=wire.sdk, async_client=wire.asdk)
    if async_mode:
        await spec.aupdate_space("space-1", name="Renamed", labels={"owner": "team"})
    else:
        spec.update_space("space-1", name="Renamed", labels={"owner": "team"})
    assert json.loads(wire.requests[0].content) == {
        "name": "Renamed",
        "replaceLabels": {"owner": "team"},
    }


@pytest.mark.parametrize("async_mode", [False, True])
async def test_list_embedders_uses_sdk_list_response(wire, async_mode):
    wire.responses.append(httpx.Response(200, json={"embedders": []}))
    spec = GoodMemToolSpec(client=wire.sdk, async_client=wire.asdk)
    result = await spec.alist_embedders() if async_mode else spec.list_embedders()
    assert result == []


def test_failed_batch_wait_does_not_report_completed_memories_as_pending(wire):
    wire.responses.append(
        httpx.Response(
            200,
            json={
                "results": [
                    {"success": True, "memory": MEMORY},
                    {
                        "success": True,
                        "memory": MEMORY | {"memoryId": "memory-2", "processingStatus": "FAILED"},
                    },
                ]
            },
        )
    )
    with pytest.raises(GoodMemIndexingError) as error:
        wait_for_memories(wire.sdk, ["memory-1", "memory-2"], 5)
    assert error.value.pending_memory_ids == ["memory-2"]


async def test_empty_document_batch_needs_no_connection():
    ingestor = GoodMemDocumentIngestor(space_id="space-1")
    assert ingestor.add_documents([]) == []
    assert await ingestor.aadd_documents([]) == []


def test_a_newer_sdk_recognizing_a_new_notice_does_not_make_it_fatal():
    from goodmem.models.good_mem_status import GoodMemStatus
    from goodmem.models.retrieve_memory_event import RetrieveMemoryEvent

    from llama_index.tools.goodmem.retriever import retrieval_result

    # Simulate a future SDK that preserves a newly recognized code as a string.
    status = GoodMemStatus.model_construct(code="FUTURE_INFORMATION", message="New SDK notice")
    events_ = [
        RetrieveMemoryEvent.model_validate(CHUNK),
        RetrieveMemoryEvent(status=status),
        RetrieveMemoryEvent.model_validate({"memoryDefinition": MEMORY}),
    ]
    result = retrieval_result(events_, strict=True)
    assert result["nodes"][0].text == "Retrieved evidence"
    assert result["partial"] and result["statuses"][0]["code"] == "FUTURE_INFORMATION"
