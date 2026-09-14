"""Opt-in live checks. Each fixture creates and cleans up its own space."""

import os
import uuid

import pytest
from goodmem import AsyncGoodmem, Goodmem
from goodmem.errors import ConflictError
from llama_index.core.schema import Document
from llama_index.core.tools import RetrieverTool
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.core.vector_stores.utils import build_metadata_filter_fn

from llama_index.tools.goodmem import (
    GoodMemDocumentIngestor,
    GoodMemRetrievalError,
    GoodMemRetriever,
    GoodMemToolSpec,
    await_memories,
    wait_for_memories,
)

from .test_filter_semantics import FILTERS

pytestmark = pytest.mark.skipif(
    not (
        os.getenv("GOODMEM_BASE_URL")
        and os.getenv("GOODMEM_API_KEY")
        and os.getenv("GOODMEM_EMBEDDER_ID")
    ),
    reason="Set GOODMEM_BASE_URL, GOODMEM_API_KEY and GOODMEM_EMBEDDER_ID for live tests",
)


@pytest.fixture
async def live():
    options = {"base_url": os.getenv("GOODMEM_BASE_URL"), "api_key": os.getenv("GOODMEM_API_KEY")}
    with Goodmem(**options) as client:
        space = client.spaces.create(
            name="llamaindex-it-" + uuid.uuid4().hex,
            space_embedders=[{"embedder_id": os.environ["GOODMEM_EMBEDDER_ID"]}],
        )
        try:
            async with AsyncGoodmem(**options) as async_client:
                yield client, async_client, space
        finally:
            client.spaces.delete(id=space.space_id)


def test_live_tools_write_then_read_text_and_duplicate_conflict(live):
    client, _, space = live
    spec = GoodMemToolSpec(client=client)
    note = spec.create_memory(
        space.space_id,
        "The deployment project is called Copper Finch.",
        {"source": "https://example.org/project"},
    )
    fetched = spec.get_memory(note["memory_id"])
    assert fetched["content"] == "The deployment project is called Copper Finch."
    assert any(s["space_id"] == space.space_id for s in spec.list_spaces(name_filter=space.name))
    assert any(m["memory_id"] == note["memory_id"] for m in spec.list_memories(space.space_id))
    assert spec.get_space(space.space_id)["space_id"] == space.space_id
    updated = spec.update_space(space.space_id, labels={"application": "llamaindex-test"})
    assert updated["labels"]["application"] == "llamaindex-test"
    with pytest.raises(ConflictError):
        spec.create_space(space.name, os.environ["GOODMEM_EMBEDDER_ID"])
    result = spec.retrieve_memories("What is the deployment project called?", [space.space_id])
    assert result["chunks"] and "Copper Finch" in result["chunks"][0]["text"]
    spec.delete_memory(note["memory_id"])


async def test_live_async_documents_and_native_metadata_filters(live):
    _, client, space = live
    values = ["O'Brien", "literal\\n", "line\nbreak", "x') OR TRUE --", "ordinary"]
    documents = [
        Document(
            text="The project deployment checklist covers routing and recovery.",
            metadata={"source": f"https://example.org/tenant/{i}", "tenant": value, "rank": i},
        )
        for i, value in enumerate(values)
    ]
    ids = await GoodMemDocumentIngestor(
        async_client=client, space_id=space.space_id
    ).aadd_documents(documents)
    await await_memories(client, ids, 120)
    for i, value in enumerate(values):
        retriever = GoodMemRetriever(
            async_client=client,
            space_ids=[space.space_id],
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="tenant", value=value),
                    MetadataFilter(key="rank", operator=">=", value=i),
                ]
            ),
        )
        tool = RetrieverTool.from_defaults(
            retriever, name="tenant_docs", description="Search this tenant"
        )
        result = await tool.acall(input="project deployment")
        assert result.raw_output
        assert {n.metadata["tenant"] for n in result.raw_output} == {value}
        assert all(n.score == -n.raw_score for n in result.raw_output)
    for operator, expected in [("in", {0, 2}), ("nin", {1, 3, 4})]:
        nodes = await GoodMemRetriever(
            async_client=client,
            space_ids=[space.space_id],
            filters=MetadataFilters(
                filters=[MetadataFilter(key="rank", operator=operator, value=[0, 2])]
            ),
        ).aretrieve("project deployment")
        assert {n.metadata["rank"] for n in nodes} == expected


def test_live_reranking_and_partial_failures(live):
    client, _, space = live
    ids = GoodMemDocumentIngestor(client=client, space_id=space.space_id).add_documents(
        [
            Document(
                text="StateGraph represents a workflow as nodes and edges.",
                metadata={"source": "https://example.org/graph"},
            )
        ]
    )
    wait_for_memories(client, ids, 120)
    reranker = os.getenv("GOODMEM_RERANKER_ID")
    if reranker:
        nodes = GoodMemRetriever(
            client=client, space_ids=[space.space_id], reranker_id=reranker
        ).retrieve("What is StateGraph?")
        assert nodes and nodes[0].metadata["source"] == "https://example.org/graph"
        assert nodes[0].score == nodes[0].raw_score
        assert nodes[0].score_kind == "reranker"
    invalid = str(uuid.uuid4())
    result = GoodMemToolSpec(client=client).retrieve_memories(
        "StateGraph", [space.space_id], reranker_id=invalid
    )
    assert result["partial"] and result["statuses"] and result["chunks"]
    with pytest.raises(GoodMemRetrievalError):
        GoodMemRetriever(client=client, space_ids=[space.space_id], reranker_id=invalid).retrieve(
            "StateGraph"
        )


async def test_live_exclusions_and_null_filter_semantics(live):
    _, client, space = live
    rows = [
        {},
        {"deleted": None, "rank": None},
        {"deleted": "yes", "rank": 1},
        {"deleted": "no", "rank": 0},
        {"rank": 2},
    ]
    documents = [
        Document(
            text="The project deployment checklist covers routing and recovery.",
            metadata=row | {"case": index, "internal_note": "HIDDEN_FROM_LLM"},
            excluded_llm_metadata_keys=["internal_note"],
        )
        for index, row in enumerate(rows)
    ]
    ids = await GoodMemDocumentIngestor(
        async_client=client, space_id=space.space_id
    ).aadd_documents(documents)
    await await_memories(client, ids, 120)
    metadata = {str(i): doc.metadata for i, doc in enumerate(documents)}
    for filters in FILTERS:
        expected_fn = build_metadata_filter_fn(metadata.__getitem__, filters)
        expected = {i for i in range(len(rows)) if expected_fn(str(i))}
        tool = RetrieverTool.from_defaults(
            GoodMemRetriever(
                async_client=client, space_ids=[space.space_id], filters=filters, top_k=10
            )
        )
        result = await tool.acall(input="project deployment")
        assert {node.metadata["case"] for node in result.raw_output} == expected, filters
        assert "HIDDEN_FROM_LLM" not in result.content and "internal_note" not in result.content
