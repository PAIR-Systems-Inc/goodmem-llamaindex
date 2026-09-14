"""Document formatting restrictions survive storage and native tool retrieval."""

import copy
import json

import httpx
import pytest
from llama_index.core.schema import Document, MetadataMode
from llama_index.core.tools import RetrieverTool

from llama_index.tools.goodmem import (
    GoodMemDocumentIngestor,
    GoodMemNodeWithScore,
    GoodMemRetriever,
)

from .conftest import CHUNK, MEMORY, ndjson


@pytest.mark.parametrize("async_mode", [False, True])
async def test_document_exclusions_roundtrip_through_http_and_retriever_tool(wire, async_mode):
    document = Document(
        text="Public instructions",
        metadata={
            "internal_note": "HIDDEN_FROM_LLM",
            "embedding_note": "HIDDEN_FROM_EMBED",
            "title": "Visible title",
        },
        excluded_llm_metadata_keys=["internal_note"],
        excluded_embed_metadata_keys=["embedding_note"],
    )
    stored = {}

    def store(request):
        memory = json.loads(request.content)["requests"][0]
        stored.update(MEMORY | {"memoryId": memory["memoryId"], "metadata": memory["metadata"]})
        return httpx.Response(200, json={"results": [{"success": True, "memory": stored}]})

    def retrieve(request):
        chunk = copy.deepcopy(CHUNK)
        chunk["retrievedItem"]["chunk"]["chunk"].update(
            memoryId=stored["memoryId"], chunkText=document.text
        )
        return ndjson(chunk, {"memoryDefinition": stored})

    wire.responses.extend([store, retrieve])
    ingestor = GoodMemDocumentIngestor(client=wire.sdk, async_client=wire.asdk, space_id="space-1")
    if async_mode:
        await ingestor.aadd_documents([document])
    else:
        ingestor.add_documents([document])
    tool = RetrieverTool.from_defaults(
        GoodMemRetriever(client=wire.sdk, async_client=wire.asdk, space_ids=["space-1"])
    )
    result = (
        await tool.acall(input="instructions") if async_mode else tool.call(input="instructions")
    )
    assert "Visible title" in result.content and "Public instructions" in result.content
    assert "internal_note" not in result.content and "HIDDEN_FROM_LLM" not in result.content
    node = result.raw_output[0].node
    assert node.metadata["internal_note"] == "HIDDEN_FROM_LLM"
    assert "internal_note" in node.excluded_llm_metadata_keys
    assert "embedding_note" in node.excluded_embed_metadata_keys
    assert "HIDDEN_FROM_EMBED" not in node.get_content(MetadataMode.EMBED)
    restored = GoodMemNodeWithScore.model_validate_json(result.raw_output[0].model_dump_json())
    assert restored.node.hash == node.hash
    assert "HIDDEN_FROM_LLM" not in restored.node.get_content(MetadataMode.LLM)
    assert document.metadata == {
        "internal_note": "HIDDEN_FROM_LLM",
        "embedding_note": "HIDDEN_FROM_EMBED",
        "title": "Visible title",
    }


def test_document_reserved_key_is_rejected_without_overwriting_data(wire):
    document = Document(text="note", metadata={"_llamaindex_goodmem": "user value"})
    with pytest.raises(ValueError, match="reserved"):
        GoodMemDocumentIngestor(client=wire.sdk, space_id="space-1").add_documents([document])
    assert not wire.requests
    assert document.metadata["_llamaindex_goodmem"] == "user value"


@pytest.mark.parametrize("options", [None, {"excluded_llm_metadata_keys": "internal_note"}])
def test_malformed_exclusions_fail_before_rendering_metadata(wire, options):
    memory = copy.deepcopy(MEMORY)
    memory["metadata"].update(internal_note="HIDDEN", _llamaindex_goodmem=options)
    wire.responses.append(ndjson(CHUNK, {"memoryDefinition": memory}))
    tool = RetrieverTool.from_defaults(GoodMemRetriever(client=wire.sdk, space_ids=["space-1"]))
    with pytest.raises(ValueError, match="Invalid"):
        tool.call(input="note")
