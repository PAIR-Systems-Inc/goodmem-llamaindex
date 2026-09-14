"""GoodMem results must retain relevance order in standard LlamaIndex consumers."""

import copy

import pytest
from llama_index.core.llms import MockLLM
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

from llama_index.tools.goodmem import GoodMemRetrievalError, GoodMemRetriever, GoodMemToolSpec

from .conftest import CHUNK, MEMORY, ndjson


def hits(scores, *extra):
    events = []
    for index, score in enumerate(scores):
        event = copy.deepcopy(CHUNK)
        hit = event["retrievedItem"]["chunk"]
        hit["relevanceScore"] = score
        hit["chunk"].update(chunkId=f"chunk-{index}", chunkText=f"Evidence {index}")
        events.append(event)
    return ndjson(*events, {"memoryDefinition": MEMORY}, *extra)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("reranked", [False, True])
async def test_framework_scores_preserve_original_scale_and_kind(wire, async_mode, reranked):
    raw_scores = [2.5, 0.0, -0.9] if reranked else [-2.5, 0.0, 0.9]
    wire.responses.append(hits(raw_scores))
    nodes = GoodMemRetriever(
        client=wire.sdk,
        async_client=wire.asdk,
        space_ids=["space-1"],
        reranker_id="reranker-1" if reranked else None,
    )
    result = await nodes.aretrieve("query") if async_mode else nodes.retrieve("query")
    assert [node.score for node in result] == [2.5, 0.0, -0.9]
    assert [node.metadata["_goodmem"]["raw_score"] for node in result] == raw_scores
    assert {node.metadata["_goodmem"]["score_kind"] for node in result} == {
        "reranker" if reranked else "negative_inner_product"
    }


@pytest.mark.parametrize("mode", list(FUSION_MODES))
@pytest.mark.parametrize("reranked", [False, True])
def test_query_fusion_keeps_best_hit_first(wire, mode, reranked):
    # Negative reranker scores are valid too; direction cannot be inferred from sign.
    wire.responses.append(hits([-0.2, -0.9] if reranked else [-0.9, -0.2]))
    retriever = GoodMemRetriever(
        client=wire.sdk,
        space_ids=["space-1"],
        reranker_id="reranker-1" if reranked else None,
    )
    fusion = QueryFusionRetriever(
        [retriever], llm=MockLLM(), mode=mode, num_queries=1, use_async=False, similarity_top_k=2
    )
    assert [node.node_id for node in fusion.retrieve("query")] == ["chunk-0", "chunk-1"]


def test_default_similarity_postprocessor_accepts_positive_vector_similarity(wire):
    wire.responses.append(hits([-0.9, -0.2]))
    nodes = GoodMemRetriever(client=wire.sdk, space_ids=["space-1"]).retrieve("query")
    assert [node.node_id for node in SimilarityPostprocessor().postprocess_nodes(nodes)] == [
        "chunk-0",
        "chunk-1",
    ]


def test_failed_reranking_raises_before_interpreting_fallback_as_reranker_scores(wire):
    failure = {"status": {"code": "RERANKING_FAILED", "message": "Using vector results"}}
    wire.responses.append(hits([-0.9, -0.2], failure))
    with pytest.raises(GoodMemRetrievalError, match="RERANKING_FAILED"):
        GoodMemRetriever(client=wire.sdk, space_ids=["space-1"], reranker_id="reranker-1").retrieve(
            "query"
        )
    # Administrative retrieval keeps server scores and partial results as documented.
    wire.responses.append(hits([-0.9, -0.2], failure))
    result = GoodMemToolSpec(client=wire.sdk).retrieve_memories(
        "query", ["space-1"], reranker_id="reranker-1"
    )
    assert result["partial"]
    assert [chunk["score"] for chunk in result["chunks"]] == [-0.9, -0.2]
