"""A failing reranker must not cost the hits the server returned.

GoodMem v1.0.320 answers a request whose reranker cannot run with NOT_FOUND (naming
the reranker) and RERANKING_FAILED, then returns the vector-stage hits, scored as
negative inner products. fixtures/retrieve_degraded_hits.ndjson is that stream as
captured live. Retrieval status contract: a problem with hits returns the hits,
partial, with the statuses (Q4a); a problem without hits returns an empty list with
a warning and a WARNING log line (Q4b); FEATURE_DISABLED and LLM_CAPABILITY_INFERRED
are noise by code alone (Q1); unknown codes are surfaced as UNKNOWN (Q3).
"""

import copy
import json
import logging
from pathlib import Path

import httpx
import pytest
from llama_index.core.postprocessor import SimilarityPostprocessor

from llama_index.tools.goodmem import GoodMemRetriever, GoodMemToolSpec

from .conftest import CHUNK, MEMORY, RERANKER_ID, SPACE_ID, ndjson

DEGRADED = (Path(__file__).parent / "fixtures" / "retrieve_degraded_hits.ndjson").read_text(
    encoding="utf-8"
)
MISSING_RERANKER = "00000000-0000-7000-8000-000000000000"
FALLBACK_TEXT = "The fixture canary is ORYX-2290. CAMEL toolkit audit.\n"
FALLBACK_RAW = -0.5845972299575806
SUMMARIZATION_OFF = {
    "status": {
        "code": "FEATURE_DISABLED",
        "message": "Abstract reply generation disabled: no LLM configured.",
        "details": {"required_param": "llm_id", "feature": "summarization"},
    }
}


def degraded(with_hits=True):
    lines = [
        line
        for line in DEGRADED.splitlines()
        if with_hits or ("retrievedItem" not in line and "memoryDefinition" not in line)
    ]
    return httpx.Response(
        200, text="\n".join(lines), headers={"content-type": "application/x-ndjson"}
    )


def hits(scores, *statuses, statuses_first=False):
    events = []
    for index, score in enumerate(scores):
        event = copy.deepcopy(CHUNK)
        hit = event["retrievedItem"]["chunk"]
        hit["relevanceScore"] = score
        hit["chunk"].update(chunkId=f"chunk-{index}", chunkText=f"Evidence {index}")
        events.append(event)
    events.append({"memoryDefinition": MEMORY})
    return ndjson(*statuses, *events) if statuses_first else ndjson(*events, *statuses)


def retriever(wire, reranker_id=MISSING_RERANKER, **options):
    return GoodMemRetriever(
        client=wire.sdk,
        async_client=wire.asdk,
        space_ids=[SPACE_ID],
        reranker_id=reranker_id,
        **options,
    )


async def retrieve(value, async_mode, query="capital of Jordan"):
    return await value.aretrieve(query) if async_mode else value.retrieve(query)


def logged(caplog):
    return [
        r
        for r in caplog.records
        if r.name.startswith("llama_index.tools.goodmem") and r.levelno >= logging.WARNING
    ]


def warned(recwarn):
    return [w for w in recwarn if "GoodMem" in str(w.message)]


@pytest.mark.parametrize("async_mode", [False, True])
async def test_failed_reranker_returns_the_servers_vector_hits(wire, async_mode, caplog, recwarn):
    wire.responses.append(degraded())
    nodes = await retrieve(retriever(wire), async_mode)
    body = json.loads(wire.requests[0].content)
    assert body["postProcessor"]["config"]["reranker_id"] == MISSING_RERANKER
    assert [node.text for node in nodes] == [FALLBACK_TEXT]
    node = nodes[0]
    assert node.raw_score == FALLBACK_RAW
    assert node.score == -FALLBACK_RAW > 0  # oriented like any vector score
    assert node.score_kind == "negative_inner_product"
    assert node.partial is True
    assert [s["code"] for s in node.statuses] == [
        "NOT_FOUND",
        "FEATURE_DISABLED",
        "RERANKING_FAILED",
    ]
    assert node.statuses[0]["details"] == {"reranker_id": MISSING_RERANKER}
    # Hits came back, so there is no warning; the problem is still logged once.
    assert not warned(recwarn)
    [record] = logged(caplog)
    assert "NOT_FOUND" in record.getMessage() and "RERANKING_FAILED" in record.getMessage()
    assert "FEATURE_DISABLED" not in record.getMessage()


def test_fallback_hits_survive_a_similarity_cutoff(wire):
    # Left un-negated, the vector score -0.58 would fall below a 0.0 cutoff.
    wire.responses.append(degraded())
    nodes = retriever(wire).retrieve("capital of Jordan")
    kept = SimilarityPostprocessor(similarity_cutoff=0.0).postprocess_nodes(nodes)
    assert [node.text for node in kept] == [FALLBACK_TEXT]


@pytest.mark.parametrize(
    "not_found",
    [
        {"message": "Reranker validation failed", "details": {"reranker_id": RERANKER_ID}},
        {"message": "Lookup failed", "details": {"rerankerId": RERANKER_ID}},
        {"message": "Reranker not found"},
    ],
    ids=["reranker_id", "rerankerId", "message"],
)
def test_reranker_not_found_alone_means_vector_scores(wire, not_found):
    status = {"status": {"code": "NOT_FOUND"} | not_found}
    wire.responses.append(hits([-0.9, -0.2], status, statuses_first=True))
    nodes = retriever(wire, RERANKER_ID).retrieve("query")
    assert [node.score for node in nodes] == [0.9, 0.2]
    assert {node.score_kind for node in nodes} == {"negative_inner_product"}
    assert all(node.partial for node in nodes)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "stream, expected",
    [
        (lambda: degraded(with_hits=False), ["NOT_FOUND", "RERANKING_FAILED"]),
        (lambda: ndjson({"status": {"code": "NEW_CODE", "message": "New failure"}}), ["UNKNOWN"]),
    ],
    ids=["failed-reranker", "unknown-code"],
)
async def test_problem_without_hits_returns_empty_with_warning_and_log(
    wire, async_mode, stream, expected, caplog
):
    wire.responses.append(stream())
    with pytest.warns(UserWarning, match="GoodMem returned no results") as caught:
        nodes = await retrieve(retriever(wire), async_mode)
    assert nodes == []
    [warning] = [w for w in caught if "GoodMem" in str(w.message)]
    [record] = logged(caplog)
    for code in expected:
        assert code in str(warning.message) and code in record.getMessage()
    assert "FEATURE_DISABLED" not in str(warning.message)


@pytest.mark.parametrize("async_mode", [False, True])
async def test_working_reranker_scores_are_unchanged(wire, async_mode, caplog, recwarn):
    wire.responses.append(hits([0.94140625, 0.484375, 0.3203125], SUMMARIZATION_OFF))
    nodes = await retrieve(retriever(wire, RERANKER_ID), async_mode)
    assert [node.score for node in nodes] == [0.94140625, 0.484375, 0.3203125]
    assert [node.raw_score for node in nodes] == [0.94140625, 0.484375, 0.3203125]
    assert {node.score_kind for node in nodes} == {"reranker"}
    assert not warned(recwarn) and not logged(caplog)


def test_vector_search_scores_are_unchanged(wire, caplog, recwarn):
    wire.responses.append(hits([-0.7853, -0.5769, -0.1115]))
    nodes = retriever(wire, None).retrieve("capital of Jordan")
    assert [node.score for node in nodes] == [0.7853, 0.5769, 0.1115]
    assert {node.score_kind for node in nodes} == {"negative_inner_product"}
    assert not warned(recwarn) and not logged(caplog)


@pytest.mark.parametrize(
    "status",
    [
        {"code": "VECTOR_SEARCH_PARTIAL", "message": "One shard timed out"},
        {"code": "NEW_CODE", "message": "A future notice"},
        {"code": "NOT_FOUND", "message": "Space not found", "details": {"space_id": SPACE_ID}},
    ],
    ids=["vector-search-partial", "unknown-code", "not-found-space"],
)
def test_an_unrelated_problem_keeps_reranker_scores(wire, status):
    wire.responses.append(hits([0.87, 0.12], {"status": status}))
    nodes = retriever(wire, RERANKER_ID).retrieve("query")
    assert [node.score for node in nodes] == [0.87, 0.12]
    assert {node.score_kind for node in nodes} == {"reranker"}
    assert all(node.partial for node in nodes)
    assert nodes[0].statuses[0]["code"] == (
        status["code"] if status["code"] != "NEW_CODE" else "UNKNOWN"
    )


@pytest.mark.parametrize(
    "reranker_id, notices",
    [
        (None, []),
        (RERANKER_ID, [SUMMARIZATION_OFF]),
        (RERANKER_ID, [{"status": {"code": "FEATURE_DISABLED", "message": "Off"}}]),
        (
            None,
            [{"status": {"code": "FEATURE_DISABLED", "message": "Off", "details": {"x": "y"}}}],
        ),
        (None, [{"status": {"code": "LLM_CAPABILITY_INFERRED", "message": "Inferred"}}]),
    ],
    ids=["vector", "reranker", "feature-disabled-no-details", "feature-disabled-other", "llm"],
)
def test_healthy_search_is_not_partial(wire, reranker_id, notices, caplog, recwarn):
    wire.responses.extend([hits([0.5], *notices), hits([0.5], *notices)])
    nodes = retriever(wire, reranker_id).retrieve("query")
    assert nodes and not any(node.partial for node in nodes)
    result = GoodMemToolSpec(client=wire.sdk).retrieve_memories(
        "query", [SPACE_ID], reranker_id=reranker_id
    )
    assert result["chunks"] and result["partial"] is False
    assert not warned(recwarn) and not logged(caplog)


@pytest.mark.parametrize("async_mode", [False, True])
async def test_tool_names_fallback_scores_as_vector(wire, async_mode):
    wire.responses.append(degraded())
    spec = GoodMemToolSpec(client=wire.sdk, async_client=wire.asdk)
    call = spec.aretrieve_memories if async_mode else spec.retrieve_memories
    result = call("capital of Jordan", [SPACE_ID], reranker_id=MISSING_RERANKER)
    result = await result if async_mode else result
    assert [chunk["score"] for chunk in result["chunks"]] == [FALLBACK_RAW]
    assert result["score_kind"] == "negative_inner_product"
    assert result["partial"] is True
    assert {"NOT_FOUND", "RERANKING_FAILED"} <= {s["code"] for s in result["statuses"]}


def test_tool_names_working_reranker_scores(wire):
    wire.responses.append(hits([0.94140625, 0.484375], SUMMARIZATION_OFF))
    result = GoodMemToolSpec(client=wire.sdk).retrieve_memories(
        "query", [SPACE_ID], reranker_id=RERANKER_ID
    )
    assert [chunk["score"] for chunk in result["chunks"]] == [0.94140625, 0.484375]
    assert result["score_kind"] == "reranker" and result["partial"] is False


def test_tool_problem_without_hits_is_empty_and_partial(wire):
    wire.responses.append(degraded(with_hits=False))
    result = GoodMemToolSpec(client=wire.sdk).retrieve_memories(
        "query", [SPACE_ID], reranker_id=MISSING_RERANKER
    )
    assert result["chunks"] == [] and result["partial"] is True
    assert {"NOT_FOUND", "RERANKING_FAILED"} <= {s["code"] for s in result["statuses"]}
