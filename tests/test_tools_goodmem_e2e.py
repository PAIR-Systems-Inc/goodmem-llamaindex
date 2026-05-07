"""End-to-end smoke test for GoodMem LlamaIndex tool spec.

Exercises every one of the 11 tools and every post-processor knob against
a live GoodMem server. Mirrors the §7 smoke test plan in the build guide.

Required environment variables:
    GOODMEM_API_KEY       -- API key (X-API-Key)
    GOODMEM_BASE_URL      -- e.g. https://localhost:8080
    GOODMEM_EMBEDDER_ID   -- UUID of an embedder to bind a new space to
    GOODMEM_RERANKER_ID   -- (optional) reranker UUID for retrieve variants
    GOODMEM_LLM_ID        -- (optional) LLM UUID for abstractReply variants
    GOODMEM_VERIFY_SSL    -- "false" to disable verification (dev only)

Usage:
    GOODMEM_API_KEY=... GOODMEM_BASE_URL=https://localhost:8080 \\
    GOODMEM_EMBEDDER_ID=... GOODMEM_RERANKER_ID=... GOODMEM_LLM_ID=... \\
    GOODMEM_VERIFY_SSL=false \\
    .venv/bin/python -m pytest tests/test_tools_goodmem_e2e.py -v -s
"""

import os
import time
import uuid

import httpx
import pytest

from llama_index.core.schema import Document
from llama_index.tools.goodmem import GoodMemToolSpec

API_KEY = os.environ.get("GOODMEM_API_KEY", "")
BASE_URL = os.environ.get("GOODMEM_BASE_URL", "")
EMBEDDER_ID = os.environ.get("GOODMEM_EMBEDDER_ID", "")
RERANKER_ID = os.environ.get("GOODMEM_RERANKER_ID", "")
LLM_ID = os.environ.get("GOODMEM_LLM_ID", "")
VERIFY_SSL = os.environ.get("GOODMEM_VERIFY_SSL", "true").lower() != "false"

LANGGRAPH_FACT = (
    "LangGraph is a framework built by LangChain for constructing "
    "stateful, multi-actor agent applications using directed graphs."
)
GOODMEM_FACT = (
    "GoodMem is a server-side memory layer for AI agents that handles "
    "embedding, vector search, reranking, and answer generation."
)
CARBONARA_DISTRACTOR = (
    "Pasta carbonara is a Roman dish made with eggs, hard cheese, "
    "cured pork, and pepper. It contains no cream."
)


# ---------------------------------------------------------------------------
# Skip the whole module if env vars aren't set
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.skipif(
    not (API_KEY and BASE_URL and EMBEDDER_ID),
    reason=(
        "Live e2e test needs GOODMEM_API_KEY, GOODMEM_BASE_URL and "
        "GOODMEM_EMBEDDER_ID set"
    ),
)


# ---------------------------------------------------------------------------
# Module-scoped state shared across the ordered tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tool_spec():
    return GoodMemToolSpec(
        api_key=API_KEY,
        base_url=BASE_URL,
        verify_ssl=VERIFY_SSL,
    )


@pytest.fixture(scope="module")
def state():
    """Shared mutable state across the ordered smoke-test steps."""
    return {
        "space_id": None,
        "memory_ids": [],
        "space_name_initial": None,
        "space_name_renamed": None,
    }


# ---------------------------------------------------------------------------
# §7 smoke test, ordered
# ---------------------------------------------------------------------------


class TestSmokeOrdered:
    """Ordered end-to-end test that runs every tool against the live server.

    Each test depends on the previous one having succeeded (mutates `state`).
    pytest runs tests in declaration order within a class by default.
    """

    # Step 1
    def test_01_list_embedders_contains_configured(self, tool_spec):
        embedders = tool_spec.list_embedders()
        assert isinstance(embedders, list)
        ids = [e.get("embedderId") or e.get("id") for e in embedders]
        assert EMBEDDER_ID in ids, (
            f"Configured embedder {EMBEDDER_ID} not in list: {ids}"
        )

    # Step 2
    def test_02_list_spaces_returns_list(self, tool_spec):
        spaces = tool_spec.list_spaces()
        assert isinstance(spaces, list)

    # Step 3
    def test_03_create_space(self, tool_spec, state):
        unique = f"llamaindex-smoke-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        result = tool_spec.create_space(name=unique, embedder_id=EMBEDDER_ID)
        assert result["spaceId"], f"create_space returned no spaceId: {result}"
        assert result["reused"] is False
        state["space_id"] = result["spaceId"]
        state["space_name_initial"] = unique

    # Step 4
    def test_04_get_space(self, tool_spec, state):
        assert state["space_id"], "previous step did not produce a spaceId"
        result = tool_spec.get_space(space_id=state["space_id"])
        assert result.get("spaceId") == state["space_id"]
        assert result.get("name") == state["space_name_initial"]

    # Step 5
    def test_05_update_space_rename(self, tool_spec, state):
        assert state["space_id"]
        new_name = state["space_name_initial"] + "-renamed"
        result = tool_spec.update_space(
            space_id=state["space_id"], name=new_name
        )
        assert result.get("name") == new_name
        state["space_name_renamed"] = new_name

    # Step 6 — three create_memory inserts
    def test_06a_create_memory_langgraph_fact(self, tool_spec, state):
        result = tool_spec.create_memory(
            space_id=state["space_id"], text_content=LANGGRAPH_FACT
        )
        assert result["memoryId"]
        assert result["contentType"] == "text/plain"
        state["memory_ids"].append(result["memoryId"])

    def test_06b_create_memory_goodmem_fact(self, tool_spec, state):
        result = tool_spec.create_memory(
            space_id=state["space_id"], text_content=GOODMEM_FACT
        )
        assert result["memoryId"]
        state["memory_ids"].append(result["memoryId"])

    def test_06c_create_memory_carbonara_distractor(self, tool_spec, state):
        result = tool_spec.create_memory(
            space_id=state["space_id"], text_content=CARBONARA_DISTRACTOR
        )
        assert result["memoryId"]
        state["memory_ids"].append(result["memoryId"])

    # Step 7
    def test_07_list_memories_count_three(self, tool_spec, state):
        # Indexing is async but list_memories sees pending memories too,
        # so this should return 3 immediately.
        result = tool_spec.list_memories(space_id=state["space_id"])
        assert result["totalMemories"] == 3, (
            f"Expected 3 memories, got {result['totalMemories']}: {result}"
        )

    # Step 8
    def test_08_get_memory_returns_correct_space(self, tool_spec, state):
        memory_id = state["memory_ids"][0]
        result = tool_spec.get_memory(
            memory_id=memory_id, include_content=False
        )
        assert "memory" in result
        assert result["memory"].get("spaceId") == state["space_id"]
        assert result["memory"].get("memoryId") == memory_id

    # Step 9a – plain retrieve, wait_for_indexing=True
    def test_09a_retrieve_plain_with_wait(self, tool_spec, state):
        docs = tool_spec.retrieve_memories(
            query="Which framework is used for stateful multi-actor agents?",
            space_ids=[state["space_id"]],
            max_results=5,
            wait_for_indexing=True,
        )
        assert isinstance(docs, list)
        assert len(docs) > 0, "retrieve found 0 results even after waiting"
        for d in docs:
            assert isinstance(d, Document)
            assert d.text
            assert "chunkId" in d.metadata

    # Step 9b – with reranker only
    def test_09b_retrieve_with_reranker(self, tool_spec, state):
        if not RERANKER_ID:
            pytest.skip("GOODMEM_RERANKER_ID not set")
        docs = tool_spec.retrieve_memories(
            query="Which framework is used for stateful multi-actor agents?",
            space_ids=[state["space_id"]],
            max_results=5,
            reranker_id=RERANKER_ID,
            wait_for_indexing=False,
        )
        assert isinstance(docs, list)
        assert len(docs) > 0

    # Step 9c – with LLM (expect non-empty abstractReply.text)
    def test_09c_retrieve_with_llm_abstract_reply(self, tool_spec, state):
        if not LLM_ID:
            pytest.skip("GOODMEM_LLM_ID not set")
        docs = tool_spec.retrieve_memories(
            query="Which framework is used for stateful multi-actor agents?",
            space_ids=[state["space_id"]],
            max_results=5,
            llm_id=LLM_ID,
            llm_temperature=0.2,
            wait_for_indexing=False,
        )
        assert len(docs) > 0
        abstract = docs[0].metadata.get("abstractReply")
        assert abstract is not None, "expected abstractReply with llm_id set"
        assert abstract.get("text"), f"abstractReply has no text: {abstract}"

    # Step 9d – reranker + LLM + threshold
    def test_09d_retrieve_full_post_processor(self, tool_spec, state):
        if not (RERANKER_ID and LLM_ID):
            pytest.skip("Need both GOODMEM_RERANKER_ID and GOODMEM_LLM_ID")
        docs = tool_spec.retrieve_memories(
            query="Which framework is used for stateful multi-actor agents?",
            space_ids=[state["space_id"]],
            max_results=5,
            reranker_id=RERANKER_ID,
            llm_id=LLM_ID,
            relevance_threshold=0.1,
            llm_temperature=0.5,
            wait_for_indexing=False,
        )
        assert len(docs) > 0
        assert docs[0].metadata.get("abstractReply", {}).get("text")

    # Step 9e – reranker + chronological resort
    def test_09e_retrieve_chronological_resort(self, tool_spec, state):
        if not RERANKER_ID:
            pytest.skip("GOODMEM_RERANKER_ID not set")
        docs = tool_spec.retrieve_memories(
            query="Which framework is used for stateful multi-actor agents?",
            space_ids=[state["space_id"]],
            max_results=5,
            reranker_id=RERANKER_ID,
            chronological_resort=True,
            wait_for_indexing=False,
        )
        assert len(docs) > 0

    # Step 9f – max_results=1
    def test_09f_retrieve_max_results_one(self, tool_spec, state):
        docs = tool_spec.retrieve_memories(
            query="Which framework is used for stateful multi-actor agents?",
            space_ids=[state["space_id"]],
            max_results=1,
            wait_for_indexing=False,
        )
        assert len(docs) == 1, f"Expected exactly 1 result, got {len(docs)}"

    # Step 10
    def test_10_delete_one_memory(self, tool_spec, state):
        target = state["memory_ids"][-1]
        result = tool_spec.delete_memory(memory_id=target)
        assert result["memoryId"] == target

        # Verify it's gone (or at least no longer fetchable)
        with pytest.raises(httpx.HTTPStatusError):
            tool_spec.get_memory(memory_id=target, include_content=False)

    # Step 11
    def test_11_delete_space_cascades(self, tool_spec, state):
        result = tool_spec.delete_space(space_id=state["space_id"])
        assert result["spaceId"] == state["space_id"]

        # Verify it's gone
        with pytest.raises(httpx.HTTPStatusError):
            tool_spec.get_space(space_id=state["space_id"])
