"""Accepted writes remain visible to agents when indexing cannot be confirmed."""

import httpx
import pytest
from goodmem.errors import AuthenticationError
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.llms import MockLLM
from llama_index.core.workflow import Context

from llama_index.tools.goodmem import GoodMemToolSpec

from .conftest import MEMORY


def indexed_response(status):
    return httpx.Response(
        200,
        json={"results": [{"success": True, "memory": MEMORY | {"processingStatus": status}}]},
    )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("failure", ["timeout", "failed", "http", "transport"])
async def test_accepted_id_survives_wait_failures(wire, async_mode, failure):
    wait_response = {
        "timeout": indexed_response("PROCESSING"),
        "failed": indexed_response("FAILED"),
        "http": httpx.Response(503, json={"message": "Status service unavailable"}),
        "transport": httpx.ReadTimeout("Status read timed out"),
    }[failure]
    wire.responses.extend(
        [
            httpx.Response(201, json=MEMORY | {"processingStatus": "PENDING"}),
            wait_response,
        ]
    )
    spec = GoodMemToolSpec(client=wire.sdk, async_client=wire.asdk, indexing_timeout=0)
    result = (
        await spec.acreate_memory("space-1", "note")
        if async_mode
        else spec.create_memory("space-1", "note")
    )
    assert result["memory_id"] == "memory-1" and result["accepted"]
    assert result["indexing"]["status"] == "unconfirmed" and result["indexing"]["error"]
    assert "get_memory" in result["next_step"] and "do not upload" in result["next_step"]
    # The creation response's PENDING status is stale after a failed wait.
    assert "processing_status" not in result
    assert [r.url.path for r in wire.requests] == ["/v1/memories", "/v1/memories:batchGet"]


async def test_agent_can_recover_accepted_write_without_creating_a_duplicate(wire):
    wire.responses.extend(
        [
            httpx.Response(201, json=MEMORY | {"processingStatus": "PENDING"}),
            indexed_response("PROCESSING"),
            httpx.Response(200, json=MEMORY),
        ]
    )
    spec = GoodMemToolSpec(async_client=wire.asdk, indexing_timeout=0)
    tools = {tool.metadata.name: tool for tool in spec.to_tool_list()}
    agent = ReActAgent(llm=MockLLM(), tools=list(tools.values()))
    context = Context(agent)
    receipt = await agent._call_tool(
        context, tools["create_memory"], {"space_id": "space-1", "text_content": "note"}
    )
    assert not receipt.is_error
    assert "memory-1" in receipt.content and "unconfirmed" in receipt.content
    assert "get_memory" in receipt.content and "do not upload" in receipt.content
    status = await agent._call_tool(
        context,
        tools["get_memory"],
        {
            "memory_id": receipt.raw_output["memory_id"],
            "include_content": False,
        },
    )
    assert not status.is_error and "COMPLETED" in status.content
    assert sum(r.url.path == "/v1/memories" and r.method == "POST" for r in wire.requests) == 1


@pytest.mark.parametrize("async_mode", [False, True])
async def test_initial_write_failure_still_raises(wire, async_mode):
    wire.responses.append(httpx.Response(401, json={"message": "Unauthorized"}))
    spec = GoodMemToolSpec(client=wire.sdk, async_client=wire.asdk)
    with pytest.raises(AuthenticationError):
        if async_mode:
            await spec.acreate_memory("space-1", "note")
        else:
            spec.create_memory("space-1", "note")
    assert len(wire.requests) == 1


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("wait", [False, True])
async def test_create_memory_retains_explicit_wait_behavior(wire, async_mode, wait):
    wire.responses.append(httpx.Response(201, json=MEMORY | {"processingStatus": "PENDING"}))
    if wait:
        wire.responses.append(indexed_response("COMPLETED"))
    spec = GoodMemToolSpec(client=wire.sdk, async_client=wire.asdk)
    result = (
        await spec.acreate_memory("space-1", "note", wait=wait)
        if async_mode
        else spec.create_memory("space-1", "note", wait=wait)
    )
    assert result["memory_id"] == "memory-1"
    assert result["processing_status"] == ("COMPLETED" if wait else "PENDING")
    assert len(wire.requests) == (2 if wait else 1)
