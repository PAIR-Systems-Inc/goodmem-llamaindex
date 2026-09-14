"""Injected SDK clients must never be replaced with environment connections."""

import httpx
import pytest
from llama_index.core.schema import Document

from llama_index.tools.goodmem import (
    GoodMemDocumentIngestor,
    GoodMemRetriever,
    GoodMemToolSpec,
    _connection,
)


@pytest.mark.parametrize("injected_async", [False, True])
@pytest.mark.parametrize("operation", ["retrieve", "ingest", "create_memory"])
@pytest.mark.parametrize("explicit_config", [False, True])
async def test_injected_client_blocks_other_mode_before_http(
    wire, monkeypatch, injected_async, operation, explicit_config
):
    monkeypatch.setenv("GOODMEM_BASE_URL", "https://other-server.test")
    monkeypatch.setenv("GOODMEM_API_KEY", "other-credential")

    def unexpected_client(**kwargs):
        pytest.fail("An injected connection must not fall back to another SDK client")

    monkeypatch.setattr(_connection, "Goodmem", unexpected_client)
    monkeypatch.setattr(_connection, "AsyncGoodmem", unexpected_client)
    options = {"async_client": wire.asdk} if injected_async else {"client": wire.sdk}
    # Even explicit scalar settings cannot substitute for the other injected client.
    if explicit_config:
        options.update(base_url="https://explicit-server.test", api_key="explicit-credential")
    if operation == "retrieve":
        obj = GoodMemRetriever(**options, space_ids=["space-1"])
        sync_call, async_call, args = obj.retrieve, obj.aretrieve, ("query",)
    elif operation == "ingest":
        obj = GoodMemDocumentIngestor(**options, space_id="space-1")
        sync_call, async_call, args = (
            obj.add_documents,
            obj.aadd_documents,
            ([Document(text="note")],),
        )
    else:
        obj = GoodMemToolSpec(**options)
        sync_call, async_call, args = obj.create_memory, obj.acreate_memory, ("space-1", "note")
    missing = "client" if injected_async else "async_client"
    with pytest.raises(ValueError, match=f"Pass {missing} "):
        if injected_async:
            sync_call(*args)
        else:
            await async_call(*args)
    assert wire.requests == []
    assert not wire.http.is_closed and not wire.ahttp.is_closed


@pytest.mark.parametrize("async_mode", [False, True])
async def test_configuration_connections_support_environment_and_explicit_options(
    wire, monkeypatch, async_mode
):
    monkeypatch.setenv("GOODMEM_BASE_URL", "https://environment.test")
    monkeypatch.setenv("GOODMEM_API_KEY", "environment-credential")
    constructed = []

    def client(**options):
        constructed.append(options)
        return wire.asdk if async_mode else wire.sdk

    monkeypatch.setattr(_connection, "AsyncGoodmem" if async_mode else "Goodmem", client)
    explicit = {"base_url": "https://explicit.test", "api_key": "explicit-credential"}
    for options in [{}, explicit]:
        wire.responses.append(httpx.Response(200, json={"embedders": []}))
        spec = GoodMemToolSpec(**options)
        assert (await spec.alist_embedders() if async_mode else spec.list_embedders()) == []
    assert constructed[0]["base_url"] == "https://environment.test"
    assert constructed[0]["api_key"] == "environment-credential"
    assert all(constructed[1][key] == value for key, value in explicit.items())
