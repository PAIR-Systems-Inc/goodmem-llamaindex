"""Non-UUID IDs are refused before the real SDK can send any HTTP request.

The official SDK interpolates IDs into URL paths unescaped, and httpx resolves dot
segments, so memory_id="../spaces/<id>" would delete a space. These tests drive the
real SDK and httpx against a local server that records every request it receives.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import ANY
from urllib.parse import urlsplit

import pytest
from goodmem import AsyncGoodmem, Goodmem
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.llms import MockLLM
from llama_index.core.schema import Document
from llama_index.core.workflow import Context

from llama_index.tools.goodmem import (
    GoodMemDocumentIngestor,
    GoodMemRetriever,
    GoodMemToolSpec,
    await_memories,
    wait_for_memories,
)

from .conftest import MEMORY, SPACE

U = "0b5c3d52-6f2e-4a8f-9c1d-2e3f4a5b6c7d"
PAYLOADS = [
    f"../spaces/{U}",
    f"a/../../spaces/{U}",
    f"%2e%2e/spaces/{U}",
    f"..%2Fspaces%2F{U}",
    f"{U}/../../spaces/{U}",
    "",
    f" {U}",
    f"{U}?x=1",
    f"{U}#frag",
    f"{U}\n",
    f"{{{U}}}",
    "memory-1",
]


class _Recorder(BaseHTTPRequestHandler):
    def _handle(self):
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.requests.append((self.command, self.path, body))
        status, payload, content_type = _respond(self.command, urlsplit(self.path).path, body)
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = do_PUT = do_DELETE = _handle

    def log_message(self, *args):
        pass


def _respond(method, path, body):
    """Answer like GoodMem for any path, so an unguarded request looks successful."""
    as_json = "application/json"
    if method == "DELETE":
        return 200, {}, as_json
    if path.endswith(":retrieve"):
        return 200, b"", "application/x-ndjson"
    if path.endswith(":batchGet"):
        ids = json.loads(body)["memoryIds"]
        results = [
            {"success": True, "memory": MEMORY | {"memoryId": i, "processingStatus": "COMPLETED"}}
            for i in ids
        ]
        return 200, {"results": results}, as_json
    if path.endswith(":batchCreate"):
        requests = json.loads(body)["requests"]
        results = [
            {"success": True, "memory": MEMORY | {"memoryId": r["memoryId"]}} for r in requests
        ]
        return 200, {"results": results}, as_json
    if path.endswith("/content"):
        return 200, b"Stored note", "text/plain"
    if path.endswith("/memories") and method == "GET":
        return 200, {"memories": []}, as_json
    if path.startswith("/v1/memories"):
        return 200, MEMORY, as_json
    return 200, SPACE, as_json


@pytest.fixture
def server(tmp_path):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    httpd.requests = []
    # A short poll interval keeps shutdown() from waiting 0.5s after every test.
    thread = threading.Thread(target=httpd.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    (tmp_path / "note.txt").write_text("note")
    try:
        yield SimpleNamespace(
            requests=httpd.requests,
            conn={"base_url": f"http://127.0.0.1:{httpd.server_port}", "api_key": "test-key"},
            uploads=tmp_path,
        )
    finally:
        httpd.shutdown()
        httpd.server_close()


def _tools(s):
    return GoodMemToolSpec(**s.conn, upload_directory=s.uploads)


def _wait(s, v):
    with Goodmem(**s.conn) as client:
        wait_for_memories(client, [v], 1)


async def _await(s, v):
    async with AsyncGoodmem(**s.conn) as client:
        await await_memories(client, [v], 1)


def _ingestor(s, v):
    return GoodMemDocumentIngestor(**s.conn, space_id=v)


# name: (field named in the error, sync call, async call, requests a valid UUID makes)
ENTRY_POINTS = {
    # IDs interpolated into a URL path by the SDK.
    "get_space": (
        "space_id",
        lambda s, v: _tools(s).get_space(v),
        lambda s, v: _tools(s).aget_space(v),
        [("GET", "/v1/spaces/{u}")],
    ),
    "update_space": (
        "space_id",
        lambda s, v: _tools(s).update_space(v, name="Renamed"),
        lambda s, v: _tools(s).aupdate_space(v, name="Renamed"),
        [("PUT", "/v1/spaces/{u}")],
    ),
    "delete_space": (
        "space_id",
        lambda s, v: _tools(s).delete_space(v),
        lambda s, v: _tools(s).adelete_space(v),
        [("DELETE", "/v1/spaces/{u}")],
    ),
    "list_memories": (
        "space_id",
        lambda s, v: _tools(s).list_memories(v),
        lambda s, v: _tools(s).alist_memories(v),
        [("GET", "/v1/spaces/{u}/memories")],
    ),
    "get_memory": (
        "memory_id",
        lambda s, v: _tools(s).get_memory(v),
        lambda s, v: _tools(s).aget_memory(v),
        [("GET", "/v1/memories/{u}"), ("GET", "/v1/memories/{u}/content")],
    ),
    "delete_memory": (
        "memory_id",
        lambda s, v: _tools(s).delete_memory(v),
        lambda s, v: _tools(s).adelete_memory(v),
        [("DELETE", "/v1/memories/{u}")],
    ),
    # IDs sent in a request body: not a traversal risk, refused for consistency.
    "create_space": (
        "embedder_id",
        lambda s, v: _tools(s).create_space("Docs", v),
        lambda s, v: _tools(s).acreate_space("Docs", v),
        [("POST", "/v1/spaces")],
    ),
    "create_memory": (
        "space_id",
        lambda s, v: _tools(s).create_memory(v, "note", wait=False),
        lambda s, v: _tools(s).acreate_memory(v, "note", wait=False),
        [("POST", "/v1/memories")],
    ),
    "upload_memory": (
        "space_id",
        lambda s, v: _tools(s).upload_memory(v, "note.txt"),
        lambda s, v: _tools(s).aupload_memory(v, "note.txt"),
        [("POST", "/v1/memories")],
    ),
    "retrieve_memories.space_ids": (
        "space_ids",
        lambda s, v: _tools(s).retrieve_memories("q", [v]),
        lambda s, v: _tools(s).aretrieve_memories("q", [v]),
        [("POST", "/v1/memories:retrieve")],
    ),
    "retrieve_memories.reranker_id": (
        "reranker_id",
        lambda s, v: _tools(s).retrieve_memories("q", [U], reranker_id=v),
        lambda s, v: _tools(s).aretrieve_memories("q", [U], reranker_id=v),
        [("POST", "/v1/memories:retrieve")],
    ),
    "retrieve_memories.llm_id": (
        "llm_id",
        lambda s, v: _tools(s).retrieve_memories("q", [U], llm_id=v),
        lambda s, v: _tools(s).aretrieve_memories("q", [U], llm_id=v),
        [("POST", "/v1/memories:retrieve")],
    ),
    "GoodMemRetriever.space_ids": (
        "space_ids",
        lambda s, v: GoodMemRetriever(**s.conn, space_ids=[v]).retrieve("q"),
        lambda s, v: GoodMemRetriever(**s.conn, space_ids=[v]).aretrieve("q"),
        [("POST", "/v1/memories:retrieve")],
    ),
    "GoodMemRetriever.reranker_id": (
        "reranker_id",
        lambda s, v: GoodMemRetriever(**s.conn, space_ids=[U], reranker_id=v).retrieve("q"),
        lambda s, v: GoodMemRetriever(**s.conn, space_ids=[U], reranker_id=v).aretrieve("q"),
        [("POST", "/v1/memories:retrieve")],
    ),
    "GoodMemDocumentIngestor.space_id": (
        "space_id",
        lambda s, v: _ingestor(s, v).add_documents([Document(text="note")]),
        lambda s, v: _ingestor(s, v).aadd_documents([Document(text="note")]),
        [("POST", "/v1/memories:batchCreate")],
    ),
    "wait_for_memories": ("memory_ids", _wait, _await, [("POST", "/v1/memories:batchGet")]),
}
PATH_ENTRY_POINTS = list(ENTRY_POINTS)[:6]


async def _invoke(server, name, value, async_mode):
    _, call, acall, _ = ENTRY_POINTS[name]
    return await acall(server, value) if async_mode else call(server, value)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("payload", PAYLOADS)
@pytest.mark.parametrize("name", list(ENTRY_POINTS))
async def test_non_uuid_id_is_refused_before_any_request(server, name, payload, async_mode):
    field = ENTRY_POINTS[name][0]
    try:
        result = await _invoke(server, name, payload, async_mode)
    except ValueError as exc:
        message = str(exc)
    else:
        received = [(m, p) for m, p, _ in server.requests]
        pytest.fail(f"{name} accepted {payload!r}; returned {result!r}; server got {received}")
    assert field in message and "UUID" in message, message
    assert server.requests == []


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("name", list(ENTRY_POINTS))
async def test_valid_uuid_reaches_only_the_intended_path(server, name, async_mode):
    # Uppercase input is normalized to GoodMem's canonical lowercase form.
    await _invoke(server, name, U.upper(), async_mode)
    expected = [(m, p.format(u=U)) for m, p in ENTRY_POINTS[name][3]]
    assert [(m, urlsplit(p).path) for m, p, _ in server.requests] == expected
    for _, target, body in server.requests:
        assert U.upper() not in target and U.upper().encode() not in body
    if name not in PATH_ENTRY_POINTS:
        assert U.encode() in server.requests[0][2]


@pytest.mark.parametrize("tool", ["delete_space", "delete_memory", "get_memory"])
async def test_agent_tool_call_reports_invalid_id_as_error(server, tool):
    tools = {t.metadata.name: t for t in GoodMemToolSpec(**server.conn).to_tool_list()}
    field = "space_id" if tool == "delete_space" else "memory_id"
    agent = ReActAgent(llm=MockLLM(), tools=list(tools.values()))
    output = await agent._call_tool(Context(agent), tools[tool], {field: f"../embedders/{U}"})
    assert output.is_error
    assert field in output.content and "UUID" in output.content
    assert server.requests == []


def test_tool_schemas_declare_ids_as_uuids(server):
    tools = _tools(server).to_tool_list()
    params = {t.metadata.name: t.metadata.get_parameters_dict()["properties"] for t in tools}
    # The annotation is printed in each description; OpenAI rejects more than 1024 characters.
    assert all(t.metadata.to_openai_tool()["function"]["description"] for t in tools)
    declared = {
        ("get_space", "space_id"),
        ("update_space", "space_id"),
        ("delete_space", "space_id"),
        ("list_memories", "space_id"),
        ("get_memory", "memory_id"),
        ("delete_memory", "memory_id"),
        ("create_space", "embedder_id"),
        ("create_memory", "space_id"),
        ("upload_memory", "space_id"),
    }
    for tool, field in declared:
        assert params[tool][field] == {"format": "uuid", "title": ANY, "type": "string"}
    assert params["retrieve_memories"]["space_ids"]["items"] == {"format": "uuid", "type": "string"}
    for field in ["reranker_id", "llm_id"]:
        schemas = params["retrieve_memories"][field]["anyOf"]
        assert schemas == [{"format": "uuid", "type": "string"}, {"type": "null"}], field
