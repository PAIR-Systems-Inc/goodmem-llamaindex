"""Every Python block in README.md runs as written against a local GoodMem server."""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from .conftest import AUDIT, SPACE_ID

README = Path(__file__).resolve().parents[1] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)


def _memory(request, status):
    return dict(
        **AUDIT,
        memoryId=request["memoryId"],
        spaceId=request["spaceId"],
        contentType=request.get("contentType", "text/plain"),
        processingStatus=status,
        pageImageStatus="COMPLETED",
        pageImageCount=0,
        metadata=request.get("metadata") or {},
        originalContentRef=request.get("originalContentRef"),
    )


class _Server(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.memories, self.requests = {}, []


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body, content_type="application/json"):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        store = self.server.memories
        self.server.requests.append((self.path, self.headers.get("x-api-key"), body))
        if self.path == "/v1/memories:batchCreate":
            store.update((r["memoryId"], r) for r in body["requests"])
            results = [_memory(r, "PENDING") for r in body["requests"]]
        elif self.path == "/v1/memories:batchGet":
            results = [_memory(store[i], "COMPLETED") for i in body["memoryIds"]]
        elif self.path == "/v1/memories:retrieve":
            events = []
            for index, (memory_id, request) in enumerate(store.items()):
                chunk = dict(
                    **AUDIT,
                    chunkId=f"chunk-{index}",
                    memoryId=memory_id,
                    chunkSequenceNumber=0,
                    chunkText=request["originalContent"],
                    vectorStatus="COMPLETED",
                )
                hit = dict(resultSetId="set", memoryIndex=index, relevanceScore=-0.8, chunk=chunk)
                events += [
                    {"retrievedItem": {"chunk": hit}},
                    {"memoryDefinition": _memory(request, "COMPLETED")},
                ]
            return self._send("\n".join(map(json.dumps, events)), "application/x-ndjson")
        else:
            self.send_error(404)
            return
        self._send({"results": [{"success": True, "memory": m} for m in results]})


@pytest.fixture
def server(monkeypatch):
    server = _Server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("GOODMEM_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("GOODMEM_API_KEY", "readme-key")
    monkeypatch.setenv("GOODMEM_SPACE_ID", SPACE_ID)
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def test_readme_has_python_examples():
    assert len(BLOCKS) == 2


@pytest.mark.parametrize("index", range(len(BLOCKS)))
def test_readme_block_runs_on_its_own(server, index, capsys):
    namespace = {"__name__": "__readme__"}
    exec(compile(BLOCKS[index], f"README.md python block {index + 1}", "exec"), namespace)
    assert server.requests or index == 1
    assert all(key == "readme-key" for _, key, _ in server.requests)
    if index == 0:
        paths = [path for path, _, _ in server.requests]
        assert paths == [
            "/v1/memories:batchCreate",
            "/v1/memories:batchGet",
            "/v1/memories:retrieve",
        ]
        assert (
            "The returns period is 30 days. https://example.com/returns" in capsys.readouterr().out
        )
    else:
        # The configured tool is usable as a model would call it.
        server.memories["7b2eac74-5f40-4b9c-9d3e-2c3d4e5f6071"] = {
            "memoryId": "7b2eac74-5f40-4b9c-9d3e-2c3d4e5f6071",
            "spaceId": SPACE_ID,
            "originalContent": "Refunds take five days.",
            "originalContentRef": "https://example.com/refunds",
        }
        output = namespace["search"].call("refund time").content
        assert "Refunds take five days." in output
        assert [k["spaceId"] for k in server.requests[-1][2]["spaceKeys"]] == [SPACE_ID]
