"""Exercise real SDK deserialization and request construction over mock HTTP."""

import json

import httpx
import pytest
from goodmem import AsyncGoodmem, Goodmem

AUDIT = dict(createdAt=1, updatedAt=1, createdById="user", updatedById="user")
MEMORY = dict(
    **AUDIT,
    memoryId="memory-1",
    spaceId="space-1",
    contentType="text/plain",
    processingStatus="COMPLETED",
    pageImageStatus="COMPLETED",
    pageImageCount=0,
    metadata={"source": "https://example.org/docs", "title": "Docs", "nested": {"yes": True}},
)
CHUNK = {
    "retrievedItem": {
        "chunk": dict(
            resultSetId="set-1",
            memoryIndex=0,
            relevanceScore=-0.82,
            chunk=dict(
                **AUDIT,
                chunkId="chunk-1",
                memoryId="memory-1",
                chunkSequenceNumber=0,
                chunkText="Retrieved evidence",
                vectorStatus="COMPLETED",
            ),
        )
    }
}
SPACE = dict(
    **AUDIT,
    spaceId="space-1",
    name="Docs",
    ownerId="user",
    labels={},
    spaceEmbedders=[
        dict(**AUDIT, spaceId="space-1", embedderId="embedder-1", defaultRetrievalWeight=1)
    ],
)


def ndjson(*events):
    return httpx.Response(
        200,
        text="\n".join(json.dumps(e) for e in events),
        headers={"content-type": "application/x-ndjson"},
    )


class Wire:
    def __init__(self):
        self.responses, self.requests = [], []
        self.http = httpx.Client(
            base_url="https://goodmem.test", transport=httpx.MockTransport(self.handle)
        )
        self.ahttp = httpx.AsyncClient(
            base_url="https://goodmem.test", transport=httpx.MockTransport(self.handle)
        )
        self.sdk = Goodmem(http_client=self.http)
        self.asdk = AsyncGoodmem(http_client=self.ahttp)

    def handle(self, request):
        self.requests.append(request)
        assert self.responses, f"Unexpected request: {request.method} {request.url}"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(request)
        return response


@pytest.fixture
async def wire():
    value = Wire()
    try:
        yield value
        assert not value.responses, "Expected HTTP requests were not made"
    finally:
        value.http.close()
        await value.ahttp.aclose()
