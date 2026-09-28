"""Exercise real SDK deserialization and request construction over mock HTTP."""

import json

import httpx
import pytest
from goodmem import AsyncGoodmem, Goodmem

# GoodMem IDs are UUIDs; the integration refuses anything else before a request.
SPACE_ID = "5f0c8a52-3d2e-4f7a-9b1c-0a1b2c3d4e5f"
SPACE_ID_2 = "6a1d9b63-4e3f-4a8b-8c2d-1b2c3d4e5f60"
MEMORY_ID = "7b2eac74-5f40-4b9c-9d3e-2c3d4e5f6071"
MEMORY_ID_2 = "8c3fbd85-6051-4cad-8e4f-3d4e5f607182"
EMBEDDER_ID = "9d40ce96-7162-4dbe-9f50-4e5f60718293"
RERANKER_ID = "ae51dfa7-8273-4ecf-8061-5f60718293a4"

AUDIT = dict(createdAt=1, updatedAt=1, createdById="user", updatedById="user")
MEMORY = dict(
    **AUDIT,
    memoryId=MEMORY_ID,
    spaceId=SPACE_ID,
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
                memoryId=MEMORY_ID,
                chunkSequenceNumber=0,
                chunkText="Retrieved evidence",
                vectorStatus="COMPLETED",
            ),
        )
    }
}
SPACE = dict(
    **AUDIT,
    spaceId=SPACE_ID,
    name="Docs",
    ownerId="user",
    labels={},
    spaceEmbedders=[
        dict(**AUDIT, spaceId=SPACE_ID, embedderId=EMBEDDER_ID, defaultRetrievalWeight=1)
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
