"""SDK sessions with explicit ownership and native async support."""

import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

from goodmem import AsyncGoodmem, Goodmem


class Connection:
    """Use injected SDK clients, or create short-lived clients from configuration.

    Args:
        client: Caller-owned synchronous SDK client. With injected clients,
            synchronous calls require this client even when environment settings exist.
        async_client: Caller-owned asynchronous SDK client. With injected clients,
            asynchronous calls require this client even when environment settings exist.
        api_key: API key, defaulting to GOODMEM_API_KEY.
        base_url: Server root, defaulting to GOODMEM_BASE_URL.
        verify_ssl: Certificate verification, or a path to a trusted CA bundle.
        timeout: SDK HTTP timeout in seconds.
    """

    def __init__(
        self,
        *,
        client: Goodmem | None = None,
        async_client: AsyncGoodmem | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        verify_ssl: bool | str = True,
        timeout: float = 120,
    ) -> None:
        self.client = client
        self.async_client = async_client
        self.options = {
            "api_key": api_key or os.getenv("GOODMEM_API_KEY"),
            "base_url": base_url or os.getenv("GOODMEM_BASE_URL"),
            "verify": verify_ssl,
            "timeout": timeout,
        }

    @contextmanager
    def sync(self) -> Iterator[Goodmem]:
        """Yield a synchronous client without closing caller-owned connections."""
        if self.client is not None:
            yield self.client
        else:
            if self.async_client is not None:
                raise ValueError("Pass client for sync calls with an injected async SDK client")
            with Goodmem(**self.options) as client:
                yield client

    @asynccontextmanager
    async def async_(self) -> AsyncIterator[AsyncGoodmem]:
        """Yield a native async client without using a worker thread."""
        if self.async_client is not None:
            yield self.async_client
        else:
            if self.client is not None:
                raise ValueError("Pass async_client for async calls with an injected SDK client")
            async with AsyncGoodmem(**self.options) as client:
                yield client
