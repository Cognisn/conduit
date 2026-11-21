"""A fake Chat Completions client that answers from a recorded replay."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from conduit.testing.replay import Recording, Replay


class FakeStream:
    """An async stream of chunks; optionally raises after some of them."""

    def __init__(self, chunks: Sequence[Any], *, error: Exception | None = None, after: int = 0):
        self._chunks = list(chunks)
        self._error = error
        self._after = after
        self.closed = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        for index, chunk in enumerate(self._chunks):
            if self._error is not None and index == self._after:
                raise self._error
            yield chunk
        if self._error is not None and self._after >= len(self._chunks):
            raise self._error

    async def close(self) -> None:
        self.closed = True


class FakeCompletions:
    def __init__(self, client: FakeClient) -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> FakeStream:
        self._client.calls.append(dict(kwargs))
        if self._client.create_error is not None:
            raise self._client.create_error
        body = {k: v for k, v in kwargs.items() if k != "timeout"}
        chunks = self._client.replay.next(body)
        if self._client.wrap is not None:
            chunks = [self._client.wrap(c) for c in chunks]
        stream = FakeStream(
            chunks, error=self._client.stream_error, after=self._client.stream_error_after
        )
        self._client.streams.append(stream)
        return stream


class FakeChat:
    def __init__(self, client: FakeClient) -> None:
        self.completions = FakeCompletions(client)


class FakeClient:
    """Stands in for ``openai.AsyncOpenAI``: ``chat.completions.create`` and ``close``."""

    def __init__(
        self,
        recording: Recording,
        *,
        create_error: BaseException | None = None,
        stream_error: Exception | None = None,
        stream_error_after: int = 0,
        wrap: Any = None,
    ) -> None:
        self.replay = Replay(recording)
        self.create_error = create_error
        self.stream_error = stream_error
        self.stream_error_after = stream_error_after
        self.wrap = wrap
        self.calls: list[dict[str, Any]] = []
        self.streams: list[FakeStream] = []
        self.closed = False
        self.chat = FakeChat(self)

    async def close(self) -> None:
        self.closed = True
