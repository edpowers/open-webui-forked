from __future__ import annotations

import re
import typing
from functools import partial

from dataclasses import dataclass

import anyio
import anyio.to_thread

# from starlette._compat import md5_hexdigest
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool

# from starlette.datastructures import URL, MutableHeaders
from starlette.types import Receive, Scope, Send
from starlette.responses import Response


Content = typing.Union[str, bytes, memoryview]
SyncContentStream = typing.Iterable[Content]
AsyncContentStream = typing.AsyncIterable[Content]
ContentStream = typing.Union[AsyncContentStream, SyncContentStream]


@dataclass
class NotAValidResponseError(Exception):
    full_response: str

    def __str__(self) -> str:
        return f"""
        This is not a valid response. Retrying text generation.
        {self.full_response=}
        """


class CustomStreamingResponse(Response):
    body_iterator: AsyncContentStream

    def __init__(
        self,
        content: ContentStream,
        status_code: int = 200,
        headers: typing.Mapping[str, str] | None = None,
        media_type: str | None = None,
        background: BackgroundTask | None = None,
        patterns_to_remove: list[str] | None = None,
        is_first_message: bool = False,
    ) -> None:
        if isinstance(content, typing.AsyncIterable):
            self.body_iterator = content
        else:
            self.body_iterator = iterate_in_threadpool(content)
        self.status_code = status_code
        self.media_type = self.media_type if media_type is None else media_type
        self.background = background
        self.is_first_message = is_first_message
        self.patterns_to_remove = patterns_to_remove or []
        self.error_patterns = r"Geplaatst|/\*\*\*\*\*\*/|ityEngine"
        self.init_headers(headers)

    async def listen_for_disconnect(self, receive: Receive) -> None:
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break

    async def _stream_response(self, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        async for chunk in self.body_iterator:
            # print(f"Starting streaming with {chunk=}")

            # chunk=b'age":{"role":"assistant","content":" law"},"done":false}
            # \n{"model":"freedom:latest","created_at":"2024-06-04T18:23:57.238914535Z",
            # "message":{"role":"assistant","content":"."},"done":false}
            # \n{"model":"freedom:latest","created_at":"2024-06-04T18:23:57.275040856Z",
            # "message":{"role":"assistant","content":" However"},"done":false}
            # \n{"model":"freedom:latest","created_at":"2024-06-04T18:23:57.311100408Z",
            # "message":{"role":"assistant","content":","},"done":false}

            if not isinstance(chunk, (bytes, memoryview)):
                chunk = chunk.encode(self.charset)

            # Remove specified patterns from the chunk
            for pattern in self.patterns_to_remove:
                chunk_test = re.sub(pattern.encode(self.charset), b"", chunk)
                # Guard to stop from delivering empty text.
                if len(chunk_test) > 10:
                    chunk = chunk_test
                else:  # If all parts of the chunk have already been removed.
                    break

            # print(f"Ending chunk {chunk=}")

            # If the chunk is empty, don't return. Wait until valid response.
            if not chunk or len(chunk) < 2:
                continue

            if len(str(chunk)) > 2:
                await send(
                    {"type": "http.response.body", "body": chunk, "more_body": True}
                )

        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def stream_response(self, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )

        buffer = bytearray()
        total_buffer_len = 0

        n = 0

        async for chunk in self.body_iterator:
            # Add 1 to the n enumerator.
            n += 1

            if not isinstance(chunk, (bytes, memoryview)):
                chunk = chunk.encode(self.charset)

            print(f"Chunk {n} with length {len(chunk)}")

            buffer.extend(chunk)
            total_buffer_len += len(chunk)

            if self.is_first_message and n == 1:
                print(f"{self.is_first_message=}")
                continue

            if len(buffer) > 256:
                full_response = buffer.decode(self.charset)
                # Check for error patterns in the full response
                # Check for error patterns in the full response
                if (
                    re.search(self.error_patterns, full_response)
                    or full_response == "Assistant:"
                    or "/******" in full_response
                    or "INST" in full_response
                ):
                    print("Raising not a valid response error")
                    raise NotAValidResponseError(full_response)

                # Remove specified patterns from the full response
                for pattern in self.patterns_to_remove:
                    full_response = re.sub(pattern, "", full_response)

                # Encode the sanitized response back to bytes
                sanitized_response = full_response.encode(self.charset)

                # Send the sanitized response in one go
                await send(
                    {
                        "type": "http.response.body",
                        "body": sanitized_response,
                        "more_body": True,
                    }
                )

                buffer = bytearray()
                total_buffer_len = 0

        print(f"Length of buffer is {len(buffer)=} and {self.is_first_message=}")

        # Decode the full buffer to a string for processing
        full_response = buffer.decode(self.charset)

        # Check for error patterns in the full response
        if (
            re.search(self.error_patterns, full_response)
            or full_response == "Assistant:"
            or "/******" in full_response
            or "INST" in full_response
        ):
            print("Raising not a valid response error")
            raise NotAValidResponseError(full_response)

        # Remove specified patterns from the full response
        for pattern in self.patterns_to_remove:
            full_response = re.sub(pattern, "", full_response)

        # Encode the sanitized response back to bytes
        sanitized_response = full_response.encode(self.charset)

        print("Reached end of streaming body.")

        # Send the sanitized response in one go
        await send(
            {
                "type": "http.response.body",
                "body": sanitized_response,
                "more_body": False,
            }
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async with anyio.create_task_group() as task_group:

            async def wrap(func: typing.Callable[[], typing.Awaitable[None]]) -> None:
                await func()
                task_group.cancel_scope.cancel()

            task_group.start_soon(wrap, partial(self.stream_response, send))
            await wrap(partial(self.listen_for_disconnect, receive))

        if self.background is not None:
            await self.background()
