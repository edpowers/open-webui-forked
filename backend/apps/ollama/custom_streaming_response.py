from __future__ import annotations

import re
import typing
from functools import partial
import json
import random

from dataclasses import dataclass

import anyio
import anyio.to_thread

# from starlette._compat import md5_hexdigest
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool

# from starlette.datastructures import URL, MutableHeaders
from starlette.types import Receive, Scope, Send
from starlette.responses import Response


from apps.ollama.utils import (
    decode_or_replace_invalid_chars,
    encode_or_replace_invalid_chars,
)


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

    http_headers_sent: bool = False

    def __init__(
        self,
        content: ContentStream,
        status_code: int = 200,
        headers: typing.Mapping[str, str] | None = None,
        media_type: str | None = None,
        background: BackgroundTask | None = None,
        last_message: str = "",
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
        self.last_message: str = last_message
        self.patterns_to_remove = patterns_to_remove or []
        self.error_patterns = r"Geplaatst|/\*\*\*\*\*\*/|ityEngine"
        self.init_headers(headers)

    async def listen_for_disconnect(self, receive: Receive) -> None:
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break

    async def stream_response(self, send: Send) -> None:
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

            if len(buffer) > 512:
                full_response = decode_or_replace_invalid_chars(buffer, self.charset)

                # Remove specified patterns from the full response
                for pattern in self.patterns_to_remove:
                    full_response = re.sub(pattern, "", full_response, re.MULTILINE)

                raise_error = False
                # Check for error patterns in the full response
                try:
                    if re.search(self.error_patterns, full_response):
                        print(f"Raising not a valid response error {full_response=}")
                        raise_error = True
                    elif full_response == "Assistant:":
                        print("Full response == Assistant")
                        raise_error = True
                    elif "/******" in full_response:
                        print(
                            f"Response contains special characters /****** {full_response=}"
                        )
                        raise_error = True
                    elif self.last_message and "test for llm throw" in str(
                        self.last_message
                    ):
                        print("Testing for LLM throw error.")

                        random_integer = random.randint(0, 1)
                        if random_integer == 1:
                            raise_error = True
                finally:
                    if raise_error:
                        self.status_code = 400
                        await self.send_http_headers_to_client(
                            send, status_code=self.status_code
                        )
                        await self.send_response_error_message_to_client(send)
                        return
                        # Send a message to the frontend indicating that the response is not valid
                        raise NotAValidResponseError(full_response=full_response)

                await self.send_http_headers_to_client(
                    send, status_code=self.status_code
                )

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

        # Guard clause for if the buffer never exceed 512 characters.
        await self.send_http_headers_to_client(send, status_code=self.status_code)

        print(f"Length of buffer is {len(buffer)=} and {self.is_first_message=}")

        # Decode the full buffer to a string for processing
        full_response = decode_or_replace_invalid_chars(buffer, self.charset)

        # Remove specified patterns from the full response
        for pattern in self.patterns_to_remove:
            full_response = re.sub(pattern, "", full_response)

        # Encode the sanitized response back to bytes
        sanitized_response = encode_or_replace_invalid_chars(
            full_response, self.charset
        )

        print("Reached end of streaming body.")

        # Send the sanitized response in one go
        await send(
            {
                "type": "http.response.body",
                "body": sanitized_response,
                "more_body": False,
            }
        )

    async def send_http_headers_to_client(
        self, send: Send, status_code: int = 200
    ) -> None:
        """Send the http headers to the client."""
        if self.http_headers_sent:
            return

        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": self.raw_headers,
            }
        )

        self.http_headers_sent = True

    async def send_response_error_message_to_client(self, send: Send) -> None:
        """Send the response error message to the client frontend."""
        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(
                    {
                        "error": True,
                        "content": "The response is not valid. Please click the regenerate button to try again.",
                    }
                ).encode(self.charset),
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
