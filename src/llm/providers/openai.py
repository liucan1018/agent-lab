"""OpenAI Chat Completions 适配器。"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAI

from ..config import Config
from ..models import Message, Response, StreamChunk, ToolCall
from ..provider import Provider


def _message_to_wire(message: Message) -> dict[str, Any]:
    """转换消息；图片 data 优先于 asset_path，工具结果需提供调用 ID。"""
    content: Any = message.content
    if isinstance(content, list):
        parts = []
        for part in content:
            if part["type"] == "text":
                parts.append({"type": "text", "text": part["text"]})
            else:
                if message.role != "user":
                    raise ValueError("Images are only supported in user messages")
                data = part.get("data")
                if not data:
                    data = base64.b64encode(Path(part["asset_path"]).read_bytes()).decode("ascii")
                url = data if data.startswith("data:") else f"data:{part['mime_type']};base64,{data}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
        content = parts

    result: dict[str, Any] = {"role": message.role, "content": content}
    metadata = message.metadata or {}
    if message.role == "assistant" and metadata.get("tool_calls"):
        result["tool_calls"] = [
            (call if isinstance(call, ToolCall) else ToolCall.model_validate(call)).to_wire_dict()
            for call in metadata["tool_calls"]
        ]
    if message.role == "tool":
        call_id = metadata.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("Tool messages require metadata['tool_call_id']")
        result["tool_call_id"] = call_id
    return result


def _usage(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    # Preserve the provider's prompt_tokens; cache_read_tokens is a subset.
    data = value if isinstance(value, dict) else (
        value.model_dump() if hasattr(value, "model_dump") else vars(value)
    )
    result = {key: val for key, val in data.items() if type(val) is int}
    details = data.get("prompt_tokens_details")
    if details is not None and not isinstance(details, dict):
        details = details.model_dump() if hasattr(details, "model_dump") else vars(details)
    cached = details.get("cached_tokens") if details is not None else None
    if type(cached) is not int:
        cached = data.get("prompt_cache_hit_tokens")
    if type(cached) is int:
        result["cache_read_tokens"] = cached
    return result


def _response(completion: Any) -> Response:
    if len(completion.choices) != 1:
        raise ValueError("Expected exactly one chat completion choice")
    choice = completion.choices[0]
    message = choice.message
    calls = [ToolCall.model_validate(call.model_dump()) for call in (message.tool_calls or [])]
    return Response(
        content=message.content,
        model=completion.model,
        tool_calls=calls or None,
        reasoning_content=getattr(message, "reasoning_content", None),
        usage=_usage(completion.usage),
        finish_reason=choice.finish_reason,
    )


class _StreamState:
    def __init__(self, model: str) -> None:
        self.model = model
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.calls: dict[int, dict[str, str]] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, int] | None = None

    def consume(self, chunk: Any) -> StreamChunk | None:
        if chunk.model:
            self.model = chunk.model
        if chunk.usage is not None:
            self.usage = _usage(chunk.usage)
        if not chunk.choices:
            return None
        if len(chunk.choices) != 1 or chunk.choices[0].index != 0:
            raise ValueError("Expected exactly one chat completion choice")
        choice = chunk.choices[0]
        delta = choice.delta
        content = delta.content or ""
        if content:
            self.content.append(content)
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            self.reasoning.append(reasoning)
        for item in delta.tool_calls or []:
            call = self.calls.setdefault(item.index, {"id": "", "name": "", "arguments": ""})
            if item.id:
                call["id"] += item.id
            if item.function:
                call["name"] += item.function.name or ""
                call["arguments"] += item.function.arguments or ""
        if choice.finish_reason is not None:
            self.finish_reason = choice.finish_reason
        # Emit reasoning as it arrives and retain it in the final response.
        metadata = {"reasoning_content": reasoning} if reasoning else None
        return StreamChunk(content=content, metadata=metadata) if content or reasoning else None

    def final(self) -> StreamChunk:
        calls = [
            ToolCall.model_validate({"id": call["id"], "function": {"name": call["name"], "arguments": call["arguments"]}})
            for _, call in sorted(self.calls.items())
        ]
        response = Response(
            content="".join(self.content), model=self.model,
            tool_calls=calls or None, reasoning_content="".join(self.reasoning) or None,
            usage=self.usage, finish_reason=self.finish_reason,
        )
        return StreamChunk(content="", finish_reason=self.finish_reason, tool_calls=calls or None, usage=self.usage, response=response)


class OpenAIProvider(Provider):
    """基于 OpenAI SDK 的单 choice 适配器。

    可注入同步或异步客户端；未注入的一侧按需创建。close/aclose 会关闭
    对应客户端，包括注入的客户端。
    """

    def __init__(self, config: Config, *, client: OpenAI | None = None, async_client: AsyncOpenAI | None = None):
        self.config = config
        options: dict[str, Any] = {"timeout": config.timeout, "max_retries": config.max_retries}
        if config.api_key is not None:
            options["api_key"] = config.api_key.get_secret_value()
        if config.base_url is not None:
            options["base_url"] = config.base_url
        self._client_options = options
        self._sync_client = client
        self._async_client = async_client

    def _params(self, messages: list[Message], model: str, tools: list[dict] | None, kwargs: dict[str, Any], *, stream: bool) -> dict[str, Any]:
        if kwargs.get("n", 1) != 1:
            raise ValueError("OpenAIProvider supports only n=1")
        if "stream" in kwargs or "messages" in kwargs:
            raise ValueError("stream and messages are controlled by the provider")
        params: dict[str, Any] = {"model": model, "messages": [_message_to_wire(m) for m in messages], "temperature": self.config.temperature}
        if self.config.max_tokens is not None and "max_tokens" not in kwargs and "max_completion_tokens" not in kwargs:
            params["max_completion_tokens"] = self.config.max_tokens
        if tools is not None:
            params["tools"] = tools
        params.update(kwargs)
        if stream:
            params["stream"] = True
            if "stream_options" not in params:
                params["stream_options"] = {"include_usage": True}
        return params

    def _sync(self) -> OpenAI:
        if self._sync_client is None:
            self._sync_client = OpenAI(**self._client_options)
        return self._sync_client

    def _async(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(**self._client_options)
        return self._async_client

    def chat(self, messages: list[Message], *, model: str, tools: list[dict] | None = None, **kwargs: Any) -> Response:
        result = self._sync().chat.completions.create(**self._params(messages, model, tools, kwargs, stream=False))
        return _response(result)

    async def achat(self, messages: list[Message], *, model: str, tools: list[dict] | None = None, **kwargs: Any) -> Response:
        result = await self._async().chat.completions.create(**self._params(messages, model, tools, kwargs, stream=False))
        return _response(result)

    def stream(self, messages: list[Message], *, model: str, tools: list[dict] | None = None, **kwargs: Any) -> Iterator[StreamChunk]:
        stream = self._sync().chat.completions.create(**self._params(messages, model, tools, kwargs, stream=True))
        state = _StreamState(model)
        try:
            for chunk in stream:
                event = state.consume(chunk)
                if event is not None:
                    yield event
        finally:
            stream.close()
        yield state.final()

    async def achat_stream(self, messages: list[Message], *, model: str, tools: list[dict] | None = None, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        stream = await self._async().chat.completions.create(**self._params(messages, model, tools, kwargs, stream=True))
        state = _StreamState(model)
        try:
            async for chunk in stream:
                event = state.consume(chunk)
                if event is not None:
                    yield event
        finally:
            await stream.close()
        yield state.final()

    def close(self) -> None:
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None
