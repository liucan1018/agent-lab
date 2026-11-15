import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from llm.config import Config
from llm.models import Message, ToolCall
from llm.providers.openai import OpenAIProvider, _message_to_wire, _usage


def usage():
    return NS(model_dump=lambda: {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "details": {}})


def call_fragment(index, *, id=None, name=None, args=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


def chunk(*, content=None, reasoning=None, calls=None, finish=None, usage_value=None, model="returned-model", choices=True):
    delta = NS(content=content, tool_calls=calls, reasoning_content=reasoning)
    choice = NS(index=0, delta=delta, finish_reason=finish)
    return NS(model=model, usage=usage_value, choices=[choice] if choices else [])


class SyncStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        self.closed = True


class AsyncStream(SyncStream):
    def __aiter__(self):
        async def iterate():
            for item in self.chunks:
                yield item
        return iterate()

    async def close(self):
        self.closed = True


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = OpenAIProvider(Config(api_key="secret", max_tokens=50))
        self.messages = [Message(role="user", content="hello")]

    def test_sync_chat_and_tool_round_trip(self):
        call = NS(model_dump=lambda: {"id": "call1", "function": {"name": "weather", "arguments": '{"city":"北京"}'}})
        message = NS(content=None, tool_calls=[call], reasoning_content=None)
        completion = NS(model="returned-model", choices=[NS(message=message, finish_reason="tool_calls")], usage=usage())
        create = Mock(return_value=completion)
        self.provider._sync_client = NS(chat=NS(completions=NS(create=create)))
        response = self.provider.chat(self.messages, model="requested", temperature=0.2)
        self.assertEqual(response.content, "")
        self.assertEqual(response.tool_calls[0].args, {"city": "北京"})
        self.assertEqual(response.usage["total_tokens"], 5)
        params = create.call_args.kwargs
        self.assertEqual(params["temperature"], 0.2)
        self.assertEqual(params["max_completion_tokens"], 50)
        self.assertEqual(_message_to_wire(response.to_message())["tool_calls"][0]["id"], "call1")
        self.assertEqual(_message_to_wire(Message(role="tool", content="ok", metadata={"tool_call_id": "call1"}))["tool_call_id"], "call1")

    def test_injected_clients_and_lazy_creation(self):
        sync_client = Mock()
        async_client = Mock()
        provider = OpenAIProvider(Config(api_key="secret"), client=sync_client, async_client=async_client)
        self.assertIs(provider._sync(), sync_client)
        self.assertIs(provider._async(), async_client)
        provider.close()
        sync_client.close.assert_called_once_with()
        with patch("llm.providers.openai.AsyncOpenAI") as create_async:
            provider = OpenAIProvider(Config(api_key="secret"), client=sync_client)
            self.assertIs(provider._async(), create_async.return_value)
            create_async.assert_called_once()

    def test_cache_usage_preserves_input_tokens(self):
        data = {"prompt_tokens": 1000, "completion_tokens": 20, "total_tokens": 1020,
                "prompt_tokens_details": {"cached_tokens": 800}}
        result = _usage(NS(model_dump=lambda: data))
        self.assertEqual(result["prompt_tokens"], 1000)
        self.assertEqual(result["cache_read_tokens"], 800)
        self.assertNotIn("prompt_tokens_details", result)
        self.assertEqual(_usage({"prompt_tokens": 1000, "prompt_cache_hit_tokens": 600})["cache_read_tokens"], 600)
        self.assertEqual(_usage({"prompt_tokens_details": {"cached_tokens": 0}, "prompt_cache_hit_tokens": 600})["cache_read_tokens"], 0)
        self.assertNotIn("cache_read_tokens", _usage({"prompt_tokens": 1000}))

    def test_reasoning_stream_and_final_response(self):
        chunks = [chunk(reasoning="first "), chunk(content="answer", reasoning="second"), chunk(finish="stop")]
        stream = SyncStream(chunks)
        client = NS(chat=NS(completions=NS(create=Mock(return_value=stream))))
        provider = OpenAIProvider(Config(), client=client)
        result = list(provider.stream(self.messages, model="m"))
        self.assertEqual([(c.content, c.metadata) for c in result[:-1]], [
            ("", {"reasoning_content": "first "}), ("answer", {"reasoning_content": "second"}),
        ])
        self.assertEqual(result[-1].response.reasoning_content, "first second")
        self.assertEqual(result[-1].response.content, "answer")

    def test_async_reasoning_stream(self):
        async def run():
            stream = AsyncStream([chunk(reasoning="thinking"), chunk(content="done", finish="stop")])
            client = NS(chat=NS(completions=NS(create=AsyncMock(return_value=stream))))
            provider = OpenAIProvider(Config(), async_client=client)
            result = [item async for item in provider.achat_stream(self.messages, model="m")]
            self.assertEqual(result[0].metadata, {"reasoning_content": "thinking"})
            self.assertEqual(result[-1].response.reasoning_content, "thinking")
            self.assertTrue(stream.closed)
        coroutine = run()
        with self.assertRaises(StopIteration):
            coroutine.send(None)

    def test_images_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(b"image")
            msg = Message(role="user", content=[{"type": "image", "asset_path": str(path)}])
            self.assertEqual(_message_to_wire(msg)["content"][0]["image_url"]["url"], "data:image/png;base64," + base64.b64encode(b"image").decode())
        with self.assertRaisesRegex(ValueError, "tool_call_id"):
            _message_to_wire(Message(role="tool", content="missing"))
        with self.assertRaises(ValueError):
            self.provider._params(self.messages, "m", None, {"n": 2}, stream=False)
        with self.assertRaises(ValueError):
            _message_to_wire(Message(role="assistant", content="", metadata={"tool_calls": [ToolCall(id="1", name="f", error="bad")]}))

    def chunks(self):
        return [chunk(content="Hi ", calls=[call_fragment(0, id="call1", name="weather", args='{"city":')]),
                chunk(content="there", calls=[call_fragment(0, args='"北京"}')]),
                chunk(finish="tool_calls"), chunk(choices=False, usage_value=usage())]

    def test_sync_stream_accumulates_and_closes(self):
        stream = SyncStream(self.chunks())
        create = Mock(return_value=stream)
        self.provider._sync_client = NS(chat=NS(completions=NS(create=create)))
        result = list(self.provider.stream(self.messages, model="requested"))
        self.assertEqual([part.content for part in result], ["Hi ", "there", ""])
        self.assertEqual(result[-1].response.content, "Hi there")
        self.assertEqual(result[-1].tool_calls[0].args, {"city": "北京"})
        self.assertEqual(result[-1].usage["total_tokens"], 5)
        self.assertTrue(stream.closed)
        self.assertEqual(create.call_args.kwargs["stream_options"], {"include_usage": True})

    def test_early_close_and_invalid_json(self):
        stream = SyncStream([chunk(content="x"), chunk(calls=[call_fragment(0, id="1", name="f", args="{")])])
        self.provider._sync_client = NS(chat=NS(completions=NS(create=Mock(return_value=stream))))
        iterator = self.provider.stream(self.messages, model="m")
        next(iterator)
        iterator.close()
        self.assertTrue(stream.closed)
        stream = SyncStream([chunk(calls=[call_fragment(0, id="1", name="f", args="{")]), chunk(finish="tool_calls")])
        self.provider._sync_client.chat.completions.create.return_value = stream
        self.assertIsNotNone(list(self.provider.stream(self.messages, model="m"))[-1].tool_calls[0].error)

    def test_async_chat_and_stream(self):
        async def run():
            message = NS(content="answer", tool_calls=None, reasoning_content=None)
            completion = NS(model="m", choices=[NS(message=message, finish_reason="stop")], usage=None)
            stream = AsyncStream(self.chunks())
            create = AsyncMock(side_effect=[completion, stream])
            self.provider._async_client = NS(chat=NS(completions=NS(create=create)))
            response = await self.provider.achat(self.messages, model="m")
            self.assertEqual(response.content, "answer")
            result = [part async for part in self.provider.achat_stream(self.messages, model="m")]
            self.assertEqual(result[-1].response.content, "Hi there")
            self.assertEqual(result[-1].tool_calls[0].args, {"city": "北京"})
            self.assertTrue(stream.closed)
        # The mocked SDK never suspends, so this also works in restricted
        # environments that cannot create the sockets used by an event loop.
        coroutine = run()
        with self.assertRaises(StopIteration):
            coroutine.send(None)


if __name__ == "__main__":
    unittest.main()
