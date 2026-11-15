import asyncio
import unittest
from types import SimpleNamespace

from llm.events import StreamErrorEvent, StreamStartEvent
from llm.models import StreamChunk, ToolCall
from llm.provider import Provider
from llm.stream import StreamAccumulator


class StreamFinishTests(unittest.TestCase):
    def test_finish_reason_is_preserved_without_final_response(self):
        for reason, expected in [("length", "length"), ("content_filter", "content_filter"),
                                 ("stop", "completed"), (None, "unknown")]:
            with self.subTest(reason=reason):
                acc = StreamAccumulator()
                acc.feed(StreamChunk(content="partial", finish_reason=reason))
                metadata = acc.finish().message.metadata
                self.assertEqual(metadata["stop_reason"], expected)
                if reason is not None:
                    self.assertEqual(metadata["finish_reason"], reason)

    def test_missing_finish_reason_does_not_imply_successful_tool_call(self):
        acc = StreamAccumulator()
        acc.feed(StreamChunk(content="", tool_calls=[ToolCall(id="1", name="f")]))
        self.assertEqual(acc.finish().message.metadata["stop_reason"], "unknown")


class StreamLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_while_waiting_closes_source(self):
        signal = SimpleNamespace(cancelled=False)
        started = asyncio.Event()
        closed = asyncio.Event()

        async def source():
            try:
                started.set()
                await asyncio.Event().wait()
                yield StreamChunk(content="never")
            finally:
                closed.set()

        async def collect():
            return [event async for event in StreamAccumulator().stream(source(), signal)]

        task = asyncio.create_task(collect())
        await started.wait()
        signal.cancelled = True
        events = await asyncio.wait_for(task, timeout=2)
        self.assertIsInstance(events[-1], StreamErrorEvent)
        self.assertEqual(events[-1].stop_reason, "cancelled")
        self.assertTrue(closed.is_set())

    async def test_closing_high_level_stream_closes_source(self):
        closed = asyncio.Event()

        class FakeProvider(Provider):
            def __init__(self, config):
                pass

            def chat(self, messages, *, model, tools=None, **kwargs):
                raise NotImplementedError

            def stream(self, messages, *, model, tools=None, **kwargs):
                raise NotImplementedError

            async def achat(self, messages, *, model, tools=None, **kwargs):
                raise NotImplementedError

            async def achat_stream(self, messages, *, model, tools=None, **kwargs):
                try:
                    yield StreamChunk(content="hello")
                    yield StreamChunk(content="later")
                finally:
                    closed.set()

        events = FakeProvider(None).astream_events([], model="test")
        self.assertIsInstance(await anext(events), StreamStartEvent)
        await events.aclose()
        self.assertTrue(closed.is_set())
