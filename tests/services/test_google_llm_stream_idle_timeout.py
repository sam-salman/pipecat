#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: GoogleLLMService must not hang forever on a stalled stream.

If Gemini's stream stops producing chunks without closing, the turn used to
stay open indefinitely -- nothing bounded how long we waited for the next
chunk. stream_idle_timeout_secs fires on_completion_timeout instead.
"""

import asyncio
from unittest.mock import patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext

try:
    from pipecat.services.google.llm import GoogleLLMService

    google_available = True
except Exception:
    google_available = False


class _StallingStream:
    """An async iterator that yields nothing and never closes on its own."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(3600)
        raise AssertionError("should never actually resolve")


class _HealthyStream:
    """An async iterator that yields a couple of normal chunks, no stalling."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _text_chunk(text):
    class _Part:
        thought = False
        thought_signature = None
        function_call = None
        inline_data = None

        def __init__(self, t):
            self.text = t

    class _Content:
        def __init__(self, t):
            self.parts = [_Part(t)]

    class _Candidate:
        def __init__(self, t):
            self.content = _Content(t)
            self.grounding_metadata = None

    class _Chunk:
        usage_metadata = None

        def __init__(self, t):
            self.candidates = [_Candidate(t)]

    return _Chunk(text)


@pytest.mark.asyncio
@pytest.mark.skipif(not google_available, reason="Google dependencies not installed")
async def test_stalled_stream_fires_completion_timeout():
    with patch.object(GoogleLLMService, "create_client"):
        service = GoogleLLMService(api_key="test-key", stream_idle_timeout_secs=0.05)

    timeout_fired = False

    @service.event_handler("on_completion_timeout")
    async def _on_timeout(_service):
        nonlocal timeout_fired
        timeout_fired = True

    with patch.object(
        service, "_stream_content_universal_context", return_value=_StallingStream()
    ):
        context = LLMContext([{"role": "user", "content": "hello?"}])
        await service._process_context(context)

    # _call_event_handler dispatches handlers via asyncio.create_task rather
    # than awaiting them inline, so give the scheduled task a chance to run.
    await asyncio.sleep(0.05)

    assert timeout_fired, "Expected on_completion_timeout to fire when the stream stalls"


@pytest.mark.asyncio
@pytest.mark.skipif(not google_available, reason="Google dependencies not installed")
async def test_healthy_stream_is_unaffected():
    with patch.object(GoogleLLMService, "create_client"):
        service = GoogleLLMService(api_key="test-key", stream_idle_timeout_secs=0.05)

    timeout_fired = False

    @service.event_handler("on_completion_timeout")
    async def _on_timeout(_service):
        nonlocal timeout_fired
        timeout_fired = True

    received_text = []

    async def _capture_push_llm_text(text):
        received_text.append(text)

    with patch.object(
        service,
        "_stream_content_universal_context",
        return_value=_HealthyStream([_text_chunk("hello "), _text_chunk("world")]),
    ), patch.object(service, "_push_llm_text", new=_capture_push_llm_text):
        context = LLMContext([{"role": "user", "content": "hi"}])
        await service._process_context(context)

    assert not timeout_fired, "A healthy, non-stalling stream must not trigger a timeout"
    assert received_text == ["hello ", "world"]
