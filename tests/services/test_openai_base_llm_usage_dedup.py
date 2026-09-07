#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: BaseOpenAILLMService must report usage metrics once per turn.

OpenAI itself only sends `usage` on the final streamed chunk, but several
OpenAI-compatible backends (Groq, some OpenRouter routes, self-hosted servers
like Speaches) send a cumulative usage snapshot on every chunk. Reporting
immediately on each one used to fire start_llm_usage_metrics -- and thus a
MetricsFrame -- once per chunk. Zoren's PipelineMetricsAggregator sums every
MetricsFrame's token counts per turn, so that would badly over-count usage
for those providers.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService


def _usage_chunk(prompt_tokens, completion_tokens, delta_content=None):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
        model="gpt-test",
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=delta_content, tool_calls=None)
                if delta_content is not None
                else None
            )
        ]
        if delta_content is not None
        else [],
    )


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def aclose(self):
        pass


def _make_service():
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(model="gpt-test")
    service._client = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_cumulative_usage_on_every_chunk_reports_only_once():
    service = _make_service()

    # Simulates a provider (e.g. Groq) sending a growing cumulative snapshot
    # on every chunk instead of only the final one.
    chunks = [
        _usage_chunk(10, 1, delta_content="Hel"),
        _usage_chunk(10, 3, delta_content="lo "),
        _usage_chunk(10, 5, delta_content="there"),
    ]

    async def _fake_stream(context):
        return _FakeStream(chunks)

    service._stream_chat_completions_universal_context = _fake_stream

    reported = []

    async def _fake_start_llm_usage_metrics(tokens):
        reported.append(tokens)

    service.start_llm_usage_metrics = _fake_start_llm_usage_metrics
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()

    context = LLMContext([{"role": "user", "content": "hi"}])
    await service._process_context(context)

    assert len(reported) == 1, (
        f"Expected exactly one usage report per turn, got {len(reported)}: {reported}"
    )
    # The reported value is the LAST chunk's snapshot, not a sum of all three.
    assert reported[0].prompt_tokens == 10
    assert reported[0].completion_tokens == 5
    assert reported[0].total_tokens == 15


@pytest.mark.asyncio
async def test_usage_still_reported_on_mid_stream_cancellation():
    """A turn cut short by interruption/cancellation must still report usage."""
    service = _make_service()

    async def _raising_stream():
        yield _usage_chunk(20, 2, delta_content="Hel")
        raise Exception("simulated interruption mid-stream")

    class _RaisingFakeStream:
        def __aiter__(self):
            return _raising_stream()

        async def aclose(self):
            pass

    async def _fake_stream(context):
        return _RaisingFakeStream()

    service._stream_chat_completions_universal_context = _fake_stream

    reported = []

    async def _fake_start_llm_usage_metrics(tokens):
        reported.append(tokens)

    service.start_llm_usage_metrics = _fake_start_llm_usage_metrics
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()

    context = LLMContext([{"role": "user", "content": "hi"}])
    with pytest.raises(Exception, match="simulated interruption"):
        await service._process_context(context)

    assert len(reported) == 1, (
        "The last known usage snapshot must still be reported even when the "
        "stream is interrupted mid-turn"
    )
    assert reported[0].prompt_tokens == 20
