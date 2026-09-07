#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: AWSBedrockLLMService must not drop parallel tool calls.

Bedrock streams each tool call as its own content block, identified by
contentBlockIndex. A single shared accumulator var (rather than one keyed by
index) silently drops all but the last tool call when the model requests
several in the same turn.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.aws.llm import AWSBedrockLLMService


def _fake_stream_events():
    async def _gen():
        # Two tool calls opened back-to-back, then their deltas interleaved,
        # then each closed independently -- this is the pattern that broke
        # the single shared tool_use_block/json_accumulator implementation.
        yield {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "call_0", "name": "get_weather"}},
            }
        }
        yield {
            "contentBlockStart": {
                "contentBlockIndex": 1,
                "start": {"toolUse": {"toolUseId": "call_1", "name": "get_time"}},
            }
        }
        yield {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '{"city": '}},
            }
        }
        yield {
            "contentBlockDelta": {
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '{"tz": "UTC"}'}},
            }
        }
        yield {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '"NYC"}'}},
            }
        }
        yield {"contentBlockStop": {"contentBlockIndex": 1}}
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {
            "messageStop": {"stopReason": "tool_use"},
        }

    return _gen()


@pytest.mark.asyncio
async def test_parallel_tool_calls_are_not_dropped():
    service = AWSBedrockLLMService(
        aws_access_key="test",
        aws_secret_key="test",
        aws_region="us-east-1",
    )

    context = LLMContext([{"role": "user", "content": "what's the weather and time?"}])

    captured_calls = []

    async def _fake_run_function_calls(function_calls):
        captured_calls.extend(function_calls)

    with patch.object(
        service,
        "_create_converse_stream",
        new=AsyncMock(return_value={"stream": _fake_stream_events()}),
    ), patch.object(
        type(service._aws_session), "client"
    ) as mock_client_factory, patch.object(
        service, "run_function_calls", new=_fake_run_function_calls
    ):
        mock_client_factory.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
        mock_client_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        await service._process_context(context)

    assert len(captured_calls) == 2, (
        f"Expected both parallel tool calls to survive, got {len(captured_calls)}: "
        f"{[c.function_name for c in captured_calls]}"
    )

    by_name = {c.function_name: c for c in captured_calls}
    assert set(by_name) == {"get_weather", "get_time"}
    assert by_name["get_weather"].tool_call_id == "call_0"
    assert by_name["get_weather"].arguments == {"city": "NYC"}
    assert by_name["get_time"].tool_call_id == "call_1"
    assert by_name["get_time"].arguments == {"tz": "UTC"}
