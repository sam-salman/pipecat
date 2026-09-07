#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: realtime LLM services must not double-encode tool results.

_send_tool_result unconditionally json.dumps()'d the result before sending it
as the ConversationItem's `output` field. When the tool's result was already a
string (as opposed to a dict most Zoren tools return), this wrapped it in an
extra layer of JSON encoding, so the model saw an escaped JSON string literal
instead of the actual result text.
"""

from unittest.mock import AsyncMock

import pytest

from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService


@pytest.mark.asyncio
async def test_openai_realtime_string_result_is_not_double_encoded():
    service = OpenAIRealtimeLLMService.__new__(OpenAIRealtimeLLMService)
    sent = []

    async def _fake_send_client_event(event):
        sent.append(event)

    service.send_client_event = _fake_send_client_event

    await service._send_tool_result("call-1", "The weather is sunny, 72F.")

    assert len(sent) == 1
    assert sent[0].item.output == "The weather is sunny, 72F."


@pytest.mark.asyncio
async def test_openai_realtime_dict_result_is_still_json_encoded():
    service = OpenAIRealtimeLLMService.__new__(OpenAIRealtimeLLMService)
    sent = []

    async def _fake_send_client_event(event):
        sent.append(event)

    service.send_client_event = _fake_send_client_event

    await service._send_tool_result("call-1", {"status": "success", "temp_f": 72})

    assert len(sent) == 1
    assert sent[0].item.output == '{"status": "success", "temp_f": 72}'


@pytest.mark.asyncio
async def test_grok_realtime_string_result_is_not_double_encoded():
    service = GrokRealtimeLLMService.__new__(GrokRealtimeLLMService)
    sent = []

    async def _fake_send_client_event(event):
        sent.append(event)

    service.send_client_event = _fake_send_client_event

    await service._send_tool_result("call-1", "The weather is sunny, 72F.")

    assert len(sent) == 1
    assert sent[0].item.output == "The weather is sunny, 72F."


@pytest.mark.asyncio
async def test_grok_realtime_dict_result_is_still_json_encoded():
    service = GrokRealtimeLLMService.__new__(GrokRealtimeLLMService)
    sent = []

    async def _fake_send_client_event(event):
        sent.append(event)

    service.send_client_event = _fake_send_client_event

    await service._send_tool_result("call-1", {"status": "success", "temp_f": 72})

    assert len(sent) == 1
    assert sent[0].item.output == '{"status": "success", "temp_f": 72}'
