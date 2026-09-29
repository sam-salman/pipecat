#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression tests: GrokRealtimeLLMService interruptions under server VAD.

_handle_interruption only sent response.cancel when server VAD was OFF, so under
server_vad (how Zoren configures Grok) interrupting the bot never cancelled its
in-flight response on the wire. _truncate_current_audio_response was also a
no-op, so server-side history kept audio the caller never heard.

Sending response.cancel unconditionally makes a benign "no active response"
server error routine (the bot often finished speaking before the interruption).
Grok doesn't reliably send the response_cancel_not_active code for it, so the
receive loop must also recognise it by message text -- otherwise it reaches
push_error (which Zoren's on_pipeline_error turns into an ended call) and the
receive loop exits.
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pipecat.services.xai.realtime import events
from pipecat.services.xai.realtime.llm import CurrentAudioResponse, GrokRealtimeLLMService


def _make_service(turn_detection_type):
    service = GrokRealtimeLLMService.__new__(GrokRealtimeLLMService)
    service._name = "GrokRealtimeLLMService"
    turn_detection = SimpleNamespace(type=turn_detection_type) if turn_detection_type else None
    service._settings = SimpleNamespace(
        session_properties=SimpleNamespace(turn_detection=turn_detection)
    )
    service._current_audio_response = None
    service._current_assistant_response = None
    service.sent = []

    async def _fake_send(event):
        service.sent.append(event)

    service.send_client_event = _fake_send
    service.stop_all_metrics = AsyncMock()
    service.push_frame = AsyncMock()
    service.push_error = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_server_vad_interruption_cancels_response_without_clearing_input():
    service = _make_service("server_vad")

    await service._handle_interruption()

    sent_types = [type(e) for e in service.sent]
    assert events.ResponseCancelEvent in sent_types, (
        "Under server VAD, an interruption must still cancel the in-flight response"
    )
    assert events.InputAudioBufferClearEvent not in sent_types, (
        "Under server VAD, the interrupting user speech must not be wiped"
    )


@pytest.mark.asyncio
async def test_manual_turn_interruption_clears_input_and_cancels():
    service = _make_service(None)

    await service._handle_interruption()

    sent_types = [type(e) for e in service.sent]
    assert events.InputAudioBufferClearEvent in sent_types
    assert events.ResponseCancelEvent in sent_types


@pytest.mark.asyncio
async def test_truncate_sends_item_truncate_at_what_the_caller_heard():
    service = _make_service("server_vad")
    service._get_output_sample_rate = lambda: 16000
    # 1s of 16kHz 16-bit mono audio received, but only ~200ms has elapsed.
    service._current_audio_response = CurrentAudioResponse(
        item_id="item-1",
        content_index=0,
        start_time_ms=int(time.time() * 1000) - 200,
        total_size=32000,
    )

    await service._truncate_current_audio_response()

    truncates = [e for e in service.sent if isinstance(e, events.ConversationItemTruncateEvent)]
    assert len(truncates) == 1
    assert truncates[0].item_id == "item-1"
    assert truncates[0].content_index == 0
    # Truncated at elapsed wall-clock time (what was heard), not the full 1000ms.
    assert 150 <= truncates[0].audio_end_ms <= 400
    assert service._current_audio_response is None


class _FakeWebsocket:
    def __init__(self, messages):
        self._messages = [json.dumps(m) for m in messages]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


def _error_event(code, message):
    return {
        "type": "error",
        "event_id": "evt-1",
        "error": {"type": "invalid_request_error", "code": code, "message": message},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,message",
    [
        (None, "Cancellation failed: no active response found"),
        ("some_other_code", "Conversation already has an active response"),
        (None, "Audio content truncation failed"),
        ("response_cancel_not_active", "Cancellation failed"),
    ],
)
async def test_benign_cancel_and_truncate_races_do_not_push_error(code, message):
    service = _make_service("server_vad")
    service._websocket = _FakeWebsocket(
        [_error_event(code, message), {"type": "ping", "event_id": "evt-2"}]
    )

    await service._receive_task_handler()

    service.push_error.assert_not_called()


@pytest.mark.asyncio
async def test_real_server_error_still_pushes_error():
    service = _make_service("server_vad")
    service._websocket = _FakeWebsocket([_error_event("invalid_api_key", "Invalid API key")])

    await service._receive_task_handler()

    service.push_error.assert_awaited_once()
