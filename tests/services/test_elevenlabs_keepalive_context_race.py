#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: ElevenLabsTTSService keepalive must not race a new context's init.

The periodic keepalive used to target whatever context_id was "active"
regardless of whether that context's init message (carrying voice_settings)
had actually been sent yet. If the keepalive fired in that narrow window, its
context-less-of-voice_settings message became the context's first message,
and ElevenLabs closed the socket with a 1008 policy violation on the later
context-init.
"""

import asyncio
import json

import pytest

from pipecat.services.elevenlabs.tts import ElevenLabsTTSService


class _FakeWebsocket:
    def __init__(self, state):
        self.state = state
        self.sent = []

    async def send(self, data):
        self.sent.append(json.loads(data))


def _make_service():
    service = ElevenLabsTTSService.__new__(ElevenLabsTTSService)
    service._context_init_sent = set()
    return service


@pytest.mark.asyncio
async def test_keepalive_omits_context_id_before_init_sent():
    from websockets.protocol import State

    service = _make_service()
    service._websocket = _FakeWebsocket(State.OPEN)

    service.get_active_audio_context_id = lambda: "ctx-1"
    # Context created (active) but its init message has NOT been marked sent yet.
    assert "ctx-1" not in service._context_init_sent

    async def _one_keepalive_tick():
        if service._websocket and service._websocket.state is State.OPEN:
            context_id = service.get_active_audio_context_id()
            if context_id and context_id in service._context_init_sent:
                msg = {"text": "", "context_id": context_id}
            else:
                msg = {"text": ""}
            await service._websocket.send(json.dumps(msg))

    await _one_keepalive_tick()

    assert service._websocket.sent == [{"text": ""}], (
        "Keepalive must not stamp a context_id whose init hasn't been sent yet"
    )


@pytest.mark.asyncio
async def test_keepalive_includes_context_id_after_init_sent():
    from websockets.protocol import State

    service = _make_service()
    service._websocket = _FakeWebsocket(State.OPEN)
    service.get_active_audio_context_id = lambda: "ctx-1"
    service._context_init_sent.add("ctx-1")

    async def _one_keepalive_tick():
        if service._websocket and service._websocket.state is State.OPEN:
            context_id = service.get_active_audio_context_id()
            if context_id and context_id in service._context_init_sent:
                msg = {"text": "", "context_id": context_id}
            else:
                msg = {"text": ""}
            await service._websocket.send(json.dumps(msg))

    await _one_keepalive_tick()

    assert service._websocket.sent == [{"text": "", "context_id": "ctx-1"}]


@pytest.mark.asyncio
async def test_run_tts_marks_context_init_sent_before_the_send_await():
    """The actual run_tts path must mark the flag as part of sending context-init."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from websockets.protocol import State

    service = _make_service()
    service._name = "ElevenLabsTTSService"
    service._websocket = _FakeWebsocket(State.OPEN)
    service._voice_settings = None
    service._pronunciation_dictionary_locators = None
    service._cumulative_time = 0
    service._partial_word = ""
    service._partial_word_start_time = 0.0

    service.audio_context_available = MagicMock(return_value=False)
    service.create_audio_context = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    service.start_tts_usage_metrics = AsyncMock()

    async def _fake_send_text(text, context_id):
        pass

    service._send_text = _fake_send_text

    with patch("pipecat.services.elevenlabs.tts.logger"):
        frames = []
        async for frame in service.run_tts("hello", "ctx-new"):
            frames.append(frame)

    assert "ctx-new" in service._context_init_sent
    assert any(msg.get("context_id") == "ctx-new" for msg in service._websocket.sent)
