#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: SarvamTTSService must emit TTSStoppedFrame on completion.

Previously the service never handled a synthesis-complete signal, so
end-of-turn detection fell back entirely to an idle timer (stop_frame_timeout_s),
visibly lagging turn-taking/barge-in readiness after every bot utterance.
"""

import asyncio
import json

import pytest

from pipecat.frames.frames import TTSAudioRawFrame, TTSStoppedFrame
from pipecat.services.sarvam.tts import SarvamTTSService


class _FakeWebSocket:
    def __init__(self, messages):
        self._messages = list(messages)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


def test_websocket_url_requests_completion_event():
    service = SarvamTTSService(api_key="test-key")
    assert "send_completion_event=true" in service._websocket_url


@pytest.mark.asyncio
async def test_final_event_emits_tts_stopped_frame():
    service = SarvamTTSService(api_key="test-key")
    context_id = "ctx-1"
    # Bypass create_audio_context()'s dependency on the serialization queue
    # (set up during start(), not __init__) -- register the context the same
    # way it does, minus that ordering machinery this test doesn't exercise.
    service._audio_contexts[context_id] = asyncio.Queue()
    service._playing_context_id = context_id  # get_active_audio_context_id()

    messages = [
        json.dumps({"type": "audio", "data": {"audio": "AAAA", "request_id": "r1"}}),
        json.dumps({"type": "event", "data": {"event_type": "final"}}),
    ]
    service._websocket = _FakeWebSocket(messages)

    await service._receive_messages()

    # remove_audio_context() doesn't delete the context synchronously -- it
    # enqueues a None sentinel for whatever's draining the queue to see.
    # Drain it directly to check what was actually appended.
    queue = service._audio_contexts[context_id]
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())

    audio_frames = [i for i in items if isinstance(i, TTSAudioRawFrame)]
    stopped_frames = [i for i in items if isinstance(i, TTSStoppedFrame)]

    assert len(audio_frames) == 1
    assert len(stopped_frames) == 1, (
        f"Expected exactly one TTSStoppedFrame after a 'final' event, got: {items}"
    )
    # The end-of-context sentinel (None) from remove_audio_context() must
    # come after the TTSStoppedFrame, confirming the context was released.
    assert items[-1] is None
    assert items.index(stopped_frames[0]) < len(items) - 1
