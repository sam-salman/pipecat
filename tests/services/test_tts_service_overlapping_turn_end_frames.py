#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: TTSService must not drop LLMFullResponseEndFrame for overlapping turns.

With push_text_frames=False (Cartesia's configuration), the original
LLMFullResponseEndFrame arriving via process_frame used to be silently
dropped; a substitute was synthesized later from _maybe_reset_word_timestamps,
gated on a single shared _llm_response_started boolean. If a second turn's
LLMFullResponseStartFrame arrived (through process_frame directly) before an
earlier turn's audio context finished draining (through the serialization
queue), the shared flag would already read True again by the time the first
context's completion ran -- silently swallowing that context's end frame.
"""

from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests import MockTTSService


def _make_service():
    service = MockTTSService(push_text_frames=False, pause_frame_processing=False)
    service._maybe_pause_frame_processing = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_earlier_context_end_frame_survives_a_later_turn_starting():
    service = _make_service()

    pushed = []

    async def _fake_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    service.push_frame = _fake_push_frame

    # Turn 1 starts and ends -- its end frame gets held (not dropped), keyed
    # by ctx-1, since push_text_frames=False.
    service._turn_context_id = "ctx-1"
    end_frame_1 = LLMFullResponseEndFrame()
    await service.process_frame(end_frame_1, FrameDirection.DOWNSTREAM)

    assert "ctx-1" in service._pending_llm_response_end_frames
    assert service._pending_llm_response_end_frames["ctx-1"] is end_frame_1

    # Turn 2 starts before context ctx-1 has finished draining -- this sets
    # _llm_response_started back to True via LLMFullResponseStartFrame's own
    # handling, which used to be exactly what made the shared-flag version
    # buggy: ctx-1's completion would then see the flag as True and think ITS
    # response was the one that started, or (worse) ctx-2 would silently eat
    # ctx-1's pending frame if keyed wrong.
    start_frame_2 = LLMFullResponseStartFrame()
    await service.process_frame(start_frame_2, FrameDirection.DOWNSTREAM)
    assert service._turn_context_id == "ctx-2" or service._turn_context_id != "ctx-1"

    # ctx-1's audio context now finishes draining.
    await service._maybe_reset_word_timestamps("ctx-1")

    end_frames_pushed = [f for f in pushed if isinstance(f, LLMFullResponseEndFrame)]
    assert len(end_frames_pushed) == 1, (
        f"Expected ctx-1's end frame to be pushed exactly once, got: {end_frames_pushed}"
    )
    assert end_frames_pushed[0] is end_frame_1, "The original frame must be re-pushed, not a new one"
    assert "ctx-1" not in service._pending_llm_response_end_frames


@pytest.mark.asyncio
async def test_second_context_still_gets_its_own_end_frame():
    service = _make_service()

    pushed = []

    async def _fake_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    service.push_frame = _fake_push_frame

    service._turn_context_id = "ctx-1"
    end_frame_1 = LLMFullResponseEndFrame()
    await service.process_frame(end_frame_1, FrameDirection.DOWNSTREAM)

    service._turn_context_id = "ctx-2"
    end_frame_2 = LLMFullResponseEndFrame()
    await service.process_frame(end_frame_2, FrameDirection.DOWNSTREAM)

    assert set(service._pending_llm_response_end_frames.keys()) == {"ctx-1", "ctx-2"}

    await service._maybe_reset_word_timestamps("ctx-2")
    await service._maybe_reset_word_timestamps("ctx-1")

    end_frames_pushed = [f for f in pushed if isinstance(f, LLMFullResponseEndFrame)]
    assert end_frames_pushed == [end_frame_2, end_frame_1], (
        "Each context must get its own original end frame, regardless of "
        "completion order"
    )
