#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: TTSService must not pause frame processing with nothing to resume it.

_maybe_pause_frame_processing used to pause unconditionally whenever
_processing_text and pause_frame_processing were both true, with no check for
whether anything would ever produce the BotStoppedSpeakingFrame that lifts the
pause. A turn whose audio contexts all completed in silence (or produced no
audio at all -- e.g. an empty TTS response) would then latch frame processing
paused forever: a permanent pipeline hang. Deepgram, Rime, and ElevenLabs TTS
(three of Zoren's providers) set pause_frame_processing=True.
"""

from unittest.mock import AsyncMock

import pytest

from pipecat.tests import MockTTSService


def _make_service(**kwargs):
    service = MockTTSService(pause_frame_processing=True, **kwargs)
    service.pause_processing_frames = AsyncMock()
    service.resume_processing_frames = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_does_not_pause_when_nothing_would_resume_it():
    service = _make_service()
    service._processing_text = True
    service._bot_speaking = False
    service._audio_contexts = {}

    await service._maybe_pause_frame_processing()

    service.pause_processing_frames.assert_not_called()


@pytest.mark.asyncio
async def test_pauses_when_bot_is_speaking():
    service = _make_service()
    service._processing_text = True
    service._bot_speaking = True
    service._audio_contexts = {}

    await service._maybe_pause_frame_processing()

    service.pause_processing_frames.assert_awaited_once()


@pytest.mark.asyncio
async def test_pauses_when_an_audio_context_is_still_open():
    service = _make_service()
    service._processing_text = True
    service._bot_speaking = False
    service._audio_contexts = {"ctx-1": object()}

    await service._maybe_pause_frame_processing()

    service.pause_processing_frames.assert_awaited_once()


@pytest.mark.asyncio
async def test_still_does_not_pause_when_not_processing_text():
    service = _make_service()
    service._processing_text = False
    service._bot_speaking = True
    service._audio_contexts = {}

    await service._maybe_pause_frame_processing()

    service.pause_processing_frames.assert_not_called()


@pytest.mark.asyncio
async def test_interruption_resumes_a_latched_pause():
    """_handle_interruption must always resume, in case a pause was somehow taken."""
    import asyncio

    from pipecat.frames.frames import InterruptionFrame
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

    service = _make_service()
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_event_loop()))
    service._task_manager = task_manager

    await service._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    service.resume_processing_frames.assert_awaited_once()

    await service._stop_audio_context_task()
