#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: InterruptibleTTSService must still reconnect on interruption.

The base TTSService._handle_interruption now also clears _bot_speaking (part
of fixing a pause_frame_processing deadlock). InterruptibleTTSService decides
whether to reconnect based on self._bot_speaking, and checked it AFTER calling
super()._handle_interruption() -- so it would always read False by then and
never reconnect, breaking the reconnect-while-bot-was-speaking behavior this
class exists for.
"""

import asyncio
from typing import AsyncGenerator, Optional
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import Frame, InterruptionFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class _FakeInterruptibleTTSService(InterruptibleTTSService):
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        return
        yield  # pragma: no cover

    async def _connect_websocket(self):
        pass

    async def _disconnect_websocket(self):
        pass

    async def _receive_messages(self):
        pass


def _make_service():
    service = _FakeInterruptibleTTSService()
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_event_loop()))
    service._task_manager = task_manager
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_reconnects_when_bot_was_speaking_at_interruption():
    service = _make_service()
    service._bot_speaking = True

    await service._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    service._disconnect.assert_awaited_once()
    service._connect.assert_awaited_once()

    await service._stop_audio_context_task()


@pytest.mark.asyncio
async def test_does_not_reconnect_when_bot_was_not_speaking():
    service = _make_service()
    service._bot_speaking = False

    await service._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    service._disconnect.assert_not_called()
    service._connect.assert_not_called()

    await service._stop_audio_context_task()
