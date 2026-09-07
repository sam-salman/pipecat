#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: DeepgramSTTService must not reconnect mid-utterance on a
settings update.

_update_settings() used to disconnect/reconnect immediately whenever
anything changed, including while the user was actively speaking --
dropping in-flight audio and risking transcription errors. The reconnect is
now deferred until VADUserStoppedSpeakingFrame confirms the user has
stopped.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pipecat.frames.frames import VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.deepgram.stt import DeepgramSTTService, DeepgramSTTSettings


def _make_service():
    service = DeepgramSTTService(api_key="test-key")
    service._connection = MagicMock()
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_settings_update_while_speaking_defers_reconnect():
    service = _make_service()
    service._user_speaking = True

    changed = await service._update_settings(DeepgramSTTSettings(language="es"))

    assert changed
    service._disconnect.assert_not_awaited()
    service._connect.assert_not_awaited()
    assert service._reconnect_pending is True


@pytest.mark.asyncio
async def test_deferred_reconnect_fires_once_user_stops_speaking():
    service = _make_service()
    service._user_speaking = True
    await service._update_settings(DeepgramSTTSettings(language="es"))
    assert service._reconnect_pending is True

    # VADUserStoppedSpeakingFrame flips _user_speaking to False in the base
    # class's process_frame before Deepgram's own handling runs.
    service._connection.send_finalize = AsyncMock()
    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    service._disconnect.assert_awaited_once()
    service._connect.assert_awaited_once()
    assert service._reconnect_pending is False


@pytest.mark.asyncio
async def test_settings_update_while_not_speaking_reconnects_immediately():
    service = _make_service()
    service._user_speaking = False

    changed = await service._update_settings(DeepgramSTTSettings(language="es"))

    assert changed
    service._disconnect.assert_awaited_once()
    service._connect.assert_awaited_once()
    assert service._reconnect_pending is False
