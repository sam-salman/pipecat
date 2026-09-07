#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression tests: VAD and turn analyzers must not leak a worker thread.

Each analyzer owns a ThreadPoolExecutor(max_workers=1) for blocking model
inference, but cleanup() used to be a no-op and nothing called it anyway --
every session leaked one thread per analyzer for the life of the process.
"""

from unittest.mock import AsyncMock

import pytest

from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import TransportParams


class _FakeVAD(VADAnalyzer):
    def num_frames_required(self) -> int:
        return 160

    def voice_confidence(self, buffer: bytes) -> float:
        return 0.0


class _FakeSmartTurn(BaseSmartTurn):
    def _predict_endpoint(self, audio_array):
        return {"prediction": 0, "probability": 0.0}


@pytest.mark.asyncio
async def test_vad_analyzer_cleanup_shuts_down_executor():
    vad = _FakeVAD(sample_rate=16000, params=VADParams())
    assert not vad._executor._shutdown
    await vad.cleanup()
    assert vad._executor._shutdown


@pytest.mark.asyncio
async def test_smart_turn_cleanup_shuts_down_executor():
    turn = _FakeSmartTurn(sample_rate=16000)
    assert not turn._executor._shutdown
    await turn.cleanup()
    assert turn._executor._shutdown


@pytest.mark.asyncio
async def test_base_input_transport_cleanup_calls_analyzer_cleanup():
    """The actual wiring: nothing calls analyzer.cleanup() without this."""
    vad = _FakeVAD(sample_rate=16000, params=VADParams())
    turn = _FakeSmartTurn(sample_rate=16000)
    vad.cleanup = AsyncMock(wraps=vad.cleanup)
    turn.cleanup = AsyncMock(wraps=turn.cleanup)

    transport = BaseInputTransport(params=TransportParams(vad_analyzer=vad, turn_analyzer=turn))
    await transport.cleanup()

    vad.cleanup.assert_awaited_once()
    turn.cleanup.assert_awaited_once()
