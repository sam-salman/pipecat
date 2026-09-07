#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: LLMUserAggregator must release its VADController's VAD
analyzer on cleanup.

This is Zoren's actual VAD wiring path (LLMUserAggregatorParams(vad_analyzer=...)),
distinct from the older, deprecated TransportParams(vad_analyzer=...) path on
BaseInputTransport. VADController.cleanup() already called vad_analyzer.cleanup(),
but nothing called VADController.cleanup() -- LLMUserAggregator._cleanup() cleaned
up its other controllers but never its _vad_controller, so the analyzer's worker
thread was never released.
"""

import pytest

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)


class _FakeVAD(VADAnalyzer):
    def num_frames_required(self) -> int:
        return 160

    def voice_confidence(self, buffer: bytes) -> float:
        return 0.0


@pytest.mark.asyncio
async def test_user_aggregator_cleanup_releases_vad_analyzer_executor():
    vad = _FakeVAD(sample_rate=16000, params=VADParams())
    aggregator = LLMUserAggregator(
        LLMContext(),
        params=LLMUserAggregatorParams(vad_analyzer=vad),
    )

    assert aggregator._vad_controller is not None
    assert not vad._executor._shutdown

    await aggregator.cleanup()

    assert vad._executor._shutdown, (
        "Expected LLMUserAggregator.cleanup() to release the VAD analyzer's "
        "worker thread via its VADController"
    )
