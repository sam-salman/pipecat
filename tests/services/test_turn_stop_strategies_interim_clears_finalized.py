#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: turn-stop strategies must not keep a stale finalized flag.

Both SpeechTimeoutUserTurnStopStrategy and TurnAnalyzerUserTurnStopStrategy
set _transcript_finalized=True on a finalized TranscriptionFrame, and only
cleared it on VADUserStartedSpeakingFrame. If an STT service finalizes a
transcript mid-utterance (during a pause too short for VAD to report a stop),
the flag stayed stale for the rest of the turn. Since the strategies use that
flag to skip the STT-latency safety wait, this could trigger the turn before
the tail of the utterance had actually been transcribed -- silently
truncating what the LLM sees, or cutting the user off.
"""

import asyncio

import pytest

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams
from pipecat.utils.time import time_now_iso8601


async def _setup(strategy):
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_event_loop()))
    await strategy.setup(tm)


@pytest.mark.asyncio
async def test_speech_timeout_strategy_clears_stale_finalized_flag_on_interim():
    strategy = SpeechTimeoutUserTurnStopStrategy()
    await _setup(strategy)

    await strategy.process_frame(
        TranscriptionFrame("first part", "user-1", time_now_iso8601(), finalized=True)
    )
    assert strategy._transcript_finalized is True

    await strategy.process_frame(InterimTranscriptionFrame("more speech", "user-1", time_now_iso8601()))

    assert strategy._transcript_finalized is False, (
        "An interim transcript must clear a stale finalized flag from an "
        "earlier mid-utterance finalization"
    )


@pytest.mark.asyncio
async def test_turn_analyzer_strategy_clears_stale_finalized_flag_on_interim():
    from unittest.mock import AsyncMock, MagicMock

    turn_analyzer = MagicMock()
    turn_analyzer.cleanup = AsyncMock()
    strategy = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn_analyzer)
    await _setup(strategy)

    await strategy.process_frame(
        TranscriptionFrame("first part", "user-1", time_now_iso8601(), finalized=True)
    )
    assert strategy._transcript_finalized is True

    await strategy.process_frame(InterimTranscriptionFrame("more speech", "user-1", time_now_iso8601()))

    assert strategy._transcript_finalized is False, (
        "An interim transcript must clear a stale finalized flag from an "
        "earlier mid-utterance finalization"
    )
