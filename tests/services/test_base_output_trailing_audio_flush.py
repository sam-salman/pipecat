#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: BaseOutputTransport must not drop trailing audio.

handle_audio_frame only queues complete audio_chunk_size chunks, buffering
any remainder. Previously that remainder was silently discarded when
_bot_stopped_speaking cleared the buffer on TTSStoppedFrame -- clipping the
tail of nearly every bot utterance that didn't happen to end on an exact
chunk boundary.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from pipecat.frames.frames import TTSAudioRawFrame, TTSStoppedFrame
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams


def _make_sender(chunk_size=160):
    transport = MagicMock()
    params = TransportParams(audio_out_enabled=True)
    sender = BaseOutputTransport.MediaSender(
        transport,
        destination=None,
        sample_rate=16000,
        audio_chunk_size=chunk_size,
        params=params,
    )
    sender._audio_queue = asyncio.Queue()
    sender._resampler = MagicMock()

    async def _identity_resample(audio, in_rate, out_rate):
        return audio

    sender._resampler.resample = _identity_resample
    return sender


@pytest.mark.asyncio
async def test_trailing_audio_shorter_than_chunk_is_flushed_on_tts_stopped():
    sender = _make_sender(chunk_size=160)

    # 100 bytes: less than one chunk (160), so handle_audio_frame buffers it
    # without queuing anything yet.
    trailing = bytes(range(100)) if False else bytes([7] * 100)
    frame = TTSAudioRawFrame(trailing, sample_rate=16000, num_channels=1)
    await sender.handle_audio_frame(frame)

    assert sender._audio_queue.qsize() == 0, "a sub-chunk remainder must not be queued yet"
    assert bytes(sender._audio_buffer) == trailing

    await sender.handle_sync_frame(TTSStoppedFrame())

    # Flushed (padded) audio frame, then the TTSStoppedFrame, in that order.
    assert sender._audio_queue.qsize() == 2
    flushed = await sender._audio_queue.get()
    stopped = await sender._audio_queue.get()

    assert isinstance(flushed, TTSAudioRawFrame)
    assert len(flushed.audio) == 160
    assert flushed.audio[:100] == trailing
    assert flushed.audio[100:] == bytes(60)  # silence padding
    assert isinstance(stopped, TTSStoppedFrame)
    assert bytes(sender._audio_buffer) == b""


@pytest.mark.asyncio
async def test_no_flush_frame_queued_when_buffer_is_empty():
    sender = _make_sender(chunk_size=160)

    # A perfectly chunk-aligned frame leaves nothing buffered.
    frame = TTSAudioRawFrame(bytes([1] * 160), sample_rate=16000, num_channels=1)
    await sender.handle_audio_frame(frame)
    assert sender._audio_queue.qsize() == 1  # the aligned chunk itself
    await sender._audio_queue.get()

    await sender.handle_sync_frame(TTSStoppedFrame())

    assert sender._audio_queue.qsize() == 1
    only = await sender._audio_queue.get()
    assert isinstance(only, TTSStoppedFrame), "no spurious flush frame when buffer is empty"
