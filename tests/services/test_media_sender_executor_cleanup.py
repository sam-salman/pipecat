#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: MediaSender must not leak a worker thread per output destination.

Each MediaSender (one per output destination, created for every call on every
transport) owns a ThreadPoolExecutor(max_workers=1) for video-frame resizing,
but BaseOutputTransport had no cleanup() at all -- only inheriting the no-op
FrameProcessor.cleanup() -- so nothing ever shut it down. Every output
destination of every call leaked one thread for the life of the process.
"""

from unittest.mock import AsyncMock

import pytest

from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams


@pytest.mark.asyncio
async def test_media_sender_cleanup_shuts_down_executor():
    sender = BaseOutputTransport.MediaSender(
        BaseOutputTransport(params=TransportParams()),
        destination=None,
        sample_rate=16000,
        audio_chunk_size=160,
        params=TransportParams(),
    )

    assert not sender._executor._shutdown
    await sender.cleanup()
    assert sender._executor._shutdown


@pytest.mark.asyncio
async def test_base_output_transport_cleanup_calls_sender_cleanup():
    """The actual wiring: nothing calls sender.cleanup() without this."""
    transport = BaseOutputTransport(params=TransportParams())
    sender = BaseOutputTransport.MediaSender(
        transport,
        destination=None,
        sample_rate=16000,
        audio_chunk_size=160,
        params=TransportParams(),
    )
    sender.cleanup = AsyncMock(wraps=sender.cleanup)
    transport._media_senders[None] = sender

    await transport.cleanup()

    sender.cleanup.assert_awaited_once()
