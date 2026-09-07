#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: SmallWebRTC PATCH /api/offer must not crash on Firefox.

handle_patch_request called candidate_from_sdp(c.candidate) unconditionally.
Firefox sends an empty-string candidate as an RFC 8840 end-of-candidates
marker on every media line, which aiortc's candidate_from_sdp can't parse,
raising an AssertionError and breaking the connection.
"""

from unittest.mock import AsyncMock

import pytest

from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequestHandler,
)


def _make_handler_with_connection(pc_id="pc-1"):
    handler = SmallWebRTCRequestHandler()
    peer_connection = AsyncMock()
    handler._pcs_map[pc_id] = peer_connection
    return handler, peer_connection


@pytest.mark.asyncio
async def test_empty_candidate_end_of_candidates_marker_does_not_crash():
    handler, peer_connection = _make_handler_with_connection()

    request = SmallWebRTCPatchRequest(
        pc_id="pc-1",
        candidates=[IceCandidate(candidate="", sdp_mid="0", sdp_mline_index=0)],
    )

    await handler.handle_patch_request(request)

    peer_connection.add_ice_candidate.assert_awaited_once_with(None)


@pytest.mark.asyncio
async def test_real_candidate_still_parsed_normally():
    handler, peer_connection = _make_handler_with_connection()

    real_candidate_sdp = (
        "2299257402 1 udp 2113937151 192.168.1.100 54321 typ host generation 0"
    )
    request = SmallWebRTCPatchRequest(
        pc_id="pc-1",
        candidates=[IceCandidate(candidate=real_candidate_sdp, sdp_mid="0", sdp_mline_index=0)],
    )

    await handler.handle_patch_request(request)

    peer_connection.add_ice_candidate.assert_awaited_once()
    (passed_candidate,), _ = peer_connection.add_ice_candidate.call_args
    assert passed_candidate is not None
    assert passed_candidate.sdpMid == "0"
    assert passed_candidate.sdpMLineIndex == 0
