#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: RimeTTSService must not emit odd-length PCM16 chunks.

Rime chops its base64-encoded PCM stream at arbitrary byte boundaries, so a
"chunk" message can end mid-sample. Pushing that straight into a
TTSAudioRawFrame with no alignment produces an odd-length buffer that crashes
downstream PCM16 processing, after which TTS goes silent for the rest of the
session.
"""

from pipecat.services.rime.tts import RimeTTSService


def _make_service():
    service = RimeTTSService.__new__(RimeTTSService)
    service._audio_remainder = b""
    service._audio_remainder_context_id = None
    return service


def test_dangling_byte_is_held_and_prepended_to_next_chunk():
    service = _make_service()

    original = bytes(range(1, 11))  # 10 bytes = 5 whole 16-bit samples
    first, second = original[:3], original[3:]  # split mid-sample (odd length)

    out1 = service._sample_aligned_audio("ctx-1", first)
    assert len(out1) % 2 == 0, "must never emit an odd-length buffer"
    assert out1 == original[:2]
    assert service._audio_remainder == original[2:3]

    out2 = service._sample_aligned_audio("ctx-1", second)
    assert len(out2) % 2 == 0
    # Everything emitted across both calls must reconstruct the original bytes.
    assert out1 + out2 == original
    assert service._audio_remainder == b""


def test_context_switch_discards_stale_remainder():
    service = _make_service()

    # Leave a dangling byte on ctx-1.
    service._sample_aligned_audio("ctx-1", b"\x01\x02\x03")
    assert service._audio_remainder == b"\x03"

    # A new context must not get ctx-1's leftover byte prepended to its audio.
    out = service._sample_aligned_audio("ctx-2", b"\x10\x20")
    assert out == b"\x10\x20"
    assert service._audio_remainder_context_id == "ctx-2"


def test_already_aligned_chunk_passes_through_with_no_remainder():
    service = _make_service()
    out = service._sample_aligned_audio("ctx-1", b"\x01\x02\x03\x04")
    assert out == b"\x01\x02\x03\x04"
    assert service._audio_remainder == b""


def test_single_byte_chunk_is_fully_held_back():
    service = _make_service()
    out = service._sample_aligned_audio("ctx-1", b"\x01")
    assert out == b""
    assert service._audio_remainder == b"\x01"
