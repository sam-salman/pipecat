#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression tests: DeepgramSTTService must not retry forever with silence.

Previously, _connection_handler caught any exception, logged a warning, and
retried in an unbounded `while True` loop -- including permanent failures
(e.g. an invalid API key rejected at the handshake), with no push_error ever
emitted. STT would silently never work again for the rest of the call.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from deepgram.core import ApiError

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


def _make_service():
    service = DeepgramSTTService.__new__(DeepgramSTTService)
    service._name = "DeepgramSTTService"
    service._connection = None
    service._connection_task = None
    from pipecat.utils.network import QuickFailureTracker

    service._quick_failure_tracker = QuickFailureTracker(
        min_stable_duration=100.0, max_consecutive_failures=2
    )
    service._connection_settled = asyncio.Event()
    service._sample_rate = 16000
    service._settings = MagicMock()

    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_event_loop()))
    service._task_manager = task_manager

    pushed_errors = []

    async def _fake_push_error(error_msg, exception=None, fatal=False):
        pushed_errors.append((error_msg, fatal))

    service.push_error = _fake_push_error

    def _fake_build_connect_kwargs():
        return {}

    service._build_connect_kwargs = _fake_build_connect_kwargs

    async def _fake_keepalive_handler():
        await asyncio.Event().wait()

    service._keepalive_handler = _fake_keepalive_handler

    return service, pushed_errors


class _FailingConnectCM:
    """An async context manager whose __aenter__ raises immediately."""

    def __init__(self, exc):
        self._exc = exc

    async def __aenter__(self):
        raise self._exc

    async def __aexit__(self, *a):
        return False


@pytest.mark.asyncio
async def test_permanent_4xx_error_gives_up_with_fatal_push_error():
    service, pushed_errors = _make_service()

    api_error = ApiError(status_code=401, body="invalid API key")
    service._client = MagicMock()
    service._client.listen.v1.connect = MagicMock(return_value=_FailingConnectCM(api_error))

    await service._connection_handler()

    assert any(fatal for _, fatal in pushed_errors), (
        f"Expected a fatal push_error on a permanent 4xx failure, got: {pushed_errors}"
    )
    # Only one connect attempt should have been made -- no retry on a 4xx.
    assert service._client.listen.v1.connect.call_count == 1


@pytest.mark.asyncio
async def test_repeated_quick_transient_failures_eventually_give_up():
    service, pushed_errors = _make_service()

    service._client = MagicMock()
    service._client.listen.v1.connect = MagicMock(
        return_value=_FailingConnectCM(ConnectionError("connection reset"))
    )

    await service._connection_handler()

    assert any(fatal for _, fatal in pushed_errors), (
        f"Expected to eventually give up and push a fatal error, got: {pushed_errors}"
    )
    # max_consecutive_failures=2 -> exactly 2 attempts before giving up.
    assert service._client.listen.v1.connect.call_count == 2


@pytest.mark.asyncio
async def test_connect_unblocks_once_settled_even_on_give_up():
    """_connect()'s caller must not hang forever if the handler gives up."""
    service, _ = _make_service()

    service._client = MagicMock()
    api_error = ApiError(status_code=403, body="forbidden")
    service._client.listen.v1.connect = MagicMock(return_value=_FailingConnectCM(api_error))

    service._connection_settled.clear()
    service._connection_task = service.create_task(service._connection_handler())

    await asyncio.wait_for(service._connection_settled.wait(), timeout=5.0)

    await service._connection_task
