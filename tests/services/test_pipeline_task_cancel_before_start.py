#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test: PipelineTask.cancel() must not hang before StartFrame reaches the sink.

_process_push_queue() blocks on _wait_for_pipeline_start(), which waits for
_pipeline_start_event -- only set once StartFrame reaches the pipeline sink.
cancel() queues a CancelFrame onto that same push queue, but if a processor
blocks during its own startup (so StartFrame never reaches the sink), that
queue is never drained: cancel() (and thus run()) would hang forever.
"""

import asyncio

import pytest

from pipecat.frames.frames import EndFrame, Frame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@pytest.mark.asyncio
async def test_cancel_sets_pipeline_start_event_when_not_yet_started():
    """Direct check of the fixed mechanism: cancel() unblocks a startup wait."""
    task = PipelineTask(Pipeline([]))

    assert not task._pipeline_start_event.is_set()

    await task._cancel()

    assert task._pipeline_start_event.is_set(), (
        "cancel() must set _pipeline_start_event so _process_push_queue, if "
        "still blocked waiting for StartFrame to reach the sink, can drain "
        "the CancelFrame instead of hanging forever"
    )


class _NeverStartsProcessor(FrameProcessor):
    """A processor whose start() never returns, so StartFrame never reaches the sink."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await asyncio.Event().wait()  # blocks forever
        elif isinstance(frame, EndFrame):
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


@pytest.mark.asyncio
async def test_cancel_before_startup_completes_does_not_hang_run():
    """End-to-end: run() must return within cancel_timeout_secs even if a
    processor never finishes handling StartFrame (so CancelFrame can never
    reach the sink either -- that part of the hang is a separate, already-
    bounded fallback via wait_for_cancel's own timeout). Without this fix,
    _process_push_queue never even reaches that fallback: it stays blocked
    forever in _wait_for_pipeline_start, which has no timeout of its own, so
    run() would never return at all, regardless of cancel_timeout_secs.
    """
    task = PipelineTask(Pipeline([_NeverStartsProcessor()]), cancel_timeout_secs=1.0)

    async def _params():
        from pipecat.pipeline.task import PipelineTaskParams

        return PipelineTaskParams(loop=asyncio.get_event_loop())

    run_task = asyncio.create_task(task.run(await _params()))

    # Give run() a moment to reach the blocked StartFrame handling, then
    # cancel before startup ever completes.
    await asyncio.sleep(0.05)
    await task.cancel()

    try:
        await asyncio.wait_for(run_task, timeout=10.0)
    except asyncio.TimeoutError:
        run_task.cancel()
        pytest.fail(
            "PipelineTask.run() hung well past cancel_timeout_secs after "
            "cancel() was called before StartFrame reached the pipeline sink"
        )
