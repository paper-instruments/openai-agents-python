from __future__ import annotations

import asyncio
from typing import TypeVar

_T = TypeVar("_T")


async def await_task_ignoring_cancellation(task: asyncio.Task[_T]) -> _T:
    """Wait for an owned task to finish even if the current task is cancelled again."""
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()
