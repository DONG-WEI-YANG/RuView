"""Lightweight async event emitter for inter-service communication.

Subscribers are async callbacks invoked via asyncio.create_task.
If a subscriber raises, the error is logged and other subscribers
continue unaffected.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

AsyncHandler = Callable[[Any], Coroutine[Any, Any, None]]


class EventEmitter:
    def __init__(self):
        self._handlers: dict[str, list[AsyncHandler]] = defaultdict(list)

    def on(self, event: str, handler: AsyncHandler) -> None:
        self._handlers[event].append(handler)

    def off(self, event: str, handler: AsyncHandler) -> None:
        handlers = self._handlers.get(event, [])
        if handler in handlers:
            handlers.remove(handler)

    async def emit(self, event: str, data: Any = None) -> None:
        for handler in self._handlers.get(event, []):
            asyncio.create_task(self._safe_call(handler, data, event))

    async def _safe_call(self, handler: AsyncHandler, data: Any, event: str) -> None:
        try:
            await handler(data)
        except Exception:
            logger.exception("Event handler error on '%s'", event)
