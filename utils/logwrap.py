from __future__ import annotations

import logging
from typing import Callable, Optional


logger = logging.getLogger(__name__)


class _GuiProxy:
    def __init__(self) -> None:
        self._push_callback: Optional[Callable[[str], None]] = None

    def bind(self, callback: Callable[[str], None]) -> None:
        self._push_callback = callback

    def push_status(self, message: str) -> None:
        if self._push_callback is None:
            return
        try:
            self._push_callback(message)
        except Exception as exc:  # pragma: no cover
            logger.debug("Failed to push GUI status: %s", exc)


gui = _GuiProxy()


def skip(reason_code: str, details: str = "") -> None:
    msg = f"SKIPPED_{reason_code}" + (f" | {details}" if details else "")
    logger.info(msg)
    gui.push_status(msg)
    return None
