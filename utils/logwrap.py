from __future__ import annotations

import logging

import gui

logger = logging.getLogger(__name__)


def skip(reason_code: str, details: str = "") -> None:
    message = f"SKIPPED_{reason_code}"
    if details:
        message = f"{message} | {details}"
    logger.info(message)
    gui.bridge.log_signal.emit(message)
    return None
