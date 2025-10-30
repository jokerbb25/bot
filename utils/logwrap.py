from __future__ import annotations

import logging

from ui_bus import bridge

logger = logging.getLogger(__name__)


def skip(reason_code: str, details: str = "") -> None:
    message = f"SKIPPED_{reason_code}"
    if details:
        message = f"{message} | {details}"
    logger.info(message)
    if bridge is not None:
        try:
            bridge.log_signal.emit(message)
        except Exception:
            logger.debug("Failed to emit log message", exc_info=True)
        try:
            bridge.status_signal.emit(message)
        except Exception:
            logger.debug("Failed to emit status message", exc_info=True)
    return None
