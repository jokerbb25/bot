from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def skip(reason_code: str, details: str = "") -> None:
    message = f"SKIPPED_{reason_code}"
    if details:
        message = f"{message} | {details}"
    logger.info(message)
    return None
