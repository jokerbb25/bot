import logging
from typing import Any

from gui import gui


logger = logging.getLogger(__name__)


def skip(reason_code: str, details: str = "") -> Any:
    message = f"SKIPPED_{reason_code}" + (f" | {details}" if details else "")
    logger.info(message)
    gui.push_status(message)
    return False
