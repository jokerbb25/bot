import logging
from typing import Any, Optional


_logger = logging.getLogger(__name__)


class _GuiBridge:
    def push_status(self, message: str) -> None:
        try:
            from engine.state import BotState, trade_state

            if (
                trade_state.get("state") == BotState.OPEN
                and message.upper().startswith("NEUTRAL")
            ):
                return
        except Exception:
            pass
        _logger.info("[GUI] %s", message)

    def mark_open(
        self,
        symbol: str,
        direction: str,
        confidence: Optional[float] = None,
        entry: Optional[float] = None,
        sl_pips: Optional[int] = None,
    ) -> None:
        _logger.info(
            "[GUI] OPEN %s %s conf=%s entry=%s sl_pips=%s",
            symbol,
            direction,
            f"{confidence:.2f}" if confidence is not None else "N/A",
            f"{entry:.5f}" if entry is not None else "N/A",
            sl_pips,
        )

    def mark_closed(self, ticket: int, result: str, pnl: float) -> None:
        _logger.info("[GUI] CLOSE ticket=%s result=%s pnl=%.2f", ticket, result, pnl)


gui = _GuiBridge()
