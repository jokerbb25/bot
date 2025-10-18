import logging
import time
from typing import Any

import requests

TOKEN = "8300367826:AAGzaMCJRY6pzZEqzjqgzAaRUXC_19KcB60"
CHAT_ID = "8364256476"

BOT_ACTIVE = True


def send_message(message: str) -> None:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=data, timeout=5)
    except Exception as exc:
        logging.error(f"[Telegram Error] {exc}")


def telegram_listener(bot_ref: Any) -> None:
    global BOT_ACTIVE
    offset = None
    while True:
        try:
            response = requests.get(
                f"https://api.telegram.org/bot{TOKEN}/getUpdates",
                params={"offset": offset, "timeout": 10},
                timeout=15,
            )
            payload = response.json()
            if not payload.get("ok"):
                time.sleep(3)
                continue
            for update in payload.get("result", []):
                offset = update.get("update_id", 0) + 1
                message = update.get("message", {}).get("text", "").lower()
                if not message:
                    continue
                if "pause" in message:
                    BOT_ACTIVE = False
                    send_message("⏸ Bot paused via Telegram.")
                elif "resume" in message:
                    BOT_ACTIVE = True
                    send_message("▶ Bot resumed.")
                elif "status" in message:
                    try:
                        accuracy = float(bot_ref.get_accuracy())
                    except Exception as exc:  # pragma: no cover
                        logging.error(f"[Telegram status error] {exc}")
                        accuracy = 0.0
                    send_message(f"📊 Current accuracy: {accuracy:.2f}%")
                elif "info" in message:
                    try:
                        info = bot_ref.get_last_contract_info()
                    except Exception as exc:  # pragma: no cover
                        logging.error(f"[Telegram info error] {exc}")
                        info = "No trades executed yet."
                    send_message(f"📄 Last contract: {info}")
        except Exception as exc:
            logging.error(f"[Telegram listener error] {exc}")
        time.sleep(3)
