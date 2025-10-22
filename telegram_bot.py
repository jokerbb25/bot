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
                if any(keyword in message for keyword in ("pause", "pausar")):
                    BOT_ACTIVE = False
                    send_message("⏸ Bot pausado manualmente.")
                elif any(keyword in message for keyword in ("resume", "reanudar")):
                    BOT_ACTIVE = True
                    send_message("▶️ Bot reanudado.")
                elif any(keyword in message for keyword in ("status", "estado")):
                    try:
                        accuracy = float(bot_ref.get_accuracy())
                        operations = int(getattr(bot_ref, "total_operations", 0))
                    except Exception as exc:  # pragma: no cover
                        logging.error(f"[Telegram status error] {exc}")
                        accuracy = 0.0
                        operations = 0
                    send_message(
                        f"📊 Precisión actual: {accuracy:.2f}% | Operaciones: {operations}"
                    )
                elif any(keyword in message for keyword in ("info",)):
                    try:
                        info = bot_ref.get_last_contract_info()
                    except Exception as exc:  # pragma: no cover
                        logging.error(f"[Telegram info error] {exc}")
                        info = "Sin operaciones registradas."
                    send_message(f"📄 Último contrato: {info}")
                elif "ayuda" in message:
                    send_message("🧠 Comandos: pausar, reanudar, estado, info")
        except Exception as exc:
            logging.error(f"[Telegram listener error] {exc}")
        time.sleep(3)
