# ================================
# Telegram Bot Handler for AXI MT5 Bot
# Created by Nano (GPT-5)
# ================================

import requests
import threading
import logging

TOKEN = "8300367826:AAGzaMCJRY6pzZEqzjqgzAaRUXC_19KcB60"
CHAT_ID = "8364256476"

BOT_ACTIVE = True  # Used to pause/resume trading from Telegram


def send_message(text: str):
    """Non-blocking Telegram send."""

    def _send():
        try:
            requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                data={"chat_id": CHAT_ID, "text": text},
            )
        except Exception as exc:
            logging.error(f"[TELEGRAM ERROR] {exc}")

    threading.Thread(target=_send, daemon=True).start()


def telegram_listener(bot):
    """Listens for Telegram commands without blocking the trading loop."""
    import time

    global BOT_ACTIVE
    offset = None

    while True:
        try:
            response = requests.get(
                f"https://api.telegram.org/bot{TOKEN}/getUpdates",
                params={"offset": offset, "timeout": 10},
                timeout=15,
            )

            data = response.json()
            if not data.get("ok"):
                continue

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message", {}).get("text", "").lower()

                if msg == "":
                    continue

                if "pause" in msg or "pausar" in msg:
                    BOT_ACTIVE = False
                    send_message("⏸ Bot paused manually.")

                elif "resume" in msg or "reanudar" in msg:
                    BOT_ACTIVE = True
                    send_message("▶️ Bot resumed.")

                elif "status" in msg or "estado" in msg:
                    precision = bot.get_accuracy()
                    operations = bot.total_operations
                    send_message(
                        f"📊 Current Status:\n"
                        f"Precision: {precision:.2f}%\n"
                        f"Operations: {operations}"
                    )

        except Exception as exc:
            logging.error(f"[TELEGRAM LISTENER ERROR] {exc}")

        time.sleep(3)
