"""Manual Telegram configuration test.

Run with: .venv/bin/python test_telegram.py
The application loads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from .env.
"""

from main import send_telegram_message


if __name__ == "__main__":
    success = send_telegram_message("Swanetra Telegram test alert.")
    raise SystemExit(0 if success else 1)