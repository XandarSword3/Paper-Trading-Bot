import os
import requests

# NOTE: token/chat id previously hardcoded in plaintext here. Rotate the old
# token via @BotFather (it was committed to git history) and set these as
# environment variables / GitHub secrets instead.
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise SystemExit(
        "Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID environment variables before running this script."
    )

r = requests.post(
    f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',
    json={
        'chat_id': TELEGRAM_CHAT_ID,
        'text': '✅ Telegram bot connected! Ready for live trading notifications.',
        'parse_mode': 'HTML'
    }
)

if r.json().get('ok'):
    print("✅ Message sent to your chat!")
else:
    print(f"❌ Error: {r.json()}")
