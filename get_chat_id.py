import os
import requests

# NOTE: token previously hardcoded in plaintext here. Rotate the old token via
# @BotFather (it was committed to git history) and set TELEGRAM_TOKEN instead.
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise SystemExit("Set the TELEGRAM_TOKEN environment variable before running this script.")

r = requests.get(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates')
updates = r.json()['result']
if updates:
    chat_id = updates[-1]['message']['chat']['id']
    print(f"Your Chat ID: {chat_id}")
else:
    print("No messages received yet")
