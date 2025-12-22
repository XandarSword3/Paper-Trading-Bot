import requests

r = requests.post(
    'https://api.telegram.org/bot8489311506:AAGyZli23sqDU6D8_VD_TJw6cq_XT0EdgL0/sendMessage',
    json={
        'chat_id': 7599276205,
        'text': '✅ Telegram bot connected! Ready for live trading notifications.',
        'parse_mode': 'HTML'
    }
)

if r.json().get('ok'):
    print("✅ Message sent to your chat!")
else:
    print(f"❌ Error: {r.json()}")
