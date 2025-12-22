import requests
r = requests.get('https://api.telegram.org/bot8489311506:AAGyZli23sqDU6D8_VD_TJw6cq_XT0EdgL0/getUpdates')
updates = r.json()['result']
if updates:
    chat_id = updates[-1]['message']['chat']['id']
    print(f"Your Chat ID: {chat_id}")
else:
    print("No messages received yet")
