import requests
import json

try:
    response = requests.post(
        'http://localhost:5005/webhooks/rest/webhook',
        headers={'Content-Type': 'application/json'},
        data=json.dumps({"message": "hello", "sender": "test"})
    )
    print("Status Code:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)
