import requests
import json

messages = [
    "i am feeling really sad today",
    "i feel hopeless", 
    "I'm stressed about my exams",
    "I'm stressed",
    "my day is not going well"
]

for msg in messages:
    response = requests.post(
        'http://localhost:5005/model/parse',
        headers={'Content-Type': 'application/json'},
        data=json.dumps({"text": msg})
    )
    result = response.json()
    print(f"Message: '{msg}'")
    print(f"Intent: {result.get('intent', {}).get('name')} (confidence: {result.get('intent', {}).get('confidence', 0):.2f})")
    print("---")