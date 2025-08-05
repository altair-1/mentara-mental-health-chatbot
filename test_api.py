import requests
import json

def send_message(text):
    try:
        response = requests.post(
            "http://localhost:5005/webhooks/rest/webhook", 
            json={"message": text, "sender": "debug_user"}
        )
        print(f"User: {text}")
        results = response.json()
        if results:
            for r in results:
                print(f"Bot: {r.get('text', 'No text response')}")
        else:
            print("Bot: [No response]")
        print("---")
    except Exception as e:
        print(f"Error: {e}")

# Test each scenario
send_message("hello")
send_message("i am feeling really sad today")
send_message("i feel hopeless")
send_message("I'm stressed about my exams")