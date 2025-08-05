Mentara – Mental Health Chatbot
Mentara is a privacy-focused, locally deployed mental health chatbot designed for Indian students. It integrates Rasa for conversational intelligence and Django for the web interface. The system runs entirely offline, ensuring no data leaves the device.

Features
- Sentiment analysis using NLTK VADER with regex overrides for phrases like "not feeling great"
- Crisis intervention with verified Indian helpline numbers (112, 9152987821, Aasra)
- Five coping techniques: breathing exercises, grounding, muscle relaxation, cognitive restructuring, and journaling prompts
- PHQ-9 and GAD-7 screenings for depression and anxiety symptoms
- Context-aware dialogue handling to avoid repetitive or irrelevant responses

Tech Stack
- Rasa – NLU and dialogue management
- Django – Web server and interface
- NLTK VADER – Sentiment analysis
- SQLite – Development database
- Python 3.8+

System Requirements
- Python 3.8 or higher
- Windows/Linux/macOS
- Minimum 2GB RAM
- Local network access for the Django server

Setup
git clone https://github.com/altair-1/mentara-mental-health-chatbot.git
cd mentara-mental-health-chatbot
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
rasa train --force

Running
Terminal 1:
rasa run actions --debug

Terminal 2:
rasa run --enable-api --cors "*" --debug

Terminal 3:
cd mentara/web
python manage.py runserver

Access the application at http://127.0.0.1:8000/

License
MIT License. See LICENSE file for details.

Repository: https://github.com/altair-1/mentara-mental-health-chatbot