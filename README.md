# Mentara - Mental Health Chatbot

Mentara is a privacy-focused, locally deployed mental health chatbot designed for Indian students. It integrates Rasa for conversational intelligence and Django for the web interface. The system runs entirely offline, ensuring no data leaves the device.

## Features

- **Sentiment Analysis**: NLTK VADER with regex overrides for phrases like "not feeling great"
- **Crisis Intervention**: Verified Indian helpline numbers (112, 9152987821, Aasra)
- **Coping Techniques**: Five evidence-based strategies (breathing exercises, grounding, muscle relaxation, cognitive restructuring, journaling)
- **Clinical Screenings**: PHQ-9 and GAD-7 assessments for depression and anxiety
- **Smart Dialogue**: Context-aware responses to avoid repetition

## Tech Stack

- **Rasa 3.x** - NLU and dialogue management
- **Django 4.x** - Web server and interface  
- **NLTK VADER** - Sentiment analysis
- **SQLite** - Development database
- **Python 3.8+** - Runtime environment

## System Requirements

- Python 3.8 or higher
- Windows/Linux/macOS
- Minimum 2GB RAM
- Local network access for Django server

## Installation

git clone https://github.com/altair-1/mentara-mental-health-chatbot.git

## Create virtual environment
python -m venv venv

## Activate virtual environment
Windows
venv\Scripts\activate

macOS/Linux
source venv/bin/activate

## Install dependencies
pip install -r requirements.txt

Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"

## Train Rasa model
rasa train --force


## Running the Application

Start three separate terminals:

**Terminal 1 - Action Server:**

rasa run actions --debug


**Terminal 2 - Rasa Core:**

rasa run --enable-api --cors "*" --debug


**Terminal 3 - Django Web Server:**

python manage.py runserver

Access the application at: `http://127.0.0.1:8000/`


## License

MIT License. See LICENSE file for details.

## Repository

https://github.com/altair-1/mentara-mental-health-chatbot

## Support

For issues or questions, please open an issue on GitHub.
