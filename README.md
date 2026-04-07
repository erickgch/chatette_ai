# 🐱 Chatette — Personal AI Assistant & Smart Home Manager (v.02)

Chatette is a DIY-spirited, privacy-focused personal AI assistant that runs on your own hardware.
Built for people who want the power of a smart assistant without giving up their privacy or
paying a subscription. Chatette runs on your machine, speaks your language, controls your home,
and works for you — not for an ad platform.

The backend runs on your PC and exposes a local API. A companion Android app connects to it
over your home WiFi. Your notes, reminders, lists and documents never leave your network.

---

## Features

### Personal Assistant

**Reminders**
Tell Chatette to remember anything — errands, ideas, appointments. She saves them locally,
reads them back, manages conflicts and removes them on request. Notifications fire daily
three times per day (default: at 9:00, 12:00 and 18:00).

**Lists**
Create and manage checkbox lists — shopping lists, packing lists, to-do lists.
Tick items off directly from the app. Lists stay available and editable even offline.

**Google Calendar**
Ask what's coming up, add new events and invite attendees — all through natural conversation.
Event notifications fire 30 minutes before each event.

**Agenda View**
Ask for your agenda and Chatette combines reminders and upcoming calendar events into a single
concise briefing in one LLM call.

**Gmail**
Ask Chatette to summarize recent emails. She searches the RAG index first and falls back to
a live Gmail API call if needed.

**Draft Creation**
Ask Chatette to help you write an email, a letter or a message. She generates a draft and
saves it to your documents folder, always in the language you asked in.

**Personal Notes / Journal**
Save personal information, thoughts or anything you want Chatette to remember about you.
She uses these to give more personalized responses.

**Weather**
Ask about today's weather, tomorrow's, or the weekly forecast. Chatette tells you
the weather in a clear and concise way. 

**General Knowledge**
Answer general questions, give recommendations, explain concepts and hold a conversation —
like any AI assistant, but running on your own terms.

---

### Voice

Speak to Chatette using your microphone. She transcribes with Whisper, responds via the LLM,
and speaks back using Piper TTS.

**Supported voice languages:** English (en_US-libritts_r-medium), German (de_DE-ramona-low),
Spanish (es_AR-daniela-high).

Select your language with the flag buttons in the app before pressing the mic.
Chatette always responds in the language you speak or write in.

---

### Smart Home

**Lights — TP-Link Tapo L530 Smart Bulb**
Control the living-room bulb from the app or via voice/chat:
- Turn on / off
- Set brightness
- Set color temperature (white light)
- Set color (full hue wheel)

**TV — Chromecast + YouTube**
Cast content to your TV from the app:
- Open named channels (ZDF, ARD, Euronews, Arte FR/DE, Milenio, TV5 Monde, NHK World)
- Search YouTube and cast directly to the TV
- Power on / off, stop, resume via control bar

**Retro Phone — Raspberry Pi Zero 2 W**
A Raspberry Pi Zero 2 W powers an old stationary phone, turning it into a Chatette hotline.
Hardware buttons trigger intents directly (weather, agenda, emails, lights, timer, alarm)
without needing to speak a command. Chatette responds in the selected language via
Piper TTS streamed to the Pi's audio output.
* See 'Chatette Hotline' project for more information.

---

### Multilingual UI

The companion app supports **English, German and Spanish** throughout — all screens, labels,
shortcuts, notifications and confirmation messages.

---

## Hardware

| Device | Role |
|---|---|
| Windows PC | Backend server (FastAPI, LLM, RAG, TTS, STT) |
| Android phone | Primary companion app (Flutter) |
| Raspberry Pi Zero 2 W | Stationary voice phone — hardware buttons + speaker |
| TP-Link Tapo L530 | Smart RGB bulb (lights control) |
| Google Chromecast | TV casting target |

---

## Architecture

```
Android App (Flutter)          Raspberry Pi Zero 2 W
        │                               │
        │ WiFi (local network)          │ WiFi (local network)
        └──────────────┬────────────────┘
                       ▼
            FastAPI Backend (Python) — PC
                       │
          ┌────────────┼─────────────────┐
          │            │                 │
     RAG Pipeline   Google APIs    Smart Home
   (ChromaDB +     Calendar API    TP-Link Tapo (bulb)
    LangChain)     Gmail API       Chromecast (TV)
          │
          ├── LLM
          │    ├── Groq API (LLaMA 3.3 70B) — default, fast
          │    └── Ollama (local) — fully offline option
          │
          ├── Whisper STT (faster-whisper, runs locally)
          └── Piper TTS (runs locally, EN / DE / ES voices)
```

**RAG (Retrieval-Augmented Generation)**
Your documents — notes, reminders, lists, calendar, emails — are stored in a local ChromaDB
vector database. When you ask a question, relevant chunks are retrieved and sent alongside
your question to the LLM, giving Chatette context about your life without exposing all your
data on every call.

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI, uvicorn |
| LLM | Groq (LLaMA 3.3 70B) or Ollama (local) |
| Embeddings | Ollama + mxbai-embed-large |
| Vector DB | ChromaDB |
| RAG Framework | LangChain |
| Speech-to-Text | faster-whisper (Whisper base, runs locally) |
| Text-to-Speech | Piper TTS (EN / DE / ES voices, runs locally) |
| Scheduler | APScheduler + Watchdog |
| Smart bulb | PyP100 (TP-Link Tapo API) |
| TV casting | PyChromecast + YouTube Data API v3 |
| Calendar / Email | Google Calendar API + Gmail API (OAuth2) |
| Android App | Flutter (Dart) — ask for APK |
| Voice phone | Raspberry Pi Zero 2 W (Python, RPi.GPIO) |

---

## Android App

The companion Android app is not publicly distributed.
If you are interested in trying the full Chatette experience including the app,
feel free to get in touch.

---

## Voice Feature (Optional)

Chatette supports voice input and output using:
- **faster-whisper** for speech-to-text (runs locally)
- **Piper TTS** for text-to-speech (runs locally, EN / DE / ES voices)

Download Piper from [rhasspy/piper releases](https://github.com/rhasspy/piper/releases)
and place `piper.exe` + voice model files in the `piper/` folder, then update your `.env`:

```env
PIPER_PATH=piper/piper.exe
PIPER_VOICE=piper/en_US-libritts_r-medium.onnx
```

Available voice models:

| Language | Model file |
|---|---|
| English | `en_US-libritts_r-medium.onnx` |
| German | `de_DE-ramona-low.onnx` |
| Spanish | `es_AR-daniela-high.onnx` |

---

## Privacy

Your documents, lists, reminders and notes never leave your home network.
Embeddings and vector search run locally via Ollama.

The only data that reaches an external server is your conversation context sent to Groq for
LLM processing. Groq does not use API data for model training, but your queries and personal
context do pass through their infrastructure.

For fully offline operation, switch to a local Ollama model in Settings — no internet required.

---

## Requirements

- **OS:** Windows 10/11 (Linux/Mac untested but should work)
- **Python:** 3.11+
- **Ollama:** https://ollama.com
- **Groq account (free):** https://console.groq.com
- **Google Cloud project** with Calendar and Gmail APIs enabled

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/erickgch/chatette_ai.git
cd chatette_ai
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### 3. Pull the embedding model

```bash
ollama pull mxbai-embed-large
```

### 4. Set up Google OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project
3. Enable **Google Calendar API**, **Gmail API**, and **YouTube Data API v3**
4. Go to **APIs & Services → Credentials**
5. Create an **OAuth 2.0 Client ID** (Desktop app)
6. Download `credentials.json` and place it in the project root
7. Go to **OAuth consent screen → Test users** and add your Gmail address

### 5. Configure your environment

```bash
copy env.template.txt .env    # Windows
cp env.template.txt .env      # Linux/Mac
```

Open `.env` and fill in your details — see `env.template.txt` for all options and comments.

The minimum required fields are:

```env
USER_NAME=YourFirstName
USER_PROFILE=I live in _____. I work as _____. I enjoy _____.
GROQ_API_KEY=your_groq_api_key_here
HOME_CITY=YourCity
HOME_LAT=0.0000
HOME_LON=0.0000
```

### 6. Set up Piper TTS (optional — see [Voice Feature](#voice-feature-optional) section)

### 7. Run Chatette

```bash
python core/api.py
```

Or double-click `start_chatette.bat` on Windows.

Chatette starts on `http://0.0.0.0:8000` and is reachable from any device on your local network.

---

## Project Structure

```
chatette_ai/
├── core/
│   ├── api.py                ← FastAPI endpoints (all HTTP routes)
│   ├── rag_lw.py             ← Intent classification, RAG pipeline, intent handlers
│   ├── ingestion.py          ← Document ingestion into ChromaDB
│   ├── note_manager.py       ← Notes, reminders, lists, drafts (file I/O)
│   ├── sync_manager.py       ← Phone ↔ PC two-way sync (merge logic)
│   ├── scheduler.py          ← Background sync and notification scheduling
│   ├── google_integration.py ← Calendar and Gmail (OAuth2)
│   ├── settings_manager.py   ← Runtime settings (read/write .env)
│   ├── weather.py            ← Open-Meteo weather API
│   ├── bulb_controller.py    ← TP-Link Tapo smart bulb control
│   ├── chat.py               ← Terminal chat (test/debug)
│   └── voice.py              ← Whisper STT + Piper TTS (desktop loop)
├── piper/                    ← Piper executable + voice model files (not in git)
├── data/                     ← User data — notes, ChromaDB, cache (not in git)
├── env.template.txt          ← Configuration template (fill in and rename to .env)
├── requirements.txt
├── start_chatette.bat        ← Windows quick-start
└── stop_chatette.bat         ← Windows quick-stop
```

---

## Chat Modes

| Mode | Behaviour |
|---|---|
| **Auto** | Chatette decides whether to search personal data or answer from general knowledge |
| **Personal** | Always searches your documents, calendar, emails and notes first |
| **General** | Answers purely from the LLM's own knowledge, ignoring personal data |

---

## Intents

Chatette classifies every message into one of these intents before responding:

| Intent | What it does |
|---|---|
| `save_reminder` | Saves a reminder with optional due date |
| `view_reminders` | Lists all active reminders |
| `delete_reminder` | Removes a specific reminder |
| `create_event` | Adds a Google Calendar event |
| `view_events` | Shows upcoming calendar events |
| `delete_event` | Removes a calendar event |
| `view_agenda` | Combined reminders + events overview in one call |
| `view_emails` | Summarises recent emails (RAG first, live fallback) |
| `get_weather` | Today / tomorrow / weekly forecast |
| `create_list` | Creates a new checkbox list |
| `delete_list` | Deletes a list |
| `save_note` | Saves a personal note or journal entry |
| `create_draft` | Generates and saves a written draft |
| `control_bulbs` | Turns lights on / off via chat |
| `about_chatette` | Answers questions about Chatette herself |
| `general` | General knowledge / conversation |

---

## License

MIT License — free to use, modify and distribute.

---

## Disclaimer

Chatette is a personal project in active development, shared for educational and personal use.
Use at your own discretion.
