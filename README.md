# 🐱 Chatette — Local AI Personal Assistant

Chatette is a DIY-spirited, privacy-focused personal AI assistant that runs on your own hardware.
Built for people who want the power of a smart personal assistant without giving up
their privacy or paying a subscription. Chatette runs on your machine, speaks your
language, and works for you — not for an ad platform.

The backend runs on your PC and exposes a local API. A companion Android app
connects to it over your home WiFi. Your notes, reminders, lists and documents
never leave your network.

---

## Features

**Reminders**
Tell Chatette to remember anything — errands, ideas, appointments. She saves them
locally and can read them back, manage conflicts and remove them on request.

**Lists**
Create and manage checkbox lists — shopping lists, packing lists, to-do lists.
Tick items off directly from the Android app.

**Google Calendar**
Chatette connects to your Google Calendar. Ask what's coming up, add new events
and invite attendees — all through natural conversation.

**Draft Creation**
Ask Chatette to help you write an email, a letter or a message. She generates a
professional draft and saves it to your documents folder.

**Personal Notes**
Save personal information, thoughts or anything you want Chatette to remember
about you. She uses these to give more personalized responses.

**General Knowledge**
Answer general questions, give recommendations, explain concepts and hold a
conversation — like any AI assistant, but running on your own terms.

**Voice Chat**
Speak to Chatette using your microphone. She transcribes with Whisper, responds
via the LLM, and speaks back using Piper TTS.

**Multilingual UI**
The companion app supports English, German and Spanish. Chatette always responds
in the language you write or speak in.

---

## Architecture

```
Android App (Flutter)
        │
        │ WiFi (local network)
        ▼
FastAPI Backend (Python) ── PC
        │
        ├── RAG Pipeline (LangChain + ChromaDB)
        │       └── Embeddings via Ollama (mxbai-embed-large)
        │
        ├── LLM
        │       ├── Groq API (LLaMA 3.3 70B) — default, fast
        │       └── Ollama (local) — fully offline option
        │
        ├── Google Calendar + Gmail (OAuth2)
        ├── Whisper STT (faster-whisper)
        └── Piper TTS
```

**RAG (Retrieval-Augmented Generation)**
Your documents — notes, reminders, lists, calendar, emails — are stored in a
local ChromaDB vector database. When you ask a question, relevant chunks are
retrieved and sent alongside your question to the LLM, giving Chatette context
about your life.

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI, uvicorn |
| LLM | Groq (LLaMA 3.3 70B) or Ollama (local) |
| Embeddings | Ollama + mxbai-embed-large |
| Vector DB | ChromaDB |
| RAG Framework | LangChain |
| Speech-to-Text | faster-whisper |
| Text-to-Speech | Piper TTS |
| Scheduler | APScheduler + Watchdog |
| Android App (ask for it) | Flutter (Dart) |
| Calendar/Email | Google Calendar API + Gmail API (OAuth2) |

---

## Privacy

Your documents, lists, reminders and notes never leave your home network.
Embeddings and vector search run locally via Ollama.

The only data that reaches an external server is your conversation context,
sent to Groq for processing. Groq does not use API data for model training,
but your queries and personal context do pass through their infrastructure.

For fully offline operation, switch to a local Ollama model in Settings —
no internet required at all.

---

## Requirements

- **OS:** Windows 10/11 (Linux/Mac untested but should work)
- **Python:** 3.11+ (tested with 3.13)
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
3. Enable **Google Calendar API** and **Gmail API**
4. Go to **APIs & Services → Credentials**
5. Create an **OAuth 2.0 Client ID** (Desktop app)
6. Download `credentials.json` and place it in the project root
7. Go to **OAuth consent screen → Test users** and add your Gmail address

### 5. Configure your .env file

```bash
copy env.template.txt .env    # Windows
cp env.template.txt .env      # Linux/Mac
```

Open `.env` and fill in:

```env
USER_NAME=YourFirstName
USER_PROFILE=I live in Berlin. I work as an engineer. I enjoy hiking.
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_CREDENTIALS_FILE=credentials.json
GOOGLE_TOKEN_FILE=token.pickle
```

Get your free Groq API key at [console.groq.com](https://console.groq.com).

### 6. Run Chatette

```bash
python core/api.py
```

Or double-click `start_chatette.bat` on Windows.

Chatette starts on `http://localhost:8000`.

Note: without the APK, Chatette can be tested by running `chat.py` or `voice.py`.

---

## Android App

The companion Android app is not publicly distributed.
If you are interested in trying the full Chatette experience including the app,
feel free to get in touch.

---

## Voice Feature (Optional)

Chatette supports voice input and output using:
- **faster-whisper** for speech-to-text (runs locally)
- **Piper TTS** for text-to-speech (runs locally)

Download Piper from [rhasspy/piper](https://github.com/rhasspy/piper/releases)
and a voice model, then update your `.env`:

```env
PIPER_PATH=piper/piper.exe
PIPER_VOICE=piper/en_US-libritts_r-medium.onnx
```

---

## Project Structure

```
chatette_ai/
├── core/
│   ├── api.py                ← FastAPI endpoints
│   ├── rag.py                ← RAG pipeline, LLM, intent handlers
│   ├── ingestion.py          ← Document ingestion into ChromaDB
│   ├── note_manager.py       ← Notes, reminders, lists, drafts
│   ├── scheduler.py          ← Background sync and notifications
│   ├── sync_manager.py       ← Phone ↔ PC two-way sync
│   ├── google_integration.py ← Calendar and Gmail
│   ├── settings_manager.py   ← Runtime settings
│   ├── chat.py               ← Chat function test
│   └── voice.py              ← Whisper STT + Piper TTS
├── data/
│   └── notes/
│       ├── lists/            ← Checkbox lists
│       └── drafts/           ← Saved drafts
├── env.template.txt          ← Configuration template
├── requirements.txt
├── setup.bat                 ← Windows installer
├── start_chatette.bat
└── stop_chatette.bat
```

---

## Chat Modes

| Mode | Behaviour |
|---|---|
| **Auto** | Chatette decides whether to search personal data or answer from general knowledge |
| **Personal** | Always searches your documents, calendar, emails and notes first |
| **General** | Answers purely from the LLM's own knowledge, ignoring personal data |

---

## Roadmap

- [ ] iOS companion app
- [ ] Wake-on-LAN support
- [ ] Multilingual voice (Piper DE/ES voices)
- [ ] Proactive suggestions
- [ ] Smart home integration

---

## License

MIT License — free to use, modify and distribute.

---

## Disclaimer

Chatette is a personal project in active development, shared for educational
and personal use. Use at your own discretion.
