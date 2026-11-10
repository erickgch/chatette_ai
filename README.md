# 🐱 Chatette — Personal AI Assistant & Smart Home Manager (v0.3)

Chatette is a DIY-spirited, privacy-focused personal AI assistant that runs on your own hardware.
Built for people who want the power of a smart assistant without giving up their privacy or
paying a subscription. Chatette runs on your machine, speaks your language, controls your home,
and works for you — not for an ad platform.

The backend runs on your PC and exposes an API protected by a secret key. A companion Android
app, a web app and a retro phone connect to it over your home network, or from anywhere via
Tailscale. Since version 0.3 the language model runs entirely on your own PC — your notes,
reminders, lists, documents and questions never leave your network.

---

## Features

### Personal Assistant

**Reminders**
Tell Chatette to remember anything — errands, ideas, appointments. She saves them locally,
reads them back, manages conflicts and removes them on request. Several reminders can be saved
in one message. Notifications fire three times per day (default: at 9:00, 12:00 and 18:00).

**Lists**
Create and manage checkbox lists — shopping lists, packing lists, to-do lists.
Add or remove items by chat, tick them off directly from the app. Lists stay available and
editable even offline.

**Google Calendar**
Ask what's coming up, add new events and invite attendees — all through natural conversation.
Weekdays ("on Friday") are resolved to the right date in code, not by the model. Every new
event is read back for a yes/no confirmation before it is saved.
Event notifications fire 30 minutes before each event.

**Agenda View**
Ask for your agenda and Chatette combines reminders and upcoming calendar events into a single
concise briefing in one LLM call.

**Gmail**
Ask Chatette to summarize recent emails. She searches the RAG index first and falls back to
a live Gmail API call if needed.

**Draft Creation**
Ask Chatette to help you write an email, a letter or a message. She generates a draft and
saves it to your docs folder, always in the language you asked in.

**Personal Notes / Journal**
Save personal information, thoughts or anything you want Chatette to remember about you.
She uses these to give more personalized responses.

**Short-term Memory**
Chatette remembers the last 3 exchanges per device, so follow-up questions ("and tomorrow?")
work. The conversation is forgotten after an hour of silence and on restart.

**Weather**
Ask about the weather now, today, tomorrow, the next few days or the whole week. Chatette
tells you the weather in a clear and concise way.

**Nearby Places**
Ask for a restaurant, café, pharmacy, supermarket and more near your home. Chatette searches
OpenStreetMap — no API key required.

**Timers & Alarms**
Set a countdown ("timer for 10 minutes") or an alarm at a set time ("wake me up at 6:30").

**General Knowledge**
Answer general questions, give recommendations, explain concepts and hold a conversation —
like any AI assistant, but running on your own terms.

---

### Voice

Speak to Chatette using your microphone. She transcribes with Whisper, responds via the LLM,
and speaks back using Piper TTS.

**Supported voice languages:** English (en_US-libritts_r-medium), German (de_DE-kerstin-low),
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
Cast content to your TV from the app or by chat:
- Open named channels (ZDF, ARD, Euronews, Arte FR/DE, Milenio, TV5 Monde, NHK World)
- Search YouTube and cast directly to the TV
- Power on / off, volume, stop, resume via control bar

**Retro Phone — Raspberry Pi Zero 2 W**
A Raspberry Pi Zero 2 W powers an old stationary phone, turning it into a Chatette hotline.
Hardware buttons trigger intents directly (weather, agenda, emails, lights, timer, alarm)
without needing to speak a command. Chatette responds in the selected language via
Piper TTS streamed to the Pi's audio output.
* See 'Chatette Hotline' project for more information.

**Wake-on-LAN**
The Pi also acts as a wake relay: when the PC is asleep or off, the apps ask the Pi to send a
magic packet and the backend wakes up. The relay accepts only requests that carry the
secret key.

---

### Multilingual UI

The companion app and the web app support **English, German and Spanish** throughout — all
screens, labels, shortcuts, notifications and confirmation messages.

---

## Hardware

| Device | Role |
|---|---|
| Windows PC (NVIDIA GPU) | Backend server (FastAPI, LLM, RAG, TTS, STT) |
| Android phone | Primary companion app (Flutter) |
| Any browser | Web app (chat, lists, notes, lights, TV) |
| Raspberry Pi Zero 2 W | Retro phone (Chatette Hotline) and Wake-on-LAN relay |
| TP-Link Tapo L530 | Smart RGB bulb (lights control) |
| Google Chromecast | TV casting target |

---

## Architecture

```
Android App (Flutter)     Web App (browser)     Raspberry Pi Zero 2 W
        │                        │               (retro phone + wake relay)
        │   home network or Tailscale — every request carries the API key
        └────────────────────────┼───────────────────────┘
                                 ▼
                  FastAPI Backend (Python) — PC
                                 │
          ┌──────────────────────┼─────────────────────┐
          │                      │                     │
     RAG Pipeline           Google APIs           Smart Home
   (ChromaDB +              Calendar API          TP-Link Tapo (bulb)
    LangChain)              Gmail API             Chromecast (TV)
          │
          ├── LLM — Ollama (local): qwen3:8b by default
          │         also selectable: qwen2.5:7b, gemma3:4b
          │
          ├── Whisper STT (faster-whisper, runs locally on the GPU)
          └── Piper TTS (runs locally, EN / DE / ES voices)
```

**RAG (Retrieval-Augmented Generation)**
Your documents — notes, reminders, lists, calendar, emails — are stored in a local ChromaDB
vector database. When you ask a question, relevant chunks are retrieved and sent alongside
your question to the LLM, giving Chatette context about your life without exposing all your
data on every call.

**Two-call workflow (Auto RAG selector)**
Every message is first classified into an intent (one LLM call). Actions and queries —
reminders, events, emails, weather, lights — go straight to their handler. For conversation,
a second, short yes/no call decides whether the question is about your personal life:
- **No** → answered from general knowledge (personal data is only added if it is a very close match).
- **Yes** → another yes/no call decides whether emails and calendar are needed. If not, only
  your notes are searched, so they aren't crowded out of the context; if nothing is found
  there, all sources are searched.

In tests this answered 28 of 30 personal questions and 30 of 30 general questions correctly.
Short replies like "yes", "ok thanks" skip the search entirely. The selector is a switch in
Settings; with it off, the chat modes below apply.

**Security**
- Every request must carry the secret key (`X-API-Key` header); requests without it are refused.
- CORS only allows the local web app.
- `tools/firewall_chatette.ps1` restricts port 8000 to the home network and Tailscale.
- Remote access goes through Tailscale (encrypted) — no ports are opened on the router.

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI, uvicorn |
| LLM | Ollama (local) — qwen3:8b by default |
| Embeddings | Ollama + mxbai-embed-large |
| Vector DB | ChromaDB |
| RAG Framework | LangChain |
| Speech-to-Text | faster-whisper (Whisper small, runs locally on the GPU) |
| Text-to-Speech | Piper TTS (EN / DE / ES voices, runs locally) |
| Scheduler | schedule + Watchdog |
| Smart bulb | tapo (TP-Link Tapo API) |
| TV casting | PyChromecast + YouTube Data API v3 |
| Calendar / Email | Google Calendar API + Gmail API (OAuth2) |
| Places | OpenStreetMap Overpass API |
| Weather | Open-Meteo |
| Remote access | Tailscale |
| Android App | Flutter (Dart) — ask for APK |
| Web App | HTML / JavaScript (no build step) |
| Voice phone | Raspberry Pi Zero 2 W (Python, RPi.GPIO) |

---

## Android App

The companion Android app is not publicly distributed.
If you are interested in trying the full Chatette experience including the app,
feel free to get in touch.

---

## Voice Feature (Optional)

Chatette supports voice input and output using:
- **faster-whisper** for speech-to-text (runs locally, needs an NVIDIA GPU with CUDA)
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
| German | `de_DE-kerstin-low.onnx` |
| Spanish | `es_AR-daniela-high.onnx` |

---

## Privacy

Your documents, lists, reminders and notes never leave your home network.
The language model, embeddings, vector search, speech recognition and speech synthesis all
run locally — your questions and personal context are not sent to any AI provider.

Up to version 0.2 Chatette used Groq (cloud) as the default LLM. Since 0.3 Groq is disabled
and everything runs on Ollama.

Chatette only contacts outside services for what they provide: Google (your calendar and
emails), Open-Meteo (weather), OpenStreetMap (nearby places) and YouTube (search for casting).

---

## Requirements

- **OS:** Windows 10/11 (Linux/Mac untested but should work)
- **Python:** 3.11+ (developed on 3.12)
- **Ollama:** https://ollama.com
- **GPU:** NVIDIA with CUDA — for the local LLM and Whisper (8 GB VRAM recommended for qwen3:8b)
- **Google Cloud project** with Calendar and Gmail APIs enabled
- **Tailscale (optional):** https://tailscale.com — to use Chatette away from home

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/erickgch/chatette_ai.git
cd chatette_ai
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### 3. Pull the models

```bash
ollama pull qwen3:8b
ollama pull mxbai-embed-large
```

### 4. Set up Google OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project
3. Enable **Google Calendar API**, **Gmail API**, and **YouTube Data API v3**
4. Go to **APIs & Services → Credentials**
5. Create an **OAuth 2.0 Client ID** (Desktop app)
6. Download the JSON file, save it in the project root as `as-desktop-credentials.json`
   (or set its name in `GOOGLE_CREDENTIALS_FILE`)
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
HOME_CITY=YourCity
HOME_LAT=0.0000
HOME_LON=0.0000
CHATETTE_API_KEY=your_random_key_here
```

Generate the API key with `python -c "import secrets; print(secrets.token_urlsafe(32))"` and
enter the same key in the settings of every app.

### 6. Set up Piper TTS (optional — see [Voice Feature](#voice-feature-optional) section)

### 7. Run Chatette

```bash
python core/api.py
```

Or double-click `start_chatette.bat` on Windows.

Chatette starts on `http://0.0.0.0:8000` and is reachable from any device on your local network
and your Tailscale network. To block everyone else, run once in an administrator PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File tools\firewall_chatette.ps1
```

---

## Project Structure

```
chatette_ai/
├── core/
│   ├── api.py                ← FastAPI endpoints (all HTTP routes, API key check)
│   ├── rag_lw.py             ← Intent classification, RAG pipeline, intent handlers
│   ├── ingestion.py          ← Document ingestion into ChromaDB
│   ├── note_manager.py       ← Notes, reminders, lists, drafts (file I/O)
│   ├── sync_manager.py       ← Phone ↔ PC two-way sync (merge logic)
│   ├── scheduler.py          ← Background sync and notification scheduling
│   ├── google_integration.py ← Calendar and Gmail (OAuth2)
│   ├── settings_manager.py   ← Runtime settings (read/write .env)
│   ├── weather.py            ← Open-Meteo weather API
│   ├── places.py             ← Nearby places (OpenStreetMap Overpass)
│   ├── bulb_controller.py    ← TP-Link Tapo smart bulb control
│   ├── chatette_tv/          ← Chromecast casting, channels, YouTube search
│   ├── dt_utils.py           ← Date/time parsing and time zone helpers
│   ├── chat.py               ← Terminal chat (test/debug)
│   └── voice.py              ← Whisper STT + Piper TTS
├── tools/
│   └── firewall_chatette.ps1     ← Limits port 8000 to home network + Tailscale
├── piper/                    ← Piper executable + voice model files (not in git)
├── data/                     ← User data — docs, ChromaDB, cache (not in git)
├── env.template.txt          ← Configuration template (fill in and rename to .env)
├── PIPELINE.txt              ← Notes on the request pipeline
├── requirements.txt
├── start_chatette.bat        ← Windows quick-start
├── start_chatette_service.bat← Windows start as background service
└── stop_chatette.bat         ← Windows quick-stop
```

---

## Chat Modes

| Mode | Behavior |
|---|---|
| **Auto** | Chatette decides whether to search personal data or answer from general knowledge |
| **Personal** | Always searches your documents, calendar, emails and notes first |
| **General** | Answers purely from the LLM's own knowledge, ignoring personal data |

With the **Auto RAG selector** switched on in Settings, the modes are ignored and Chatette
decides per question (see [Architecture](#architecture)).

**Thinking mode** (Settings, off by default) lets reasoning models like qwen3 think before
answering — more thorough, but noticeably slower.

---

## Intents

Chatette classifies every message into one of these intents before responding:

| Intent | What it does |
|---|---|
| `create_reminder` | Saves one or more reminders, with optional due dates |
| `view_reminders` | Lists all active reminders |
| `delete_reminder` | Removes a specific reminder |
| `create_event` | Adds a Google Calendar event (after a yes/no confirmation) |
| `view_events` | Shows upcoming calendar events |
| `delete_event` | Removes a calendar event |
| `view_agenda` | Combined reminders + events overview in one call |
| `view_emails` | Summarizes recent emails (RAG first, live fallback) |
| `get_weather` | Now / today / tomorrow / next days / weekly forecast |
| `find_places` | Finds nearby restaurants, cafés, shops and more |
| `create_list` | Creates a new checkbox list |
| `add_to_list` | Adds items to an existing list |
| `remove_from_list` | Removes an item from a list |
| `delete_list` | Deletes a list |
| `save_personal_note` | Saves a personal note or journal entry |
| `create_draft` | Generates and saves a written draft |
| `control_bulbs` | Controls the lights via chat (on / off, brightness, color) |
| `set_alarm` | Sets a countdown timer or an alarm at a set time |
| `cast_youtube` | Searches YouTube and plays the result on the TV |
| `cast_channel` | Opens a live TV channel |
| `cast_tv_power` | Turns the TV on or off |
| `cast_volume` | Changes the TV volume |
| `cast_stop` | Stops TV playback |
| `about_chatette` | Answers questions about Chatette herself |
| `general` | General knowledge / conversation (Auto RAG selector decides on personal data) |

---

## License

MIT License — free to use, modify and distribute.

---

## Disclaimer

Chatette is a personal project in active development, shared for educational and personal use.
Use at your own discretion.
