# Speak Local, Understand Global

**Offline‑first live lecture translation for multilingual classrooms (English, Hindi, Telugu).**
All inference runs locally – no cloud dependency – so the system works in environments with limited or no internet connectivity.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Architecture](#architecture)
5. [Project Structure](#project-structure)
6. [Installation & Setup](#installation--setup)
7. [Usage](#usage)
8. [ML & DL Model Statistics](#ml--dl-model-statistics)
9. [API Documentation](#api-documentation)
10. [Engineering Decisions](#engineering-decisions)
11. [Testing](#testing)
12. [Limitations](#limitations)
13. [Future Improvements](#future-improvements)

---

## Project Overview
The goal is to provide **real‑time, offline captioning and translation** of live lectures. A lecturer’s speech is captured in the browser, streamed via WebSocket to a FastAPI backend where it is:
1. **Transcribed** using Whisper (local inference).
2. **Classified** for language with classic ML models.
3. **Translated** via the best‑available local MT model (IndicTrans2 for Indic‑Indic pairs, NLLB‑200 for other directions).
4. **Optionally enriched** by an on‑device LLM (Ollama Mistral) that provides explanations or study notes.

Results are displayed instantly (or near‑real‑time) to students in their preferred language, stored in PostgreSQL, and available for later review.

---

## Features
- **Live captioning** – audio captured in the browser, streamed over WebSockets, and displayed as translated captions with sub‑second latency for most language pairs.
- **Three supported languages** – English ↔  Hindi ↔  Telugu (all six translation directions).
- **Optimised STT per language** – Whisper `small` for EN/HI, Whisper `large‑v3` for TE (higher accuracy where needed).
- **AI study assistant** – local Ollama Mistral model answers contextual questions; falls back to deterministic templates when the model is offline.
- **Session management** – join‑code workflow, duration tracking, per‑caption feedback, and downloadable transcripts/recordings.
- **Evaluation pipeline (Modules 1‑8)** – speech data collection, preprocessing, ML/DL classification, NLP analysis, offline translation comparison (rule vs AI, BLEU‑4), and end‑to‑end system evaluation with satisfaction analytics.
- **Dashboards** – student (Overview / Live / History / Analytics with Plotly charts) and lecturer (live studio, stats, AI assistant).

---

## Tech Stack
| Layer | Technology |
|---|---|
| **Backend** | Python 3.13 • FastAPI • SQLAlchemy (async) • Alembic • WebSockets |
| **Database** | PostgreSQL 18 |
| **STT** | faster‑whisper (`small`, `large‑v3`) |
| **MT** | IndicTrans2 (`en‑indic‑1B`, `indic‑indic‑1B`) • NLLB‑200‑distilled‑600M • MarianMT |
| **LLM** | Ollama (Mistral 7B) – fully offline |
| **Frontend** | Vanilla HTML/CSS/JS • Tailwind CDN • Plotly (static, no build step) |
| **ML/DL** | scikit‑learn • XGBoost • PyTorch (RNN / LSTM / Transformer) • librosa |

---


## Architecture

```
┌─────────────────┐          ┌──────────────────────────────┐
│  Lecturer UI    │          │       Backend API            │
│ (lecture.html)  │          │   (FastAPI + WebSocket)      │
└────────┬────────┘          └─────────────┬────────────────┘
         │                                 │
         │ Audio Input                     │ Captions + Segments
         │ (WebSocket)                     │ (persisted in Postgres)
         ▼                                 │
┌─────────────────┐                        │
│     Whisper     │                        │
│   STT Model     │                        │
└────────┬────────┘                        │
         │                                 │
         │ Raw text + detected language    │
         ▼                                 ▼
┌─────────────────┐          ┌──────────────────────────────┐
│ Language Router │─────────▶│   Translation Layer          │
│                 │          │  • IndicTrans2 (Indic pairs) │
│                 │          │  • NLLB-200 (other pairs)    │
└─────────────────┘          └─────────────┬────────────────┘
                                           │
                                           │ Translated captions
                                           ▼
┌─────────────────┐          ┌──────────────────────────────┐
│ Student Dashboard│◀────────│      Frontend UI             │
│                 │ Feedback │   (Vanilla HTML/JS)          │
└─────────────────┘          │  • Live captions             │
                             │  • Live chat (Ollama)        │
                             └──────────────────────────────┘
```

### Data Flow

1. **Browser Mic** → audio chunks over WebSocket → **FastAPI Backend**
2. Backend → **Librosa** (pre-processing) → **Whisper** (STT) → raw text + detected language
3. **Language routing**:
   - Indic language pairs → **IndicTrans2**
   - All other pairs → **NLLB-200**
4. Optional: **Ollama (Mistral)** for explanations / study notes
5. Results pushed via WebSocket to the **vanilla frontend** for live caption UI
6. All transcripts and feedback persisted in **PostgreSQL**


## Project Structure
```
speak-local-understand-global/
├── architecture.md          # high‑level design & flow
├── prd.md                  # product requirements
├── phases.md               # 4‑week sprint plan
├── memory.md               # detailed progress tracker
├── backend/                # FastAPI app (api, ml_models, services, schemas, core)
│   ├── api/               # REST + WebSocket endpoints
│   ├── ml_models/         # Whisper, IndicTrans2, NLLB, MarianMT wrappers
│   └── ...
├── frontend/               # static HTML/CSS/JS (no build step)
│   ├── index.html
│   ├── lecture.html
│   └── dashboard.html
├── scripts/                # data download, preprocessing, training, evaluation
├── database/               # Alembic migrations
├── deployment/             # Dockerfiles & docker‑compose.yml
├── data/                   # raw / processed / recordings (git‑ignored)
└── docs/                   # additional diagrams & assets
```

---

## Installation & Setup
### Prerequisites
- macOS / Linux with **Python 3.13** (default path `/opt/anaconda3/bin/python`).
- **PostgreSQL 18** (`brew services start postgresql@18`).
- **Ollama** with the Mistral model pulled (`ollama serve && ollama pull mistral`).
- Hugging Face access for gated IndicTrans2 models.

### Steps
```bash
# 1. Install dependencies
/opt/anaconda3/bin/python -m pip install -r backend/requirements.txt

# 2. Start required services
brew services start postgresql@18
brew services start ollama   # first time will pull the model

# 3. Apply database migrations
cd database && /opt/anaconda3/bin/python -m alembic upgrade head && cd ..

# 4. Run the application (API + static frontend on port 8000)
/opt/anaconda3/bin/python -m uvicorn backend.main:app --port 8000
```
Open **http://localhost:8000** – `lecture.html` for the lecturer studio and `dashboard.html` for the student view.

---

## Usage
1. **Lecturer** opens `lecture.html`, generates a join code, and starts a session.
2. **Students** open `dashboard.html`, enter the join code, and select their preferred language.
3. Audio is streamed to the backend, captioned, translated, and displayed in near‑real‑time.
4. Users can give per‑caption thumbs‑up/down feedback which is stored in PostgreSQL.
5. After a session ends, a transcript and optional audio recording can be downloaded.
6. The AI assistant (via the chat‑panel) can answer contextual questions; if Ollama is offline, deterministic templates are used.

---

## ML & DL Model Statistics
**ML classifiers (Module 3)** – XGBoost achieved the best F1 score of **0.659**; RandomForest 0.655, DecisionTree 0.567, LogisticRegression 0.555.

**Deep learning models (Module 4)** – Transformer achieved **F1 0.936**, LSTM **0.845**, RNN **0.319** on the speech‑translation task.

**MT latency (benchmarks on Apple M4, 10‑core, 16 GB)**:
- IndicTrans2 `en‑indic‑1B`: 1.2‑1.7 s per sentence (EN→HI/TE).
- IndicTrans2 `indic‑indic‑1B`: 1.5‑3.8 s per sentence (HI↔TE).
- NLLB‑200 distilled 600 M: 0.34‑1.7 s per sentence depending on direction.

**LLM assistant** – Ollama Mistral inference averages **12 s** per request (including prompt construction).

> Full statistics are stored in `data/processed/*` and summarized in the analytics endpoint.

---

## API Documentation
| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | Health‑check.
| `/api/v1/sessions` | POST | Create a new lecture session (returns join code).
| `/api/v1/sessions/{id}/join` | POST | Student joins a session by code.
| `/ws/{session_id}` | WebSocket | Stream audio chunks, receive live caption & translation updates.
| `/api/v1/translate` | POST | Direct MT request (IndicTrans2 / NLLB fallback).
| `/api/v1/assistant` | POST | AI study‑assistant query (question → answer).
| `/api/v1/analytics` | GET | System‑level metrics: ML model scores, BLEU, latency, satisfaction.
| `/api/v1/sessions/{id}/transcript` | GET | Download plain‑text transcript of a finished session.

All schemas are defined in `backend/schemas/` and documented in the OpenAPI spec auto‑generated by FastAPI (`/docs`).

---

## Engineering Decisions
- **Backend in Python** – the ML/DL stack (Whisper, PyTorch, Transformers) is native to Python; avoiding a Node backend prevents an extra IPC hop.
- **WebSocket transport** – required for low‑latency streaming of partial transcripts.
- **Model routing** – IndicTrans2 is the default for Indic‑Indic pairs; NLLB‑200 serves as a fallback for EN↔X directions (see `backend/core/config.py`).
- **Vanilla frontend** – kept lightweight, no build step, to satisfy the “offline‑first” constraint and simplify deployment.
- **Docker Compose** – encapsulates Postgres, backend, and frontend services for reproducible local testing.
- **Separate inference layer** – all model calls are wrapped in `backend/ml_models/` so they can be swapped without touching API or UI code (see Architecture section).

---

## Testing
```bash
/opt/anaconda3/bin/python -m pytest backend/tests -q   # 102 tests, requires Postgres
```
The test suite covers API contracts, WebSocket streaming, model wrapper correctness, and the full evaluation pipeline. Continuous integration runs on each push to ensure coverage remains high.

---

## Limitations
- **Real‑time latency** can exceed the 3‑5 s target on low‑powered hardware, especially for Telugu (Whisper `large‑v3`).
- **Model size** – IndicTrans2 models total ~10 GB; initial download is required before offline use.
- **Language coverage** – Only three languages are fully supported at launch; adding new languages requires additional MT checkpoints.
- **Frontend CDN assets** – Tailwind and Plotly are loaded from CDN on first page load; subsequent offline use works once cached.
- **LLM offline guarantee** – If Ollama crashes or the model is missing, the assistant falls back to static templates, reducing interactivity.

---

## Future Improvements
- **GPU acceleration** for Whisper and Transformer models to reduce latency.
- **Expand language support** beyond EN/HI/TE (add more Indic languages and possibly European languages).
- **Full UI redesign** with a component library for better theming and dark‑mode support.
- **Streaming LLM** – integrate a lightweight local LLM that can respond faster than Mistral.
- **Mobile native clients** – iOS/Android apps for on‑device capture without a browser.
- **Automated deployment** to cloud‑off‑grid edge devices (e.g., Raspberry Pi, Jetson) for classroom use in remote areas.


