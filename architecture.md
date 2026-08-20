# Architecture — Speak Local, Understand Global

## 1. Tech Stack (confirmed choices)

| Layer | Technology | Notes |
| --- | --- | --- |
| Frontend | React (Vite) + TypeScript | JS-only React is fine too, but TS catches integration bugs early in a 1-month sprint |
| Frontend state | Zustand or Redux Toolkit | Zustand is lighter, recommended for this scope |
| Backend API | Python + FastAPI | Not Node — your ML/speech stack (Whisper, PyTorch, MarianMT) is Python-native; a Node backend would force a second inter-process hop for every inference call |
| Realtime transport | WebSockets (FastAPI native) | Needed for streaming partial transcripts/translations, not plain REST |
| ML/Speech | Whisper (STT) + Librosa (audio preprocessing/feature extraction) | Both run as local inference/processing services in the backend |
| NLP/Translation | IndicTrans2 + MarianMT (both, not either/or) | IndicTrans2 for Indic-language pairs, MarianMT for others/general pairs — route by language pair, or run both and compare per Module 8 evaluation |
| LLM (optional module) | Ollama + Mistral 7B (local) | Only if "offline" truly means no internet at all |
| Database | PostgreSQL | Confirmed working via Homebrew (postgresql@18) + pgAdmin 4 for GUI access. No single-writer queue mitigation needed — Postgres handles concurrent writes natively. |
| ORM | SQLAlchemy + Alembic | Alembic handles migrations — maps to database/migrations |
| Auth | JWT (simple), or skip auth entirely for MVP demo | [Guessing] — confirm if multi-user login is actually required for the demo, or if a session-code join flow is enough |
| Deployment | Docker Compose (frontend, backend, postgres as services) | Matches deployment/ folder |

## 2. High-Level Flow

```
[Browser mic] --(audio chunks, WebSocket)--> [FastAPI backend]
      --> Librosa (preprocessing/noise reduction) --> Whisper (STT) --> raw text + detected language
      --> route by language pair --> IndicTrans2 (Indic pairs) or MarianMT (other pairs) --> translated text
      --> (optional) Ollama LLM --> term explanation
      --> WebSocket push --> [React frontend, live caption UI]
      --> PostgreSQL (async write) --> session/transcript/feedback storage
```

## 3. Folder Structure

```
speak-local-understand-global/
├── prd.md
├── architecture.md
├── rules.md
├── phases.md
├── design.md
├── frontend/
│   └── src/
│       ├── components/     # reusable UI: CaptionBox, LanguageSelector, MicButton, FeedbackWidget
│       ├── pages/           # LectureView, Dashboard, SessionHistory, Login
│       ├── hooks/           # useWebSocket, useMic, useTranslationStream
│       ├── services/        # api.ts, websocket.ts (all backend calls isolated here)
│       ├── store/           # Zustand/Redux state
│       ├── utils/           # formatters, constants
│       └── assets/
├── backend/
│   ├── api/                 # FastAPI route definitions (sessions, translations, feedback)
│   ├── core/                 # config, settings, websocket manager
│   ├── services/             # business logic: session_service, feedback_service
│   ├── ml_models/             # STT/MT/LLM wrapper classes — isolate model calls behind an interface
│   ├── schemas/               # Pydantic request/response models
│   ├── utils/
│   └── tests/
├── database/
│   ├── migrations/            # Alembic migration files
│   └── seeds/                 # seed/demo data scripts
├── data/
│   ├── raw/                   # raw audio samples (dev/testing only, gitignored)
│   └── processed/
├── scripts/                    # setup.sh, download_models.sh, benchmark_latency.py
├── deployment/                  # Dockerfile(s), docker-compose.yml
└── docs/                          # diagrams, architecture images referenced from architecture.md
```

## 4. Key Architectural Rule

**All model inference (Whisper/MT/LLM) sits behind a Python interface in `backend/ml_models/`, never called directly from route handlers.** This is what lets you swap "cut" modules (e.g., add back Module 3/4 later) without touching API or frontend code — treat it as the seam between the "real product" and the "coursework modules."

## 5. Data Model (minimum tables)

- `users` (id, name, role, preferred_language)
- `sessions` (id, lecturer_id, subject, started_at, ended_at)
- `transcript_segments` (id, session_id, source_text, source_lang, translated_text, target_lang, timestamp, latency_ms)
- `feedback` (id, segment_id, user_id, rating, comment)

## 6. Open Question [Guessing]

Whether the backend runs inference on CPU or needs GPU access will determine if Docker Compose is sufficient or if you need a separate inference server. Confirm your deployment hardware before Week 2.
