# Speak Local, Understand Global

**Offline-first live lecture translation for English, Hindi & Telugu classrooms.**

Real-time speech-to-text with Whisper, translated captions via IndicTrans2 / NLLB, and an on-device AI study assistant (Ollama / Mistral) — every model runs locally, no cloud dependency.

> Short description (260 chars): *Offline-first live lecture translation for English, Hindi & Telugu classrooms. Whisper STT + IndicTrans2/NLLB MT + Ollama AI assistant, student/lecturer dashboards, and a full ML/DL/NLP evaluation pipeline — all local, no cloud.*

---

## Features

- **Live captioning** — lecturer speaks; students see translated captions streaming in near real time (sentence-buffered).
- **3 languages** — English (en), Hindi (hi), Telugu (te), all pairs via the best available model:
  - en → hi / te via IndicTrans2 (`en-indic-1B`)
  - hi ↔ te via IndicTrans2 (`indic-indic-1B`)
  - hi / te → en via NLLB-200 fallback
- **STT tuned per language** — Whisper `small` for en/hi, `large-v3` for Telugu (higher accuracy at ~2× cost, still near-real-time).
- **AI study assistant** — chat panel in both student and lecturer views, answered by a local Ollama (Mistral) model grounded in the live captions; degrades gracefully to templates when Ollama is offline.
- **Session flows** — join codes, live sessions with duration tracking, per-caption feedback, recording + transcript download.
- **Evaluation pipeline (Modules 1–8)** — speech data collection, preprocessing, ML/DL classifiers, NLP analysis, offline translation comparison (rule vs AI, BLEU-4), and end-to-end system evaluation with satisfaction analytics.
- **Dashboards** — student (Overview / Live / History / Analytics with Plotly charts) and lecturer (live studio, stats, AI assistant).

## Tech Stack

| Layer | Tech |
| --- | --- |
| Backend | Python 3.13 · FastAPI · SQLAlchemy (async) · Alembic · WebSockets |
| Database | PostgreSQL 18 |
| STT | faster-whisper (`small`, `large-v3`) |
| MT | IndicTrans2 (`ai4bharat/indictrans2-*-1B`) · NLLB-200-distilled-600M · MarianMT |
| LLM | Ollama (Mistral), fully offline |
| Frontend | Vanilla HTML/CSS/JS + Tailwind CDN + Plotly (static, no build step) |
| ML/DL | scikit-learn · XGBoost · PyTorch (RNN / LSTM / Transformer) · librosa |

## Repository Layout

```
backend/            FastAPI app (api, ml_models, services, schemas, models, core)
  api/              REST endpoints + websocket live pipeline + AI assistant
  ml_models/        whisper, indic2, nllb, marian services
frontend/           static site (index, dashboard, lecture) + js/
scripts/            dataset download, preprocessing, training, evaluation, Module 7 assistant
database/           Alembic migrations
deployment/         Dockerfiles, docker-compose.yml, nginx.conf
data/               raw / processed / recordings / cache  (git-ignored)
docs/, *.md         PRD, architecture, design, phases, memory, rules
```

## Prerequisites

- macOS / Linux with Python 3.13 (this project uses `/opt/anaconda3/bin/python`)
- PostgreSQL 18: `brew services start postgresql@18`
- Ollama with a model pulled: `ollama serve` then `ollama pull mistral`
- Hugging Face access for gated IndicTrans2 models (`huggingface-cli login` as `Uday533`)

## Setup

```bash
# 1. Install dependencies
/opt/anaconda3/bin/python -m pip install -r backend/requirements.txt

# 2. Start services
brew services start postgresql@18
brew services start ollama        # ollama pull mistral (first time)

# 3. Apply database migrations
cd database && /opt/anaconda3/bin/python -m alembic upgrade head && cd ..

# 4. Run the server (serves API + static frontend on :8000)
/opt/anaconda3/bin/python -m uvicorn backend.main:app --port 8000
```

Open http://localhost:8000 — **Lecturer studio** (`lecture.html`) to start a session, **Student view** (`dashboard.html`) to join by code.

## Tests

```bash
/opt/anaconda3/bin/python -m pytest backend/tests -q   # 102 tests, needs Postgres running
```

## Pipeline Modules (official 9-step plan)

| # | Module | Status |
| --- | --- | --- |
| 1 | Speech data collection (datasets + recorder) | ✅ |
| 2 | Data preprocessing (clean + VAD + normalization) | ✅ |
| 3 | ML classifiers (LogReg / DT / RF / XGBoost) | ✅ |
| 4 | Deep learning (RNN / LSTM / Transformer speech) | ✅ |
| 5 | NLP tasks (script detection, text-lang ID, embeddings) | ✅ |
| 6 | Offline translation comparison (rule vs AI, BLEU-4) | ✅ |
| 7 | AI translation assistant (STT → MT → LLM) | ✅ live in UI |
| 8 | End-to-end system evaluation | ✅ |
| 9 | Deployment (Docker Compose) | 🚧 written, untested |

## Notes

- Frontend uses Tailwind/Plotly from CDN — needs internet on first load (design samples matched the user's style guide).
- IndicTrans2 transliterates all Indic scripts through Devanagari internally; official pre/post transliteration (via `indic-nlp-library`) is applied so Telugu captions render in Telugu script.
- XGBoost artifacts are saved as native `.json` (joblib round-trip segfaults under pytest).
- See `memory.md` for the full session-by-session history and known gotchas.