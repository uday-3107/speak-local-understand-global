# Rules — What to Use, What to Avoid

## 1. Libraries — Use

- **Frontend**: React 18+, Vite, TypeScript, Zustand, Axios (REST) + native WebSocket API, TailwindCSS
- **Backend**: FastAPI, Pydantic v2, SQLAlchemy 2.0 (async), Alembic, uvicorn
- **Speech/ML**: `openai-whisper` or `faster-whisper` (faster-whisper is CPU-friendlier — prefer it if no GPU), `transformers` (for MarianMT/IndicTrans2), `librosa` for audio preprocessing
- **LLM (if used)**: Ollama Python client — keeps the model fully local
- **Testing**: pytest (backend), Vitest + React Testing Library (frontend)

## 2. Libraries — Avoid

- **No OpenAI/Anthropic/cloud LLM APIs** — breaks the offline requirement; if you want a cloud fallback for a stretch demo, gate it behind a feature flag and document clearly that it requires internet
- **No Redux (plain) or Redux-Saga** — too much boilerplate for a 1-month timeline; use Redux Toolkit or Zustand only
- **No raw `fetch` scattered across components** — all HTTP/WS calls go through `frontend/src/services/`
- **No ORM-less raw SQL scattered in route handlers** — always through SQLAlchemy models
- **No training deep learning models from scratch for production paths** — pretrained models only, per PRD scope decision. Training-from-scratch code, if written for the learning-outcome writeup, stays isolated in a `research/` or notebook context and never gets imported into `backend/`

## 3. Error Handling

- Every ML inference call (`backend/ml_models/`) must catch and translate library exceptions into typed application errors — never let a raw Whisper/transformers stack trace reach the API response
- WebSocket disconnects must be handled gracefully — no silent failures on the frontend; show a reconnect indicator
- All API responses use a consistent error shape: `{ "error": { "code": "...", "message": "..." } }`
- Audio input validation (empty stream, unsupported format, silence) happens before it reaches the STT model, not after

## 4. Boundaries for AI-Assisted Development (Claude Code / Copilot / etc.)

- AI tools may scaffold boilerplate (CRUD routes, React components, migrations) but **every ML pipeline integration point must be manually reviewed** — a plausible-looking but wrong Whisper/MarianMT integration will fail silently with garbled output, not with an error
- Do not let AI tools invent library APIs or config flags for Whisper/MarianMT/IndicTrans2/Ollama — verify against actual docs before merging; these libraries change fast and training data goes stale
- AI-generated database migrations must be reviewed against `architecture.md`'s data model before applying — do not auto-apply to a shared dev database
- No AI-generated code touches `deployment/` (Dockerfiles, compose files, env secrets) without a human reading every line — this is your offline/security boundary
- Each module's "AI-written vs human-written vs human-reviewed" status should be trackable (e.g., a short note in each service file's docstring) — this matters for your capstone evaluation, since graders will likely ask what you built vs. generated

## 5. Git / Workflow

- Feature branches per module (`feature/stt-pipeline`, `feature/live-caption-ui`)
- No committing model weights or `data/raw/` audio files — `.gitignore` these; use `scripts/download_models.sh` instead
- No committing `.env` files — use `.env.example`
