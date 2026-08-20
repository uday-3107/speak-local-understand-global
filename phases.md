# Phases — 4-Week Plan

**[Guessing]** — durations assume 1 person working full-time or a small team part-time; adjust based on your actual headcount. The order is fixed regardless of team size: get one language pair working end-to-end before adding breadth.

## Decision Gate — Before Week 1 Starts

Confirm the offline definition and language list in `prd.md` §1. Scope is fixed at full-9-module — no cuts. **Add a hard checkpoint at end of Week 1**: if the STT→classification→translation→DL comparison spike isn't producing usable results by then, the team needs to know immediately, not in Week 4.

## Week 1 — Foundation + Core Pipeline Spike

- Set up repo structure, Docker Compose (postgres + backend + frontend skeletons)
- FastAPI skeleton with health-check route, PostgreSQL connection, Alembic init
- React skeleton with Vite, routing, basic layout
- **Spike (highest risk item, do this first)**: get Whisper (STT) → MarianMT/IndicTrans2 (MT) running locally on your actual hardware, one audio file in, translated text out. Measure latency.
- Output of Week 1: a benchmark number for real-time feasibility. If latency is too high for "live" captioning, this is when you pivot to near-real-time (e.g., sentence-by-sentence with a short buffer) instead of word-by-word streaming — decide this now, not in Week 3.

## Week 2 — End-to-End MVP Path

- Backend: WebSocket endpoint streaming audio in, translated text out
- Backend: session/transcript/feedback DB models + migrations
- Frontend: mic capture, WebSocket hook, live caption display component
- Frontend: language selector, basic session start/join flow
- Integration: full loop working for 1 language pair (e.g., English→Hindi), even if rough

## Week 3 — Breadth + Remaining Modules

- Add 2nd/3rd language pair
- Session history page (past transcripts from PostgreSQL)
- Feedback capture (thumbs up/down) wired to DB
- Module 3: train/compare classical ML classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) on language classification, log accuracy/precision/recall/F1
- Module 4: train/compare RNN, LSTM, Transformer models for speech translation, log which performs best offline
- LLM assistant module (Ollama) for term explanation
- Analytics dashboard (latency, BLEU score, ML/DL model comparison tables)

**Note**: Modules 3 and 4 are the highest-risk items in Week 3 — training and comparing multiple model families is itself normally a multi-week task on its own. If Week 1's spike showed tight timing, this is where it will bite. Flag early rather than silently descoping mid-week.

## Week 4 — Hardening + Demo Prep

- Error handling pass (per `rules.md` §3) — WebSocket disconnects, bad audio input, model failures
- Load test: does it survive a 10-minute continuous session without memory leak/crash?
- UI polish pass (see `design.md`)
- Write technical documentation + model evaluation writeup (BLEU scores, accuracy tables — required deliverable per original brief)
- Record demo video as a fallback in case live demo hardware fails
- Final packaging: source code, trained/downloaded model list, docs, evaluation report

## Explicit Risk Log

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Real-time latency too high on demo hardware | [Likely] | Benchmark in Week 1, not Week 4; fall back to near-real-time |
| Scope creep back to full original brief (all 9 modules production-grade) | [Likely] if not actively resisted | Re-check `prd.md` scope table weekly |
| PostgreSQL/backend integration bugs eating Week 2 | [Guessing] | Get the DB connection + one full write/read cycle working Day 1 of Week 1, not deferred |
| Team member unfamiliar with FastAPI/async SQLAlchemy | [Guessing — depends on team] | Flag now if true; changes Week 1 time allocation |
