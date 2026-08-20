# PRD — Speak Local, Understand Global

## 1. Scope Decision (read first)

**Decision: full original scope, all 9 modules in, nothing cut.** This is the client's explicit instruction, overriding my earlier recommendation to trim Modules 3/4. Recorded for the record: [Likely] this compresses per-module time to roughly 2–3 days each across a 1-month window, which is tight for training/comparing multiple DL architectures (Module 4) on top of a full-stack app. No new information has been given that changes this risk assessment — it is being accepted, not resolved. Revisit at the Week 1 checkpoint in phases.md if velocity doesn't hold.

One item still needs a decision because it's a technical contradiction, not a scope trade-off:

| Decision | Default (change if needed) |
| --- | --- |
| "Offline" means | Confirmed: no internet dependency for the live translation path (STT/MT/LLM all run locally, zero cloud calls during a lecture session). Internet is acceptable only for one-time model downloads and non-critical async syncing (e.g. analytics). [Likely — inferred from brief's own "offline AI deployment" learning outcome, not stated explicitly; revisit if wrong] |
| Languages at launch | English ↔ Hindi ↔ 1 regional language (pick one, e.g. Telugu) — expand further only after MVP path is stable |
| LLM assistant | Small local model via Ollama (e.g. Mistral 7B) — not GPT-4/OpenAI, to keep the offline claim true |

## 2. What to Build

A web app where a lecturer's speech is captured, transcribed, translated in near-real-time, and shown to students in their preferred language, with a chat-style assistant that can explain translated terms.

## 3. Target Users

- **Students** in multilingual classrooms — primary user, needs low-latency, readable translated captions
- **Lecturers/Admins** — secondary — start/stop sessions, view session analytics, manage supported languages

## 4. Core Features (full scope — all in)

1. Audio capture (mic input, browser-based)
2. Speech-to-text (Whisper, local inference)
3. Language detection
4. Classical ML language classification module (Logistic Regression/Decision Tree/Random Forest/XGBoost, compared on accuracy/precision/recall/F1) — Module 3
5. Deep learning speech translation models (RNN, LSTM, Transformer) trained and compared — Module 4
6. Text translation (IndicTrans2 + MarianMT, both integrated — routed by language pair or run in parallel for comparison, local inference) — Module 5/6
7. Live translated caption display (streaming, low-latency UI)
8. LLM-based translation assistant (local model via Ollama) — Module 7
9. Language/preference selection per student
10. Session history (past lecture transcripts + translations, stored in PostgreSQL)
11. Feedback capture (thumbs up/down per translated segment)
12. Translation analytics dashboard (BLEU score, latency, accuracy across all models) — Module 8/9

## 5. Stretch Features (only if all of the above is stable)

- Rule-based vs AI-based translation comparison view
- Multi-speaker support
- Additional languages beyond the initial 3

## 6. Explicitly Out of Scope

- Training custom DL models from scratch for production use
- Mobile native app
- Real cloud/OpenAI LLM usage (breaks "offline" requirement)
- Support for more than 3 languages at launch

## 7. Non-Functional Requirements

- **[Guessing — confirm your hardware]** Real-time captioning target: end-to-end latency under ~3–5s per sentence on the dev machine you'll actually demo on. This number is a placeholder — benchmark Whisper+MarianMT on your actual GPU/CPU in Week 1 before committing to it.
- Must run without internet access after initial model download (true "offline" claim)
- Responsive UI (desktop primary, tablet secondary)

## 8. Success Criteria for Demo Day

- Live mic → translated caption on screen, working end-to-end, for at least 1 language pair, offline
- Session history retrievable from PostgreSQL
- No crash during a 10-minute continuous demo
