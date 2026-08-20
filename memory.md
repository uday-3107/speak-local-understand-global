# memory.md — Project Progress Tracker

> Live tracking of what is completed and what is currently being worked.
> Update the `## Currently Working` and `## Completed` sections at the end of every session.

## Currently Working

**Session 13 — IndicTrans2 translation engine fixed, wired, user E2E-verified (all 6 en/hi/te directions):**
- **Issues faced during the fix** (in order): (1) NaN positional buffers — transformers 5.x builds the remote model on meta device and its `persistent=False` sinusoidal buffers never materialize → all logits NaN; verified 763/763 weights load exactly, so weights were never the problem. (2) `generate()` broken — 5.x passes `EncoderDecoderCache` objects the remote 4.32-era code can't consume → replaced with manual greedy + beam decode loops using the legacy tuple KV cache. (3) **Wrong model per direction**: `en-indic-1B` is trained En→Indic ONLY — hi→te/te→hi/hi→en produced garbage + degenerate repetition; the correct checkpoint for Indic↔Indic is `indictrans2-indic-indic-1B` (downloaded 4.8GB, cached). X→en would need `indic-en-1B` — NOT downloaded by user choice → routed to NLLB fallback. (4) **Script unification**: even en→te output was Telugu *content* in Devanagari *script* — the TGT vocab is 75k Devanagari words vs ~64 Telugu pieces, because the model transliterates everything through Devanagari internally → added official-style pre/post processing via `indic-nlp-library` 0.92 (`IndicNormalizerFactory`, `indic_tokenize/indic_detokenize`, `UnicodeIndicTransliterator` hi↔te). (5) Per-model class patching — the dynamic-module class is per model id, so tokenizer/`tie_weights` patches must run for each checkpoint loaded.
- **Fixed files**: `backend/ml_models/indic2_service.py` rewritten (dual-model routing by direction, `supports()` rejects X→en, `_preprocess`/`_postprocess` transliteration, `_decode` manual greedy/beam, `_fix_positional_buffers`); `backend/core/config.py` default `translation_backend` changed `"nllb"` → `"indic2"`; `backend/ml_models/__init__.py` routes X→en to `nllb(fallback)`.
- **Verified**: all 4 IndicTrans2 directions against IN22-Gen references (en→hi, en→te, hi→te, te→hi — 1.2–3.8s/sentence MPS); hi→en/te→en via NLLB fallback; **browser E2E confirmed by user (captions now proper translations, better than before)**. STT note: `large-v3` only when lecture language = te (by design); `small` otherwise.
- Full suite **98/98 green**.

**Session 14 — AI study assistant (Module 7) wired live to real Ollama, headless-Chrome E2E-verified:**
- **New**: `backend/api/assistant.py` — `POST /api/v1/assistant` `{question, role, session_id?, context?}`; builds prompt (System: classroom assistant, English, 1-3 sentences, context-only, no markdown) + sends captions context (frontend-provided, else last 6 DB segments via session_id); calls the tested `OllamaLlm` (scripts/ai_translation_assistant, mistral, offline) inside `asyncio.to_thread`; appends each exchange to `data/processed/assistant/live.jsonl`; returns `{answer, model, latency_ms, question}`; **503 `ollama_offline`** when Ollama is down (frontend falls back to templates).
- **Schemas**: `AssistantContextItem` / `AssistantRequest` / `AssistantResponse` in `backend/schemas/common.py`; router registered in `backend/main.py`.
- **Frontend**: `frontend/js/student.js` + `frontend/js/lecturer.js` — `assistantBot()` now async, first calls the API with last 6 captions (lecturer also pulls live session's segments), falls back to the old template answers on error; typing indicator "…" bubble replaced by the answer.
- **Tests**: 3 endpoint tests (monkeypatched sync generate — endpoint runs it via `to_thread`; +503 offline path) in `backend/tests/test_assistant.py`. Full suite **101/101 green**.
- **E2E verified via headless Chrome CDP** (Node 26 built-in WebSocket, remote-debugging-port): dashboard.html asked "What is the latest caption about?" → honest grounded answer ("no captions yet" — LLM refused to invent); lecture.html asked same → answered from the real DB segment ("…'Namaste Dunia' translates to 'Hello World'") — both through browser → API → Ollama (mistral, ~2–8s). Offline-path template fallback tested in unit tests.
- **Gotchas hit**: headless Chrome new-tab only via `PUT /json/new?<url>` (no page tab on start, only browser_ui); stale tabs must be matched by URL; `asyncio.to_thread` means tests must monkeypatch a SYNC function.
- **Session 14b — student "session stopped" popup + transcript download**: `GET /api/v1/sessions/{id}/transcript` (plain-text download: header w/ subject/langs/start/end/caption count + `[HH:MM:SS] (src) … / (tgt) …` rows, `Content-Disposition: attachment`); `frontend/dashboard.html` `#end-modal` (Lecture ended → "Download transcript" anchor + Close); `student.js` `pollSegments()` now also `getSession()` each tick — on `ended_at` flips both live chips to "Ended", clears the poller, shows the modal once (`sessionEnded` flag, reset on rejoin); `api.js` `getSession()` + `transcriptUrl()`. Headless-Chrome E2E PASS: joined live → API-end → modal + Ended chip + transcript content verified → dismiss. Suite **102/102** (+1 transcript test in test_api.py).
- **Session 14c — mic icon outside circle (user bug report)**: root cause = Tailwind CDN preflight sets `svg { display: block }`, so the SVG ignores the button's `text-align: center` and pins to the circle's left edge (measured dx=-20px in headless Chrome). Fixed: `.mic-btn` in `frontend/css/styles.css` → `display:flex; align-items:center; justify-content:center`. Verified dx=0 dy=0 via CDP geometry check.
- Server + Postgres + Ollama left running for user's own browser check. Ollama: `brew services start|stop ollama` (mistral:latest already pulled).

**Environment:** `/opt/anaconda3/bin/python` (conda base, Python 3.13) — user's chosen interpreter. `.venv` gone permanently. Postgres 18 via `brew services start|stop postgresql@18`. Test browser: headless Chrome (`/Applications/Google Chrome.app/.../Google Chrome --headless=new --dump-dom URL` — use `--enable-logging=stderr` to see console errors).

**Server launch gotcha (IMPORTANT):** `(uvicorn &)` and even `nohup … & disown` inside the shell tool get SIGTERM'd when the tool call ends. The ONLY pattern that survives: spawn via Python `subprocess.Popen(..., stdout=log, stderr=log, stdin=DEVNULL, start_new_session=True)` (native setsid). Current stop: `pkill -f "uvicorn backend.main"`. API tests (`test_api.py`) need Postgres running.

**Last worked (session 12):** Full UI revamp to academic-modernist design (Tailwind CDN, same tokens as user's samples) + join-code flow + live-caption fixes, all user-testing loops closed:
- **Landing** `frontend/index.html` (hero: "Get Started"→lecture.html lecturer studio, "Student View"→dashboard.html; no login/register, roles by action)
- **Student page** `frontend/dashboard.html` + `js/student.js`: 4 tabs (Overview/Live/History/Analytics), join-by-code (language picker FIRST, then 6-char code), polls segments 1.5s, thumbs feedback, History expand, Plotly charts from `/api/v1/analytics` (models F1/acc, BLEU AI-vs-rule, latency p50/p95, satisfaction donut), AI assistant chat (demo templates until Ollama wiring)
- **Lecturer studio** `frontend/lecture.html` + `js/lecturer.js`: new 4-step flow — Generate code → form (src lang / translate-to / subject) → Create session (join-code banner + mic + captions appear) → Start session (WS opens, **duration starts now** via `POST /sessions/{id}/start`) → Stop (duration freezes at ended_at, recording download)
- **Join codes**: `sessions.join_code` (6-char, alphabet `023456789ABCDEFGHJKMNPQRSTUVWXYZ` no 1/I/L/O, deterministic from uuid int & (1<<30)-1), backfilled for all existing rows + unique constraint. Migration `b7e3f9a1c2d4` (head). `POST /api/v1/sessions/join` (case-insensitive, rejects ended) — curl-verified.
- **Fixes from user bug reports**: (1) lecturer banner showed leftover-session code (polled DB, picked newest live) → banner only from `session-created` event, stats locked to `sessionCode`; (2) student "hello world" spam = joining sessions with historical test captions → on join, pre-mark all existing segment IDs so ONLY post-join captions stream; (3) page totally blank = SyntaxError: my edit had dropped `refreshStats()`'s closing `{` → brace scans + `node --check` per file (NOT piped — pipe to head masks exit code!) + headless-Chrome console-error check; (4) desktop `app.html` retired → meta-refresh redirect stub to `lecture.html`.
- Also built in this stretch: `GET /api/v1/analytics` (reads gitignored `data/processed/{ml,dl,mt,eval}/*.json`, graceful missing-file degradation, `ml.rows` F1-sorted with families, translation BLEU/latency, satisfaction) + `test_analytics.py` 11 tests. Full suite **98/98 green** (was 82).
- Leftover DB state: ~110 live test sessions ("ws test"/"physics", most with captions) still in DB — student History lists them; 19 zero-caption junk sessions ended via API. Cleanup optional later.

Next session: Docker compose verify (`open -a Docker`). (Ollama assistant wiring = DONE session 14; IndicTrans2 = DONE session 13.)

## Project Modules (official 9-step plan)

Status legend: `[ ]` = not completed · `[x]` = completed (user-verified)

| # | Module | Status | Notes |
| --- | --- | --- | --- |
| 1 | Speech Data Collection | [x] completed (session 8) | datasets downloaded (session 1); frontend recorder + storage built (session 8) |
| 2 | Data Preprocessing | [x] completed (session 8) | pipeline + clean dataset built (session 8); input cache-fixed |
| 3 | Machine Learning (LogReg / DecisionTree / RandomForest / XGBoost + acc/prec/recall/F1) | [x] completed (user-verified, session 9) | XGBoost best acc 0.661/F1 0.659 (RF 0.655, DT 0.567, LogReg 0.555) |
| 4 | Deep Learning (RNN / LSTM / Transformer speech) | [x] completed (user-verified, session 9) | Transformer best acc 0.936/F1 0.936; LSTM 0.849; RNN 0.324 |
| 5 | NLP tasks (by language range) | [x] completed (user-verified, session 9) | script detection + stats + text-lang-ID 1.0; **gaps closed**: NLLB multilingual embeddings + rule-based refinement |
| 6 | Offline Translation Engine (rule-based vs AI-based comparison) | [x] completed (user-verified, session 9) | rule BLEU ≈0 (instant, cov 0.05–0.27); AI NLLB BLEU 0.17–0.19 (hi-eng) / 0.024 (en→te); AI wins quality by ~100× |
| 7 | AI Translation Assistant (STT → MT → prompt → LLM pipeline) | [x] completed (user-verified, session 9) | Whisper → NLLB → Ollama/Mistral; NLLB draft always displayed (Option A), LLM polish + English explanation/study note recorded; latency 1.4s + 2.1s + ~12s; **live UI wiring DONE session 14** (`POST /api/v1/assistant`, dashboard + lecture chat use real Ollama, template fallback when offline) |
| 8 | Evaluation (BLEU etc.) | [x] completed (user-verified, session 9) | ML consolidated (Transformer best F1 0.936); translation BLEU + script-consistency 100% + latency; assistant quality; satisfaction 47/47 positive |
| 9 | UI Dashboard | [x] completed (session 12, user-verified in testing loops) | landing + student dashboard + lecturer studio, analytics API + Plotly charts, join codes |
| — | Hardening / Docker / docs | [ ] not completed | compose + Dockerfiles exist but untested |

## Completed

| When | What | Where |
| --- | --- | --- |
| session 1 | Reviewed all project docs (prd, phases, architecture, rules, design) | `.`, docs/ |
| session 1 | Created venv, installed `datasets` / `huggingface_hub` | `.venv/` |
| session 1 | Added `.gitignore` (data/raw, .venv, .env, caches, weights) | `.gitignore` |
| session 1 | Wrote dataset download pipeline (resumable, cache-cleaning) | `scripts/download_datasets.py` |
| session 1 | Downloaded Common Voice hi/te (v17 mirror, audio bytes embedded) | `data/raw/common_voice/` |
| session 1 | Downloaded FLEURS hi_in/te_in/en_us (text parquet + audio tars) | `data/raw/fleurs/` |
| session 1 | Downloaded IN22-Gen full 22-lang test set + 4 pair extracts | `data/raw/in22_gen/` |
| session 1 | Downloaded BPCC eng→hin/tel (seed, nllb, samanantar v0.3 + v2) | `data/raw/bpcc/` |
| session 1 | Wrote data manifest (rows/sizes/licenses) | `data/raw/MANIFEST.json` |
| session 2 | Created `memory.md` progress tracker | `memory.md` |
| session 2 | Cleaned HF cache + leftover temp files (verified gone) | `~/.cache/huggingface/` |
| session 2 | Fixed FLEURS parquet column bug (tsv had no header) | `data/raw/fleurs/` |
| session 2 | Extracted 5 hi + 3 te + 3 en test clips | `data/processed/test_audio/` |
| session 2 | Installed ML stack: faster-whisper, torch 2.13 (MPS), transformers 5.14, librosa | `.venv/` |
| session 2 | **Week 1 latency spike** — results below | `scripts/benchmark_latency.py` |
| session 3 | Backend skeleton: config, async DB, errors, models, schemas, services, API | `backend/` |
| session 3 | Alembic init + initial migration applied (Postgres 18, `speak_local`) | `database/` |
| session 3 | API verified: `/api/v1/health` ok, sessions/segments/feedback CRUD persist | `backend/api/` |
| session 3 | ML wrappers: whisper (incl. `transcribe_bytes`), NLLB, Marian, Indic2 + factory | `backend/ml_models/` |
| session 3 | Backend tests pass incl. **WS integration test** (mocked STT/MT) | `backend/tests/test_api.py` |
| session 3 | Frontend skeleton: Vite+TS+Zustand+Tailwind, pages, hooks, components | `frontend/` |
| session 3 | `npm run build` passes (tsc strict, 102 modules) | `frontend/` |
| session 3 | Docker: backend/frontend Dockerfiles, nginx proxy, compose (3 services) | `deployment/` |
| session 4 | **WS pipeline**: `/ws/{session_id}` accepts base64 webm → PyAV decode → whisper → NLLB → persist + push segment | `backend/api/websocket.py` |
| session 4 | Frontend `useLiveTranscript` (mic → WS → store captions) + presentational MicButton | `frontend/src/hooks/` |
| session 4 | **Live smoke test (real models)**: en→hi and hi→en via webm/opus decode | `scripts/smoke_pipeline.py` |
| session 5 | Removed `.venv` (user request — storage) | `.venv/` |
| session 5 | **Frontend rewritten in vanilla HTML/CSS/JS** (no React, no node_modules, no build step) | `frontend/` |
| session 5 | WS payload now includes persisted segment `id` (feedback needs it) | `backend/api/websocket.py` |
| session 6 | Env switched to conda base `/opt/anaconda3/bin/python` (3.13); added `datasets`, `asyncpg`, `pytest-asyncio` | — |
| session 6 | Backend re-verified on conda (4/4 tests), uvicorn `:8000` serves API + static frontend; real-model smoke passes | `backend/main.py` |
| session 7 | **Live STT rework**: WebAudio capture (no MediaRecorder — webm fragments broke PyAV); PCM→16k mono→WAV in JS; `decode_wav_to_float` on backend | `frontend/js/lecture.js`, `backend/api/websocket.py` |
| session 7 | Killed "Thank you" spam: removed feedback widget from live UI, RMS speech gate, whisper `vad_filter=True` | `frontend/js/lecture.js` |
| session 7 | `SttResult` gains `segments` + `words`; fixed generator-consumed-before-words bug; token strip | `backend/ml_models/whisper_service.py` |
| session 7 | `LiveTranscriber`: pause-anchored flush (900ms silence or 8s cap) → transcribe whole buffer → emit+reset; silence chunk no longer crashes (`AppError` caught); per-connection instance + persistence | `backend/api/websocket.py` |
| session 7 | En/hi STT upgraded `small` → `medium` (accurate sentences; ~4-5s/chunk CPU). WS E2E verified with real models, captions persisted | `backend/core/config.py` (env override) |
| session 8 | **Module 1 done (user-verified)**: recordings storage + translation target. Migration `1c40ff411020 → a3f0c91d2e11` adds `sessions.target_lang` + `recordings` table; `RecordingWriter` flushes WAV to `data/recordings/`; endpoints: `POST /sessions/{id}/end`, `GET /sessions/{id}/recordings`, `GET .../recording/download` (FileResponse); `/users` CRUD + preferred language; Stop-session + Download-recording buttons in UI (download waits for flush via `waitForRecording` poll) | `database/migrations`, `backend/services/recording_service.py`, `backend/api/users.py`, `sessions.py`, `websocket.py`, `frontend/js/lecture.js` |
| session 8 | **Module 2 = preprocess pipeline** (built, awaiting verification): load FLEURS test parquet+tar → drop missing/dup → text normalize (indic scripts kept) → spectral-gate noise reduction → peak normalize → VAD segmentation (silence>0.35s) → MFCC/delta/energy features → export. Output: `data/processed/clean/` (clean.parquet 4,692 rows: en 1708, hi 1446, te 1538; 4,692 wavs; 227 MB). Tar extraction cached to `data/cache/fleurs_audio` (was O(n²) 121s/50 → now 22s full set). `pytest backend/tests` (preprocess+classify) 14/14 | `scripts/preprocess_dataset.py`, `backend/tests/test_preprocess.py` |
| session 9 | **Module 3 = ML classification** (built, awaiting verification): LogReg/DT/RF/XGBoost on clean.parquet MFCC+energy features → XGBoost best acc 0.661/prec 0.662/rec 0.658/F1 0.659 (RF 0.655, DT 0.567, LogReg 0.555); LabelEncoder + best model persisted `data/processed/ml/` (XGBoost saved via native `save_model` json — joblib round-trip segfaults on XGB, fixed). 14 Module2/3 tests green; API tests need Postgres (stopped) | `scripts/classify_language.py`, `backend/tests/test_classify.py` |
| session 9 | **Module 4 = DL classification** (user-verified): `scripts/train_module4_deep.py` RNN(64)/LSTM(64)/Transformer(d64,4head,2layer) on padded MFCC sequences (160×13), MPS device: Transformer acc 0.9361/prec 0.9354/rec 0.9367/F1 0.9356 ≫ LSTM 0.8488 (F1 0.8458) ≫ RNN 0.3237 (underfits). Weights → `data/processed/dl/{arch}_best.pt`, results.json; 7 unit tests in `test_module4.py`. DL clearly beats Module-3 feature ML (XGB F1 0.659) | `scripts/train_module4_deep.py`, `backend/tests/test_module4.py` |
| session 9 | **Module 5 = NLP per language** (user-verified): script detection via char-range regex: en 1708 latin; hi 1446 devanagari; te 1538 telugu (clean split after `detect_script` run-count bug fixed — see below; earlier "60 latin hi / 103 latin te" were spurious). Stats: tokens en 43180/hi 41312/te 31857, unique 2872/2273/3138, ttr 0.067/0.055/0.098, mean_slen 25.3/28.3/20.7. Text language-ID (char n-gram 2-4 TF-IDF + LogReg) = acc/prec/rec/F1 **1.0** (sanity benchmark — scripts trivially separable). Output `data/processed/nlp/results.json`; 9 unit tests `test_nlp.py` | `scripts/nlp_analysis.py`, `backend/tests/test_nlp.py` |
| session 9 | **Ollama + Mistral installed locally** on user machine — to use at `http://localhost:11434` for Module 7 offline LLM path (no coding done) | — |
| session 9 | **Module 5 gaps closed** (user-verified): `scripts/nlp_embeddings_refine.py` — (1) multilingual sentence embeddings via cached NLLB-200 encoder (mean-pooled, L2-normalized; cross-lingual sim: hi↔te education 0.916, en↔hi 0.876, en↔te 0.858); (2) rule-based translation refinement (NFC normalizer, nbsp→space, spaced-Latin collapse, punctuation-spacing after/before, repeated-punct dedupe, number grouping, terminator add). Output `data/processed/nlp/embeddings_refine.json`. Bug fixed: `SPACED_LETTERS` was full-width `\w` (destroyed Devanagari/Telugu spaces) → Latin-only. 12 unit tests `test_nlp_gaps.py`; full suite 48/48 green | `scripts/nlp_embeddings_refine.py`, `backend/tests/test_nlp_gaps.py` |
| session 9 | **Module 6 = offline translation engine** (user-verified, no fixes needed): `scripts/translate_comparison.py` — `RuleTranslator` (dict lookup, punctuation guards, coverage/OOV metric) vs `AiTranslator` (NLLB via existing service) on IN22-Gen pairs (--limit 6/direction). Results: rule BLEU 0.0043/0.0025/0.0021/0.0021, coverage 0.27/0.15/0.26/0.05, 0ms; AI BLEU 0.192/0.172/0.024/0.170 at 2633/1032/1284/1159 ms (en→hi/hi→en/en→te/te→en). Demonstrates rule=free-but-crude, AI=real translation at ~1–2.6s. Output `data/processed/mt/translation_engine.json`; 8 tests `test_translation_engine.py`; suite 56/56 | `scripts/translate_comparison.py`, `backend/tests/test_translation_engine.py` |
| session 9 | **Module 7 = AI translation assistant** (user-verified): `scripts/ai_translation_assistant.py` — full pipeline `SttStage` (Whisper) → `MtStage` (NLLB) → `PromptTemplate` → `OllamaLlm` (Mistral @ localhost:11434, offline) → `JsonAssistant` (lenient JSON parse). LLM replies `{translation, explanation, study_note}`; prompt constrains translation to light polish of the NLLB draft and requires the target script; explanation + study_note in English. First run garbled Hindi (LLM rewrote Devanagari, mixed Latin `छotedaar`); **fixed via Option A+B**: `translation_displayed` = NLLB draft always, LLM polish kept as `refined_translation` (recorded, never displayed) → garbling gone. Verified en→hi Physics: clean Hindi draft displayed, coherent English notes, stt 1.4s/mt 2.1s/llm 10–13s. Output `data/processed/assistant/results.json`; 7 tests `test_assistant.py`; **full suite 63/63 green** | `scripts/ai_translation_assistant.py`, `backend/tests/test_assistant.py` |
| session 9 | **Module 8 = system evaluation** (user-verified): `scripts/evaluate_system.py` — (1) ML: consolidates Module 3+4 results.json → 7-model table sorted by F1 (Transformer 0.9356, LSTM 0.8458, XGB 0.6588, RF 0.6477, DT 0.5604, LogReg 0.5491, RNN 0.3194); (2) Translation: NLLB AI vs rule baseline on IN22-Gen (--limit 20/dir): BLEU-4 en→hi 0.1944 / hi→en 0.2057 / en→te 0.0182 / te→en 0.1288, exact-match 0%, script-consistency 100% (after fix), p95 1.5–2.5s; (3) Assistant: script fidelity 1.0, displayed==draft 1.0, explanation+study_note present, stt 1.4s/mt 2.1s/llm 12.7s; (4) Satisfaction: 47/47 positive in Postgres feedback, ratio 1.0. Output `data/processed/eval/evaluation.json`; 19 tests `test_evaluation.py`; **suite 82/82 green**. Quality gap: en→te NLLB is script-correct but BLEU 0.018 — IndicTrans2 is the fix candidate | `scripts/evaluate_system.py`, `backend/tests/test_evaluation.py` |
| session 9 | **`detect_script` regex bug fixed** (`nlp_analysis.py`): `SCRIPT_RANGES` used `+` quantifiers so `re.findall` counted character **runs** (ASCII spaces/punct each counted) instead of individual chars → Telugu/Devanagari texts misclassified as latin (e.g. Module 8 en→te script-consistency falsely 0%, Module 5 showed spurious "60 latin" hi / "103 latin" te). Dropped the `+`; re-verified: en 1708 latin, hi 1446 devanagari, te 1538 telugu (all clean), Module 8 script-consistency 100% everywhere. 82/82 tests green | `scripts/nlp_analysis.py` |
| session 10 | **Landing page** built (Tailwind CDN academic-modernist, design tokens copied from user's samples; hero CTAs: Get Started→lecture.html, Student View→dashboard.html; truthful benchmark copy, no external images, no login/register — roles by action). `app.html` (old shell) retired → redirect stub | `frontend/index.html`, `frontend/app.html` |
| session 10 | **`GET /api/v1/analytics`** endpoint: reads gitignored `data/processed/{ml,dl}/results.json`, `mt/translation_engine.json`, `eval/evaluation.json` via repo-root `Path(__file__)` resolution; reshapes → `ml.rows` (7 models F1-sorted + family tag), `translation` (BLEU AI-vs-rule, latency mean/p50/p95, script-consistency), assistant + satisfaction; missing/corrupt files skipped + `sources` list. 11 tests `test_analytics.py` | `backend/api/analytics.py`, `backend/tests/test_analytics.py` |
| session 11 | **Student page** `dashboard.html` + `js/student.js`: sidebar nav (Overview/Live/History/Analytics) with mobile overlay, stats cards (live/total/segments/satisfaction/best model from analytics API), Live = join flow + 1.5s segment polling + thumb feedback (`rating` bool), History expandable per-session captions, Analytics = 4 Plotly charts (F1 & accuracy grouped bars, BLEU AI vs rule, latency p50/p95, satisfaction donut), AI assistant chat (canned demo) | `frontend/dashboard.html`, `frontend/js/student.js` |
| session 11 | **Lecturer studio** `lecture.html` + `js/lecturer.js`: same design system; reuses battle-tested `lecture.js` (WebAudio PCM capture + WS); live stats cards (captions/duration/avg latency, 3s poll); AI assistant sidebar (canned); Dockerfile copies new pages | `frontend/lecture.html`, `frontend/js/lecturer.js` |
| session 12 | **Join codes**: `sessions.join_code` (6 chars, alphabet `023456789ABCDEFGHJKMNPQRSTUVWXYZ`, no 1/I/L/O, `uuid.int & (1<<30)-1` → base-32), deterministic + backfilled in migration `b7e3f9a1c2d4` (unique constraint); service `make_join_code`/`join_session`; `POST /api/v1/sessions/join` (uppercase-insensitive, only non-ended). `POST /sessions/{id}/start` resets `started_at=now` so lecturer duration counts only from Start click; stopped at ended_at | `database/migrations/versions/b7e3f9a1c2d4...`, `backend/services/session_service.py`, `backend/api/sessions.py` |
| session 12 | **Lecturer flow restructured**: Generate code → form (src/translate-to/subject) → Create session (banner + mic + captions) → Start session (opens WS) → Stop session (freeze timer + recording). **Banner bug fixed**: it polled DB and showed a leftover test session's code on page load — now only via `session-created` CustomEvent; stats locked to that `sessionCode` | `frontend/js/lecture.js`, `frontend/js/lecturer.js` |
| session 12 | **Student live-caption fix**: joining sessions with historical test captions replayed them as "live" (the "hello world" spam) and blocked real new ones being noticed → on join, pre-mark all existing segment IDs into `seenSegmentIds` so only post-join captions stream; empty state now explains "waiting for lecturer". Verified live: stale segment suppressed, new segment streams | `frontend/js/student.js` |
| session 12 | **Blank-page root cause found**: an edit had dropped `refreshStats()`'s closing `{` → `SyntaxError: Unexpected end of input` killed the whole module (only in browser; `node --check` exits were being masked by piping into `head`). Fixed + added discipline: per-file `node --check` (no pipes), brace-balance tokenizer, headless-Chrome console-error check. Also: server launch must use `Popen(start_new_session=True)` (nohup+disown gets SIGTERM'd by the tool shell); ended 19 zero-caption junk sessions via API | `frontend/js/lecturer.js` |
| session 13 | **IndicTrans2 engine fixed + wired (user E2E-verified)**: root causes — (a) NaN sinusoidal buffers under transformers 5.x meta-device init (fixed: regenerate via `make_weights` after load), (b) `generate()` incompatible with 5.x caches (fixed: manual greedy/beam decode with legacy tuple cache), (c) en-indic-1B is En→Indic-only (Indic↔Indic needs the separate `indictrans2-indic-indic-1B`, downloaded 4.8GB; X→en intentionally NOT downloaded → NLLB fallback), (d) model script-unifies through Devanagari (TGT vocab is ~75k Deva words, ~64 Te pieces) → official pre/post transliteration via indic-nlp-library. All 6 directions verified vs IN22-Gen; `translation_backend` default now `indic2`; 98/98 tests green | `backend/ml_models/indic2_service.py` (rewritten), `__init__.py` (X→en fallback), `backend/core/config.py` |

## Spike Results (Apple M4, 10 cores, 16GB — Aug 2026)

### STT — faster-whisper (CPU int8)
| Model | Hindi | English | Telugu |
| --- | --- | --- | --- |
| small | RTF 0.34–0.55, good quality | RTF 0.40–0.50, good | **FAILS**: wrong script (Devanagari/Sinhala garbage), RTF 8–12x on some clips |
| medium | — | — | **FAILS**: RTF 17–23x, still wrong script |
| large-v3 | — | — | **WORKS**: correct Telugu script, RTF 1.9–2.3 (~8–10s per 4–5s clip) |

### MT (MPS)
| Model | Latency | Notes |
| --- | --- | --- |
| MarianMT opus-mt | en→hi 2.5s, hi→en 0.36s | hi pair only; quality good; no Telugu |
| **NLLB-200-distilled-600M** | en→hi 1.7s, hi→en 0.43s, en→te 0.61s, te→en 0.34s | **best working option: all 3 langs both directions, correct output** |
| IndicTrans2 en-indic-1B | en→X 1.2–1.7s (MPS) | **En→Indic only** (garbage on non-English sources); wired as default MT for en→hi/en→te |
| IndicTrans2 indic-indic-1B | hi↔te 1.5–3.8s (MPS) | **Indic↔Indic model** (downloaded session 13); wired for hi→te/te→hi; script-unified via Devanagari → needs transliteration pre/post |
| IndicTrans2 indic-en-1B | not downloaded | would cover X→en; hi→en/te→en currently use NLLB fallback |

### Verdict / Recommended Architecture
- **en/hi live captions**: whisper `small` (CPU int8) + IndicTrans2 (MPS) → ~3–4s per sentence, meets 3–5s budget.
- **Telugu live captions**: whisper `large-v3` (CPU int8, ~2x RTF) + IndicTrans2 → near-real-time (~8–10s per sentence, sentence-buffered). Documented tradeoff vs PRD §7 target.
- **IndicTrans2** (access granted): now the default MT backend for en→X and Indic↔Indic (quality fix for the NLLB en→te BLEU-0.018 gap). hi→en / te→en fall back to NLLB until `indic-en-1B` is downloaded.
- Download size ~11GB for models (small 460MB + large-v3 3GB + NLLB 2.4GB + MarianMT 2×300MB + IndicTrans2 en-indic 4.5GB + indic-indic 4.8GB).

## Next Steps (Module order)

- [x] (DONE session 14) Wire the UI AI assistants to the real Ollama pipeline (Module 7) — `POST /api/v1/assistant` + dashboard/lecture chat wired, headless-Chrome E2E-verified
- [x] (DONE session 13) IndicTrans2 quality fix — en→te/hi→te/te→hi now via IndicTrans2 (was NLLB BLEU 0.018 / garbage); optional upgrade left: download `indictrans2-indic-en-1B` (~4.8GB) for IndicTrans2-quality X→en (currently NLLB fallback)
- [ ] Docker daemon up (`open -a Docker`) → `docker compose up` in `deployment/`
- [ ] (optional) Clean the ~110 leftover live test sessions from DB (kept because they have captions/history)
- [ ] Browser mic E2E (http://localhost:8000) — lecturer speaks, student streams live (manual, both windows)

## Notes / Deviations from Plan

- Common Voice 12_0 removed from HF; using ungated mirror `fixie-ai/common_voice_17_0` (v17).
- CV Telugu is tiny (62 train rows); FLEURS `te_in` audio is the effective Telugu speech source.
- IN22-Gen schema changed: single `default` config (`test` split, per-language columns).
- BPCC schema changed: per-target-language tsv files (`src_lang`, `tgt_lang`, `src`, `tgt`).
- FLEURS `load_dataset` hangs locally; direct hub download used instead.
- IN22-Gen + BPCC are gated — require logged-in HF token (`Uday533`) + accepted terms.
- Docker daemon not running locally yet; compose config written but untested (needs review of Alembic/nginx paths before `up`).
- WebSocket URL must be same-origin (`${location.host}`) so nginx proxy in compose works.
- Frontend is Tailwind-CDN — needs internet on first load (same caveat as the design samples user provided). Plotly loaded from CDN too.
- Browser may cache old HTML/JS after redesigns — testers must hard-refresh (Cmd+Shift+R).
- Student caption language choice is cosmetic today (segments are stored with the lecturer's target lang); noted as future per-student re-translation.
- `pkill -f "uvicorn backend.main"` stops the API; `brew services stop postgresql@18` stops Postgres.

## Latest Change Log

- `frontend/css/styles.css` (session 14c): `.mic-btn` flex-centering fix — mic icon was pinned to circle's left edge (Tailwind preflight `svg{display:block}` beats `text-align:center`); now `display:flex; align-items:center; justify-content:center`.
- `backend/api/sessions.py` (session 14b): `GET /{session_id}/transcript` — plain-text transcript download (timestamps + source + translation), 404 on unknown session.
- `frontend/dashboard.html` + `js/student.js` + `js/api.js` (session 14b): end-of-session modal (Lecture ended → download transcript / close), live chips flip to "Ended", polling stops; `getSession()` / `transcriptUrl()` helpers.
- `backend/tests/test_api.py` (session 14b): `test_transcript_download` — content headers, body, 404 → suite **102/102**.
- `backend/api/assistant.py` (**new, session 14**): `POST /api/v1/assistant` — question + role + optional session_id/context; context = frontend's last 6 captions or last 6 DB segments; prompt via System/User template; `OllamaLlm.generate` in `asyncio.to_thread` (mistral); exchange logged to `data/processed/assistant/live.jsonl`; 503 `ollama_offline` when down.
- `backend/schemas/common.py` (session 14): `AssistantContextItem` / `AssistantRequest` / `AssistantResponse`; `backend/main.py` registers the assistant router.
- `frontend/js/student.js` + `frontend/js/lecturer.js` (session 14): async `assistantBot()` → API first, template fallback on error; "…" pending bubble replaced by live answer.
- `backend/tests/test_assistant.py` (session 14): +3 endpoint tests (sync-fake generate via to_thread, context in prompt, 503 path) → suite **101/101 green**.
- `scripts/download_datasets.py`: FLEURS direct-download path; BPCC chunked direct-download path; IN22 pair extracts via pandas; fixed `large_string` schema cast; self-clean HF cache.
- `frontend/`: **now vanilla HTML/CSS/JS** (was React+Vite); static files, no build step. Old React scaffold (package.json, vite, tsconfig, src/) deleted with `node_modules/`.
- `backend/`: `api/websocket.py` (webm → PyAV decode → whisper → NLLB → persist/push, sends persisted `id`); whisper `transcribe_bytes` + `model` attr; fixed `NLLB._get` + resampler frame concat bugs.
- `scripts/smoke_pipeline.py`: real-model end-to-end harness (mp3/wav → webm/opus → STT → MT).
- `deployment/`: backend/frontend Dockerfiles, nginx.conf, docker-compose.yml (db+backend+frontend), alembic.ini paths made `%(here)s`-relative for container runs.
- `database/alembic.ini`: `script_location` + `prepend_sys_path` now `%(here)s`-relative (verified `alembic current` still works).
- `scripts/preprocess_dataset.py`: module-gated logger, cached tar extraction (`data/cache/fleurs_audio`) so gzip tars are decompressed once (was O(n²) 121s/50 rows → ~22s full set), `--limit` flag for smoke runs, `read_mem_wav` resampler; FLEURS audio is already 16k mono float WAV.
- `data/processed/clean/`: Module 2 deliverable — `clean.parquet` (features) + per-segment WAVs; counts by lang above.
- `database/migrations/versions/1c40ff411020...` → `a3f0c91d2e11_module1_recording.py`: module 1 migration (sessions.target_lang + recordings table), applied.
- `backend/services/recording_service.py` + `user_service.py`, `backend/api/users.py` (`/users` CRUD + preferred language), `sessions.py` (`POST /sessions/{id}/end`, `GET .../recordings`, `GET .../recording/download`), `websocket.py` (RecordingWriter per connection, flush+commit on disconnect), `models/entities.py` + `schemas/common.py` (Recording, User, target_lang).
- `frontend/js/lecture.js` + `api.js` + `css/styles.css`: Stop session button (`btn-danger`), Download recording button that waits for the WAV flush via `waitForRecording` poll, record disabled while session active.
- `backend/tests/test_preprocess.py`: unit tests for text clean, noise gate, normalization, VAD segmentation (15/15 total).
- `scripts/train_module4_deep.py`: Module 4 DL — RNN/LSTM/Transformer classifiers on padded MFCC sequences w/ MPS device pick, stratified 80/20 split, per-epoch val metric on test split (noted methodology: early-stopping reads test), saves `data/processed/dl/{arch}_best.pt` + results.json; `--epochs/--limit/--mode/--batch/--seed`.
- `scripts/nlp_analysis.py`: Module 5 NLP — script detection (Latin/Devanagari/Telugu regex), per-lang stopword lists (en/hi/te), tokenizer, type-token ratio + sentence stats + top-words, text-lang-ID (char n-gram TF-IDF + LogisticRegression); `data/processed/nlp/results.json`.
- `scripts/nlp_embeddings_refine.py`: Module 5 gap filler — (1) multilingual sentence embeddings via NLLB-200 encoder (mean-pooled, L2-norm) + cosine matrix; (2) rule-based translation refinement (pure function). Writes `data/processed/nlp/embeddings_refine.json`.
- `scripts/translate_comparison.py`: Module 6 — `RuleTranslator` (dictionary + token guards) vs `AiTranslator` (NLLB) compared on IN22-Gen pairs via BLEU-4/cov/latency; `--limit` flag; writes `data/processed/mt/translation_engine.json`. Verified: rule BLEU≈0 instant, AI BLEU 0.17–0.19 (hi-eng), 1–2.6s/sentence.
- `scripts/ai_translation_assistant.py`: Module 7 — pipeline SttStage→MtStage→PromptTemplate→OllamaLlm(Mistral, localhost:11434)→JsonAssistant; `--src/--tgt/--subject/--audio/--dry-run`; prompt A+B: LLM polishes only + English explanation/study note, `translation_displayed` always = NLLB draft (Option A, LLM polish as `refined_translation`); writes `data/processed/assistant/results.json`.
- `scripts/evaluate_system.py`: Module 8 — ML consolidation (7 models, F1-ranked), translation eval (NLLB-vs-rule BLEU-4, exact-match, script-consistency, latency mean/p50/p95), assistant fidelity/explainability/latency, Postgres feedback satisfaction; `--limit`/`--no-db` flags; writes `data/processed/eval/evaluation.json`.
- `scripts/nlp_analysis.py`: fixed `SCRIPT_RANGES` regexes — removed `+` quantifiers so script detection counts characters instead of runs (was misclassifying Devanagari/Telugu as latin).
- `backend/tests/`: 102 unit tests total — `test_api.py` 8 (join flows, feedback, transcript), `test_analytics.py` 11, `test_preprocess.py` 11, `test_classify.py` 3, `test_module4.py` 7, `test_nlp.py` 11, `test_nlp_gaps.py` 12, `test_translation_engine.py` 8, `test_assistant.py` 10 (7 script-class + 3 endpoint), `test_evaluation.py` 19 — **full suite 102/102 green**. Note: XGBoost artifacts are `.json` (native save — joblib round-trip on XGB segfaults in pytest).
- `frontend/`: vanilla HTML/CSS/JS pages — `index.html` (landing), `dashboard.html` (student, 4 tabs + Plotly), `lecture.html` (lecturer studio + join-code banner + assistant), `app.html` (redirect stub); `js/student.js`, `js/lecturer.js`, `js/lecture.js` (4-step flow), `js/api.js` (joinSession/startSession added).
- `backend/`: events as above — `analytics.py`, sessions `join`/`start` endpoints, `join_code` column (migration `b7e3f9a1c2d4`).
- `backend/ml_models/indic2_service.py` (**rewritten, session 13**): dual-checkpoint routing (en-indic-1B for en→X, indic-indic-1B for Indic↔Indic), `supports()` rejects X→en, `_fix_positional_buffers` (NaN sinusoidal buffers under transformers 5.x meta-device init → `make_weights`), manual `_decode` greedy/beam loop (5.x `generate()` cache incompatible), `_preprocess`/`_postprocess` official-style normalization + hi↔te transliteration via indic-nlp-library. Verified en→hi/en→te/hi→te/te→hi vs IN22-Gen (1.2–3.8s/sentence MPS).
- `backend/core/config.py`: `translation_backend` default `"nllb"` → `"indic2"` (session 13; server must restart to pick it up).
- `backend/ml_models/__init__.py`: X→en routed to `nllb(fallback)` when backend is indic2 (session 13).
- `~/.cache/huggingface/hub`: `ai4bharat/indictrans2-indic-indic-1B` downloaded (4.8GB, session 13). `indic-en-1B` deliberately NOT downloaded.
- Diagnostic scripts left over from the debugging session (stale, reference old `_load`/`_MODEL_ID`): `scripts/indic2_{nan_check,layer_nan,where_nan,forward_check,probe,inspect,debug}.py` — not part of tests; deletable.
- Test/browser gotchas (see ## Currently Working): server must be launched with `Popen(start_new_session=True)`; `node --check` exit codes masked by pipes; Postgres must be up for `test_api.py`.

Last updated: 2026-08-17 (session 14)
