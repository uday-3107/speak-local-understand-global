"""Module 7: AI Translation Assistant — the core pipeline.

Pipeline (per official spec):
  Speech Input -> Whisper STT -> NLLB translation -> Prompt Template
               -> LLM (Ollama / Mistral, offline) -> Translated Response
               -> Language Assistance (explanation + study note)

Output per utterance: transcript, NLLB draft translation (always the
displayed translation), optional LLM polish (recorded, not displayed), and
language assistance (explanation + study note in English — Mistral produces
far better English than Devanagari). The LLM is constrained to light polish
only (no rewrites / invented words), so NLLB quality never regresses.
Results saved to data/processed/assistant/results.json

Prerequisites (user side):
  - Ollama running:  `ollama serve`  (model: mistral, already pulled)
  - Test clips: data/processed/test_audio/*.{wav,mp3}

Usage (from repo root):
    /opt/anaconda3/bin/python -m scripts.ai_translation_assistant \
        --audio data/processed/test_audio/en_10004088536354799741.wav \
        --src en --tgt hi --subject "Physics lecture"
    # pipeline without the LLM (if Ollama is off):
    /opt/anaconda3/bin/python -m scripts.ai_translation_assistant \
        --audio ... --src en --tgt hi --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import time

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
OUT_DIR = "data/processed/assistant"

SYSTEM_PROMPT = (
    "You are an AI language assistant for a multilingual classroom. "
    "You receive a lecture sentence in {src} and its machine translation in {tgt}. "
    "Reply ONLY with a JSON object containing exactly these keys: "
    '"translation", "explanation", "study_note". '
    '"translation": only light polish of the provided {tgt} machine translation — '
    'the translation must be written in {tgt} ({tgt} script, not {src}); '
    'fix grammar/punctuation and word order; do NOT rewrite, do NOT add or invent '
    'words, do NOT change the meaning. If it is already correct, copy it verbatim. '
    '"explanation": a short ONE-sentence summary of the key idea, in English. '
    '"study_note": one short memorable takeaway for a student, in English. '
    "Never invent facts. Return only the JSON, nothing else."
)


def log(msg: str) -> None:
    print(f"[module7] {msg}")


class SttStage:
    """Speech Input -> Speech Recognition (Whisper, cached)."""

    name = "whisper"

    def run(self, audio_path: str, language: str | None) -> tuple[str, int]:
        from backend.ml_models.whisper_service import WhisperService

        res = WhisperService().transcribe(audio_path, language=language)
        return res.text, res.latency_ms


class MtStage:
    """Translation Model (NLLB)."""

    name = "nllb"

    def run(self, text: str, src: str, tgt: str) -> tuple[str, int]:
        from backend.ml_models.nllb_service import NllbService

        return NllbService().translate(text, src, tgt)


class PromptTemplate:
    """Prompt Template builder (A+B: LLM polishes only + explains in English)."""

    def build(self, transcript: str, draft: str, src: str, tgt: str,
              subject: str = "") -> str:
        system = SYSTEM_PROMPT.format(src=src, tgt=tgt)
        user = (
            f"Lecture subject: {subject or 'general'}\n"
            f"Original {src} sentence:\n{transcript}\n"
            f"Machine translation ({tgt}):\n{draft}\n"
            f"Polish the translation if needed and return the JSON answer."
        )
        return f"### System\n{system}\n\n### User\n{user}\n\n### Assistant"


class OllamaLlm:
    """LLM stage — Ollama (Mistral), fully offline."""

    name = "ollama_mistral"

    def __init__(self, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL,
                 timeout: int = 120):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> tuple[str, int]:
        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "")
        return text.strip(), int((time.perf_counter() - t0) * 1000)

    def ping(self) -> bool:
        try:
            return requests.get(f"{self.url}/api/tags", timeout=3).status_code == 200
        except requests.RequestException:
            return False


class JsonAssistant:
    """Parses the LLM answer (lenient fallback: whole text as translation)."""

    @staticmethod
    def parse(raw: str) -> dict:
        import re

        try:
            start = raw.find("{")
            end = raw.rfind("}")
            return json.loads(raw[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return {"translation": raw, "explanation": "", "study_note": ""}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="path to lecture clip")
    parser.add_argument("--src", required=True, choices=["en", "hi", "te"])
    parser.add_argument("--tgt", required=True, choices=["en", "hi", "te"])
    parser.add_argument("--subject", default="general lecture")
    parser.add_argument("--dry-run", action="store_true",
                        help="skip the LLM stage (Ollama off)")
    args = parser.parse_args()

    if args.src == args.tgt:
        raise SystemExit("--src and --tgt must differ")
    if not os.path.exists(args.audio):
        raise SystemExit(f"audio not found: {args.audio}")

    os.makedirs(OUT_DIR, exist_ok=True)
    llm = OllamaLlm()
    if not args.dry_run and not llm.ping():
        raise SystemExit(
            "Ollama is not running. Start it with `ollama serve` "
            "or re-run with --dry-run to skip the LLM stage.")

    stt, mt, tpl = SttStage(), MtStage(), PromptTemplate()

    log(f"STT (whisper)…")
    transcript, stt_ms = stt.run(args.audio, args.src)
    log(f"  -> {transcript[:80]}")
    log(f"MT (NLLB)…")
    draft, mt_ms = mt.run(transcript, args.src, args.tgt)
    log(f"  -> {draft[:80]}")

    result = {
        "audio": args.audio,
        "src": args.src, "tgt": args.tgt, "subject": args.subject,
        "transcript": transcript,
        "draft_translation": draft,
        "latency_ms": {"stt": stt_ms, "mt": mt_ms},
    }

    if args.dry_run:
        result["llm"] = {"skipped": True, "reason": "dry-run"}
        log("LLM skipped (dry-run)")
    else:
        prompt = tpl.build(transcript, draft, args.src, args.tgt, args.subject)
        log(f"LLM ({llm.name})…")
        raw, llm_ms = llm.generate(prompt)
        answer = JsonAssistant.parse(raw)
        refined = (answer.get("translation") or "").strip()
        result["translation_displayed"] = draft
        result["refined_translation"] = refined
        result["explanation"] = answer.get("explanation", "")
        result["study_note"] = answer.get("study_note", "")
        result["latency_ms"]["llm"] = llm_ms
        log(f"  refined -> {refined[:80]}")
        log(f"  note    -> {answer.get('study_note', '')[:80]}")

    out = os.path.join(OUT_DIR, "results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n=== Module 7 output ===")
    for k, v in result.items():
        print(f"  {k}: {v if not isinstance(v, dict) else json.dumps(v)}")
    log("saved " + out)


if __name__ == "__main__":
    main()