from __future__ import annotations

import os
import re
from typing import Any

from openai import OpenAIError

from src.ai_provider import create_openai_client
from src.privacy import redact_sensitive_info


MAX_AUDIO_BYTES = 10 * 1024 * 1024
DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"


class AudioTranscriptionError(RuntimeError):
    """A transcription failure that is safe to show in the UI."""


def transcribe_interview_audio(
    audio_data: bytes,
    *,
    filename: str = "interview-question.wav",
    content_type: str = "audio/wav",
    client: Any | None = None,
) -> str:
    if not audio_data:
        raise AudioTranscriptionError("没有检测到可转写的录音。")
    if len(audio_data) > MAX_AUDIO_BYTES:
        raise AudioTranscriptionError("单段录音不能超过 10 MB。")
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename) or "question.wav"
    model = (
        os.getenv("OPENAI_TRANSCRIPTION_MODEL", "").strip()
        or DEFAULT_TRANSCRIPTION_MODEL
    )
    active_client = client or create_openai_client()
    try:
        response = active_client.audio.transcriptions.create(
            model=model,
            file=(safe_name, audio_data, content_type or "audio/wav"),
        )
    except OpenAIError as exc:
        raise AudioTranscriptionError(
            "语音转写失败，请检查 API 密钥、模型权限或网络后重试。"
        ) from exc
    text = response if isinstance(response, str) else getattr(response, "text", "")
    cleaned = redact_sensitive_info(str(text or "")).strip()
    if len("".join(cleaned.split())) < 2:
        raise AudioTranscriptionError(
            "没有识别到清晰的面试问题，请靠近麦克风后重试。"
        )
    return cleaned
