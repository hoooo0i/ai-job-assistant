from types import SimpleNamespace

import pytest

from src.audio_transcription import (
    AudioTranscriptionError,
    MAX_AUDIO_BYTES,
    transcribe_interview_audio,
)


class FakeTranscriptions:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)


class FakeClient:
    def __init__(self, text: str) -> None:
        self.audio = SimpleNamespace(transcriptions=FakeTranscriptions(text))


def test_transcribes_audio_without_writing_a_file(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
    client = FakeClient("请介绍一个你负责的数据项目。")

    result = transcribe_interview_audio(
        b"synthetic wav bytes",
        filename="面试 问题.wav",
        client=client,
    )

    call = client.audio.transcriptions.calls[0]
    assert result == "请介绍一个你负责的数据项目。"
    assert call["model"] == "gpt-4o-mini-transcribe"
    assert call["file"][0] == "_____.wav"
    assert call["file"][1] == b"synthetic wav bytes"


def test_rejects_empty_or_oversized_audio() -> None:
    with pytest.raises(AudioTranscriptionError, match="没有检测"):
        transcribe_interview_audio(b"")
    with pytest.raises(AudioTranscriptionError, match="10 MB"):
        transcribe_interview_audio(b"x" * (MAX_AUDIO_BYTES + 1))
