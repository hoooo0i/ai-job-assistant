from types import SimpleNamespace

from src.realtime_transcription import (
    DEFAULT_REALTIME_MAX_MINUTES,
    RealtimeTranscriptionSession,
    looks_like_interview_question,
    realtime_event_message,
    realtime_max_seconds,
)


def test_detects_chinese_and_english_interview_questions() -> None:
    assert looks_like_interview_question("请介绍一个你推动项目落地的例子。")
    assert looks_like_interview_question("How did you resolve that conflict?")
    assert not looks_like_interview_question("我负责整理数据并完成复盘。")
    assert not looks_like_interview_question("好的")


def test_maps_realtime_delta_and_completed_events() -> None:
    delta = SimpleNamespace(
        type="conversation.item.input_audio_transcription.delta",
        item_id="item_1",
        delta="请介绍",
    )
    completed = {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "item_1",
        "transcript": "请介绍一下你的项目。",
    }

    assert realtime_event_message(delta) == {
        "type": "delta",
        "item_id": "item_1",
        "text": "请介绍",
    }
    assert realtime_event_message(completed) == {
        "type": "completed",
        "item_id": "item_1",
        "text": "请介绍一下你的项目。",
    }
    assert realtime_event_message({"type": "session.updated"}) is None


def test_session_drops_audio_until_it_can_start_and_stop(monkeypatch) -> None:
    session = RealtimeTranscriptionSession(max_seconds=1)
    starts: list[bool] = []

    monkeypatch.setattr(session, "start", lambda: starts.append(True) or True)
    session.append_audio(b"pcm")
    assert starts == [True]

    session.stop()
    assert session._stop_requested.is_set()


def test_realtime_duration_defaults_to_two_hours_and_is_bounded(monkeypatch) -> None:
    monkeypatch.delenv("INTERVIEW_REALTIME_MAX_MINUTES", raising=False)
    assert realtime_max_seconds() == DEFAULT_REALTIME_MAX_MINUTES * 60

    monkeypatch.setenv("INTERVIEW_REALTIME_MAX_MINUTES", "999")
    assert realtime_max_seconds() == 240 * 60

    monkeypatch.setenv("INTERVIEW_REALTIME_MAX_MINUTES", "invalid")
    assert realtime_max_seconds() == DEFAULT_REALTIME_MAX_MINUTES * 60
