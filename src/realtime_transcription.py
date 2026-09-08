from __future__ import annotations

import asyncio
import base64
import os
import queue
import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
from av import AudioFrame, AudioResampler
from openai import AsyncOpenAI, OpenAIError


DEFAULT_REALTIME_TRANSCRIPTION_MODEL = "gpt-live-transcribe"
DEFAULT_REALTIME_MAX_MINUTES = 120
MAX_REALTIME_MINUTES = 240
_STOP_AUDIO = object()
_COMMIT_AUDIO = object()


class RealtimeTranscriptionError(RuntimeError):
    """A real-time transcription failure that is safe to show in the UI."""


def realtime_max_seconds() -> int:
    """Read a bounded session duration without allowing an endless listener."""
    configured = os.getenv("INTERVIEW_REALTIME_MAX_MINUTES", "").strip()
    try:
        minutes = int(configured) if configured else DEFAULT_REALTIME_MAX_MINUTES
    except ValueError:
        minutes = DEFAULT_REALTIME_MAX_MINUTES
    return min(max(minutes, 10), MAX_REALTIME_MINUTES) * 60


def looks_like_interview_question(text: str) -> bool:
    """Return whether a completed speech turn looks like an interview question."""
    compact = "".join(str(text or "").split())
    if len(compact) < 4:
        return False
    if "?" in compact or "？" in compact:
        return True
    question_markers = (
        "请介绍",
        "请谈谈",
        "说说",
        "讲一下",
        "举一个",
        "举例",
        "为什么",
        "怎么",
        "如何",
        "是否",
        "有没有",
        "能否",
        "什么",
        "哪一个",
        "哪些",
        "你会",
        "你认为",
        "tell me",
        "describe",
        "why ",
        "how ",
        "what ",
        "which ",
        "could you",
        "can you",
    )
    lowered = compact.casefold()
    return any(marker in lowered for marker in question_markers)


def realtime_event_message(event: Any) -> dict[str, str] | None:
    """Convert an SDK Realtime event into a small, UI-safe message."""
    if isinstance(event, Mapping):
        event_type = str(event.get("type", ""))
        get_value = event.get
    else:
        event_type = str(getattr(event, "type", ""))
        get_value = lambda name, default="": getattr(event, name, default)

    if event_type == "conversation.item.input_audio_transcription.delta":
        delta = str(get_value("delta", "") or "")
        if delta:
            return {
                "type": "delta",
                "item_id": str(get_value("item_id", "") or "current"),
                "text": delta,
            }
    if event_type == "conversation.item.input_audio_transcription.completed":
        transcript = str(get_value("transcript", "") or "").strip()
        if transcript:
            return {
                "type": "completed",
                "item_id": str(get_value("item_id", "") or "current"),
                "text": transcript,
            }
    if event_type == "error":
        return {
            "type": "error",
            "text": "实时转写服务返回错误，请停止后重新开始。",
        }
    return None


class RealtimeTranscriptionSession:
    """Bridge browser PCM audio to one OpenAI Realtime transcription session."""

    def __init__(self, *, max_seconds: int | None = None) -> None:
        self.max_seconds = max_seconds or realtime_max_seconds()
        self._audio_queue: queue.Queue[bytes | object] = queue.Queue(maxsize=200)
        self._event_queue: queue.Queue[dict[str, str]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._stop_requested = threading.Event()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> bool:
        """Start the background session once. Returns True for a new session."""
        with self._state_lock:
            if self.running:
                return False
            self._stop_requested.clear()
            self._audio_queue = queue.Queue(maxsize=200)
            self._thread = threading.Thread(
                target=self._thread_main,
                name="interview-realtime-transcription",
                daemon=True,
            )
            self._thread.start()
            return True

    def append_audio(self, pcm_data: bytes) -> None:
        if not pcm_data or self._stop_requested.is_set():
            return
        if not self.running:
            self.start()
        try:
            self._audio_queue.put_nowait(pcm_data)
        except queue.Full:
            # Keep latency bounded: discard the oldest chunk instead of building a
            # delayed recording backlog.
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(pcm_data)
            except (queue.Empty, queue.Full):
                pass

    def stop(self) -> None:
        self._stop_requested.set()
        try:
            self._audio_queue.put_nowait(_STOP_AUDIO)
        except queue.Full:
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(_STOP_AUDIO)
            except (queue.Empty, queue.Full):
                pass

    def commit_turn(self) -> None:
        """Commit a locally detected speech turn without retaining its audio."""
        if not self.running or self._stop_requested.is_set():
            return
        try:
            self._audio_queue.put_nowait(_COMMIT_AUDIO)
        except queue.Full:
            pass

    def drain_events(self) -> list[dict[str, str]]:
        events: list[dict[str, str]] = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                return events

    def _emit(self, event_type: str, text: str = "", item_id: str = "") -> None:
        message = {"type": event_type}
        if text:
            message["text"] = text
        if item_id:
            message["item_id"] = item_id
        self._event_queue.put(message)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception:
            self._emit(
                "error",
                "实时转写连接异常，请检查 API 配置和网络后重新开始。",
            )
        finally:
            self._emit("stopped")

    async def _run(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            self._emit("error", "尚未配置 OPENAI_API_KEY，无法开始实时转写。")
            return
        transcription_model = (
            os.getenv("OPENAI_REALTIME_TRANSCRIPTION_MODEL", "").strip()
            or DEFAULT_REALTIME_TRANSCRIPTION_MODEL
        )
        started_at = time.monotonic()
        client = AsyncOpenAI(api_key=api_key)
        try:
            # Dedicated transcription sessions use intent=transcription and must
            # not pass a conversational Realtime model in the connection query.
            async with client.realtime.connect(
                extra_query={"intent": "transcription"}
            ) as connection:
                await connection.session.update(
                    session={
                        "type": "transcription",
                        "audio": {
                            "input": {
                                "format": {"type": "audio/pcm", "rate": 24000},
                                "noise_reduction": {"type": "far_field"},
                                "transcription": {
                                    "model": transcription_model,
                                    "languages": ["zh-cn", "en"],
                                    "delay": "low",
                                    "keywords": [
                                        "岗位职责",
                                        "项目经历",
                                        "STAR",
                                        "KPI",
                                        "产品经理",
                                    ],
                                },
                                # gpt-live-transcribe streams deltas continuously,
                                # while completed turns are committed by the local
                                # VAD in RealtimeAudioProcessor.
                                "turn_detection": None,
                            }
                        },
                    }
                )
                self._emit("connected")
                sender = asyncio.create_task(
                    self._send_audio(connection, started_at),
                    name="realtime-audio-sender",
                )
                try:
                    async for event in connection:
                        message = realtime_event_message(event)
                        if message:
                            self._event_queue.put(message)
                        if sender.done():
                            break
                finally:
                    if not sender.done():
                        sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
        except OpenAIError:
            self._emit(
                "error",
                "实时转写失败，请检查 API 密钥、模型权限或网络后重试。",
            )
        finally:
            await client.close()

    async def _send_audio(self, connection: Any, started_at: float) -> None:
        while not self._stop_requested.is_set():
            if time.monotonic() - started_at >= self.max_seconds:
                minutes = self.max_seconds // 60
                self._emit(
                    "limit",
                    f"单次实时辅助已达 {minutes} 分钟，请重新开始。",
                )
                self._stop_requested.set()
                break
            chunk = await asyncio.to_thread(self._audio_queue.get)
            if chunk is _STOP_AUDIO:
                break
            if chunk is _COMMIT_AUDIO:
                await connection.input_audio_buffer.commit()
                continue
            if not isinstance(chunk, bytes):
                continue
            encoded = base64.b64encode(chunk).decode("ascii")
            await connection.input_audio_buffer.append(audio=encoded)
        await connection.close()


class RealtimeAudioProcessor:
    """Resample WebRTC frames to mono 24 kHz PCM16 without retaining audio."""

    def __init__(self, session: RealtimeTranscriptionSession) -> None:
        self.session = session
        self.resampler = AudioResampler(format="s16", layout="mono", rate=24000)
        self._speech_active = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0

    def __call__(self, frame: AudioFrame) -> AudioFrame:
        frames = self.resampler.resample(frame)
        if frames is None:
            return frame
        if isinstance(frames, AudioFrame):
            frames = [frames]
        for resampled in frames:
            samples = resampled.to_ndarray()
            self.session.append_audio(samples.tobytes())
            duration_ms = max(resampled.samples / 24000 * 1000, 0.0)
            float_samples = samples.astype(np.float32, copy=False)
            rms = float(np.sqrt(np.mean(np.square(float_samples)))) if samples.size else 0.0
            if rms >= 500:
                self._speech_active = True
                self._silence_ms = 0.0
                self._speech_ms += duration_ms
            elif self._speech_active:
                self._silence_ms += duration_ms
                self._speech_ms += duration_ms
            if self._speech_active and (
                self._silence_ms >= 700 or self._speech_ms >= 15_000
            ):
                self.session.commit_turn()
                self._speech_active = False
                self._silence_ms = 0.0
                self._speech_ms = 0.0
        return frame
