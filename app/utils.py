from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np


def parse_video_source(source: str | int) -> str | int:
    """Convert numeric camera indexes from string to int, keep paths/URLs unchanged."""
    if isinstance(source, int):
        return source

    value = str(source).strip()
    if value.isdigit():
        return int(value)
    return value


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def draw_status_panel(
    frame: np.ndarray,
    fps: float,
    violation_count: int,
    stream_status: str,
) -> None:
    """Render compact runtime metrics without covering most of the video frame."""
    lines = [
        f"FPS: {fps:.1f}",
        f"Violations: {violation_count}",
        f"Stream: {stream_status}",
        "Press q to quit",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    padding = 8
    line_height = 24
    width = 230
    height = padding * 2 + line_height * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + width, 10 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 10 + padding + 16
    for line in lines:
        cv2.putText(frame, line, (18, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height


class FPSCounter:
    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self._last_time = time.perf_counter()
        self._fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        elapsed = max(now - self._last_time, 1e-6)
        current_fps = 1.0 / elapsed
        self._fps = current_fps if self._fps == 0.0 else (
            self.smoothing * self._fps + (1.0 - self.smoothing) * current_fps
        )
        self._last_time = now
        return self._fps


class ViolationRecorder:
    """Save violation screenshots with a cooldown to avoid duplicate bursts."""

    def __init__(self, output_dir: str | Path, cooldown_seconds: float = 2.0) -> None:
        self.output_dir = ensure_directory(output_dir)
        self.cooldown_seconds = cooldown_seconds
        self._last_saved_at = 0.0
        self.count = 0

    def record_if_needed(self, frame: np.ndarray, has_violation: bool) -> bool:
        if not has_violation:
            return False

        now = time.monotonic()
        if now - self._last_saved_at < self.cooldown_seconds:
            return False

        filename = self.output_dir / f"violation_{now_timestamp()}.jpg"
        saved = cv2.imwrite(str(filename), frame)
        if saved:
            self._last_saved_at = now
            self.count += 1
        return saved


class ThreadedVideoStream:
    """Background video reader with reconnect handling for CCTV/RTSP streams."""

    def __init__(
        self,
        source: str | int,
        retry_delay: float = 2.0,
        max_retries: int = 0,
        queue_latest_only: bool = True,
    ) -> None:
        self.source = parse_video_source(source)
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.queue_latest_only = queue_latest_only
        self.reconnectable = self._is_reconnectable_source(self.source)

        self._capture: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._opened = threading.Event()
        self._retries = 0
        self.status = "initializing"

    def start(self) -> "ThreadedVideoStream":
        self._thread = threading.Thread(target=self._reader, name="video-stream-reader", daemon=True)
        self._thread.start()
        return self

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self) -> None:
        self._stopped.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release_capture()

    def wait_until_opened(self, timeout: float = 10.0) -> bool:
        return self._opened.wait(timeout)

    def _reader(self) -> None:
        while not self._stopped.is_set():
            if not self._open_capture():
                if not self._can_retry():
                    self.status = "failed"
                    break
                self._sleep_before_retry()
                continue

            self._retries = 0
            self.status = "connected"
            self._opened.set()

            while not self._stopped.is_set():
                ok, frame = self._capture.read() if self._capture else (False, None)
                if not ok or frame is None:
                    self.status = "disconnected"
                    self._clear_frame()
                    self._release_capture()
                    if not self._can_retry():
                        self.status = "ended"
                        self._stopped.set()
                    else:
                        self._sleep_before_retry()
                    break

                with self._lock:
                    self._frame = frame if self.queue_latest_only else frame.copy()

    def _open_capture(self) -> bool:
        self.status = "connecting"
        self._release_capture()

        capture = cv2.VideoCapture(self.source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not capture.isOpened():
            capture.release()
            self.status = f"retrying ({self._retries})"
            return False

        self._capture = capture
        return True

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _clear_frame(self) -> None:
        with self._lock:
            self._frame = None

    def _can_retry(self) -> bool:
        if not self.reconnectable:
            return False
        return self.max_retries <= 0 or self._retries < self.max_retries

    def _sleep_before_retry(self) -> None:
        self._retries += 1
        self.status = f"retrying ({self._retries})"
        self._stopped.wait(self.retry_delay)

    @staticmethod
    def _is_reconnectable_source(source: str | int) -> bool:
        if isinstance(source, int):
            return True

        parsed = urlparse(str(source))
        return parsed.scheme.lower() in {"rtsp", "rtmp", "http", "https"}


def safe_destroy_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def format_exception(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def source_to_string(source: Any) -> str:
    return str(source)
