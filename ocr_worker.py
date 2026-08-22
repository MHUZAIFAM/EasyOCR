"""Runs OCR on a background thread so the display loop never blocks on it.

Camera capture/draw is fast (tens to hundreds of FPS); EasyOCR inference on
CPU is slow (roughly 1 FPS). Running both in one loop makes the whole
program only as fast as the slowest step. This worker instead always keeps
just the latest submitted frame, processes it whenever it's free, and
publishes results the main loop can read without waiting.
"""

import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np

from ocr_engine import Detections

IDLE_POLL_SECONDS = 0.005


class OCRWorker:
    def __init__(self, process_frame: Callable[[np.ndarray], Detections]):
        self._process_frame = process_frame
        self._lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._results: Detections = []
        # Bumped on each completed pass. The display loop polls far faster
        # than OCR completes, so callers that must react once per *result*
        # (rather than once per poll) can watch this instead of the list.
        self._version = 0
        self._running = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2)

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending_frame = frame

    def latest_results(self) -> Detections:
        with self._lock:
            return self._results

    def latest_versioned_results(self) -> Tuple[int, Detections]:
        """Results plus the version they came from, read atomically."""
        with self._lock:
            return self._version, self._results

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                time.sleep(IDLE_POLL_SECONDS)
                continue

            results = self._process_frame(frame)

            with self._lock:
                self._results = results
                self._version += 1
