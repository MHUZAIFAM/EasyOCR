"""Runs OCR on a background thread so the display loop never blocks on it.

Camera capture/draw is fast (tens to hundreds of FPS); EasyOCR inference on
CPU is slow (roughly 1 FPS). Running both in one loop makes the whole
program only as fast as the slowest step. This worker instead always keeps
just the latest submitted frame, processes it whenever it's free, and
publishes results the main loop can read without waiting.
"""

import threading
from typing import Callable, List, Tuple

import numpy as np


class OCRWorker:
    def __init__(self, process_frame: Callable[[np.ndarray], List[Tuple[np.ndarray, str, float]]]):
        self._process_frame = process_frame
        self._lock = threading.Lock()
        self._pending_frame: np.ndarray | None = None
        self._results: List[Tuple[np.ndarray, str, float]] = []
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

    def latest_results(self) -> List[Tuple[np.ndarray, str, float]]:
        with self._lock:
            return self._results

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                threading.Event().wait(0.005)
                continue

            results = self._process_frame(frame)

            with self._lock:
                self._results = results
