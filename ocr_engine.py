"""Thin wrapper around EasyOCR so the reader is created once and reused."""

import threading
from typing import List, Optional, Tuple

import numpy as np

# A single detection: (box corners, recognized text, confidence).
Detection = Tuple[np.ndarray, str, float]
Detections = List[Detection]


class OCREngine:
    def __init__(self, languages: Optional[List[str]] = None, use_gpu: bool = False, min_confidence: float = 0.4):
        import easyocr  # imported lazily so --help etc. don't need torch installed

        self.reader = easyocr.Reader(languages or ["en"], gpu=use_gpu)
        self.min_confidence = min_confidence
        # EasyOCR's reader isn't guaranteed safe to call concurrently; the GUI
        # can trigger a one-off capture/image read while the continuous
        # background worker is mid-inference, so serialize access to it.
        self._lock = threading.Lock()

    def read(self, frame: np.ndarray) -> Detections:
        with self._lock:
            results = self.reader.readtext(frame)
        return [
            (np.array(box, dtype=int), text, conf)
            for box, text, conf in results
            if conf >= self.min_confidence
        ]
