"""Thin wrapper around EasyOCR so the reader is created once and reused."""

from typing import List, Tuple

import numpy as np


class OCREngine:
    def __init__(self, languages: List[str] = None, use_gpu: bool = False, min_confidence: float = 0.4):
        import easyocr  # imported lazily so --help etc. don't need torch installed

        self.reader = easyocr.Reader(languages or ["en"], gpu=use_gpu)
        self.min_confidence = min_confidence

    def read(self, frame: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
        results = self.reader.readtext(frame)
        return [
            (np.array(box, dtype=int), text, conf)
            for box, text, conf in results
            if conf >= self.min_confidence
        ]
