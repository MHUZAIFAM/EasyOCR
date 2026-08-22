"""Thin wrapper around EasyOCR so the reader is created once and reused."""

import threading
from typing import List, Optional, Tuple

import numpy as np

# A single detection: (box corners, recognized text, confidence).
Detection = Tuple[np.ndarray, str, float]
Detections = List[Detection]


class OCREngine:
    def __init__(self, languages: Optional[List[str]] = None, use_gpu: bool = False,
                 min_confidence: float = 0.4, allowlist: Optional[str] = None):
        import easyocr  # imported lazily so --help etc. don't need torch installed

        self.reader = easyocr.Reader(languages or ["en"], gpu=use_gpu)
        self.min_confidence = min_confidence
        # Restricting the character set is a large accuracy win when the text
        # has a known format: on a clock face, "3:32" came back as "3.32"
        # unconstrained but "3:32" at 1.00 confidence with allowlist="0123456789:".
        self.allowlist = allowlist
        # EasyOCR's reader isn't guaranteed safe to call concurrently; the GUI
        # can trigger a one-off capture/image read while the continuous
        # background worker is mid-inference, so serialize access to it.
        self._lock = threading.Lock()

    def read(self, frame: np.ndarray, mag_ratio: float = 1.0) -> Detections:
        """Run detection + recognition on one frame.

        mag_ratio is EasyOCR's own pre-detection magnification. Raising it
        recovers small glyphs that are otherwise dropped -- a clock's colon
        read as "332" at 1.0 and "3:32" at 2.0 -- but costs time roughly in
        proportion (1.7s -> 5.8s on a 1280px frame). So the live loop leaves
        it at 1.0 and only the one-off capture/image paths raise it.
        """
        kwargs = {"mag_ratio": mag_ratio}
        if self.allowlist:
            kwargs["allowlist"] = self.allowlist

        with self._lock:
            results = self.reader.readtext(frame, **kwargs)
        return [
            (np.array(box, dtype=int), text, conf)
            for box, text, conf in results
            if conf >= self.min_confidence
        ]
