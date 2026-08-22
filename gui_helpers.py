"""Pure helper functions for the GUI, kept free of any GUI/display
dependency (no tkinter import here) so they can be unit tested without a
display server -- useful since CI runners often lack Tk entirely."""

from typing import Set, Tuple

import cv2
import numpy as np

from ocr_engine import Detections


def select_new_detections(results: Detections, already_logged: Set[str]) -> Tuple[Detections, Set[str]]:
    """Split out only the detections whose text hasn't been logged yet, so a
    steadily-visible sign doesn't spam the log every single frame."""
    current_texts = {text for _, text, _ in results}
    new_texts = current_texts - already_logged
    new_results = [r for r in results if r[1] in new_texts]
    return new_results, current_texts


def resize_for_display(frame: np.ndarray, max_width: int = 800) -> np.ndarray:
    """Shrink a frame for on-screen display only -- purely cosmetic, does
    not affect the resolution OCR actually runs on."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
