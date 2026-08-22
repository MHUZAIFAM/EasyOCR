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


def fit_frame_to_box(frame: np.ndarray, box_width: int, box_height: int) -> np.ndarray:
    """Scale a frame up or down (preserving aspect ratio) to fit within a
    box -- e.g. the live video panel, which should grow to fill the window
    on maximize rather than stay pinned at its original capture size.

    Purely cosmetic, like resize_for_display: never touches the resolution
    OCR actually runs on, only what gets drawn on screen."""
    h, w = frame.shape[:2]
    if box_width <= 0 or box_height <= 0 or h <= 0 or w <= 0:
        return frame
    scale = min(box_width / w, box_height / h)
    if scale <= 0:
        return frame
    new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
