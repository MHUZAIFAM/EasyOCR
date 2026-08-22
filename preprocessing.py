"""Frame preprocessing to keep OCR usable under poor lighting/visibility."""

import cv2
import numpy as np


def estimate_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def gamma_correct(frame: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / max(gamma, 1e-3)
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(frame, table)


def apply_clahe(gray: np.ndarray, clip_limit: float = 3.0, tile_grid: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10)


def enhance_for_ocr(frame: np.ndarray, target_brightness: float = 130.0) -> np.ndarray:
    """Adaptively brighten/contrast-correct a frame before running OCR.

    Handles two failure modes that plain OCR struggles with: very dark frames
    (low-light rooms, backlit shots) and washed-out/glare frames.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = estimate_brightness(gray)

    if brightness < target_brightness - 40:
        gamma = min(2.5, target_brightness / max(brightness, 1.0))
        frame = gamma_correct(frame, gamma)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif brightness > target_brightness + 60:
        gamma = max(0.5, target_brightness / max(brightness, 1.0))
        frame = gamma_correct(frame, gamma)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE degenerates on near-flat regions (e.g. a blown-out glare patch):
    # with almost no variance to redistribute, histogram equalization can
    # push the whole region to full white instead of leaving it alone.
    if gray.std() > 2.0:
        gray = apply_clahe(gray)

    if brightness < 60:
        gray = denoise(gray)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
