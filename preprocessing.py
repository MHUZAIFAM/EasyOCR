"""Frame preprocessing to keep OCR usable under poor lighting/visibility."""

from typing import Sequence

import cv2
import numpy as np

# Below roughly this width, EasyOCR's recognizer starts losing characters
# because glyphs fall under the ~25-32px height it wants. Upscaling a sharp
# but small frame measurably helps (0.44 -> 0.73 confidence in testing).
OCR_MIN_WIDTH = 1280


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


def sharpness(frame: np.ndarray) -> float:
    """Variance of the Laplacian -- the standard focus/blur measure. Sharp
    edges produce a wide spread of second derivatives; blur flattens them."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pick_sharpest(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Choose the least motion-blurred frame from a burst.

    Motion blur is by far the worst thing for recognition accuracy -- in
    testing it dropped confidence from 0.44 to 0.01 and turned the text into
    garbage, and upscaling could not recover it. Picking the sharpest of the
    last few frames costs nothing and avoids capturing mid-movement.
    """
    if not len(frames):
        raise ValueError("no frames to choose from")
    return max(frames, key=sharpness)


def upscale_for_ocr(frame: np.ndarray, min_width: int = OCR_MIN_WIDTH) -> np.ndarray:
    """Enlarge a frame so tiny glyphs reach the size the recognizer expects.

    Opt-in, not a default, because measurement showed it cuts both ways: with
    ~12px-tall characters it lifted confidence 0.51 -> 0.75, but at ~22px it
    dropped 0.85 -> 0.68. EasyOCR does its own internal rescaling, and
    interpolating on top of that can hurt as easily as help. Worth enabling
    only for genuinely small text (distant signs, seven-segment displays).

    It also cannot rescue a blurred frame -- pair it with pick_sharpest.
    """
    h, w = frame.shape[:2]
    if w <= 0 or w >= min_width:
        return frame
    scale = min_width / w
    return cv2.resize(frame, (min_width, int(h * scale)), interpolation=cv2.INTER_CUBIC)
