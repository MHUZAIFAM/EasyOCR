import numpy as np

from preprocessing import apply_clahe, enhance_for_ocr, estimate_brightness, gamma_correct


def solid_frame(value: int, shape=(100, 100, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def test_estimate_brightness_matches_mean_pixel_value():
    gray = np.full((10, 10), 42, dtype=np.uint8)
    assert estimate_brightness(gray) == 42.0


def test_gamma_correct_identity_at_gamma_one():
    frame = solid_frame(120)
    corrected = gamma_correct(frame, gamma=1.0)
    assert np.allclose(corrected, frame, atol=1)


def test_gamma_correct_brightens_with_gamma_above_one():
    frame = solid_frame(60)
    corrected = gamma_correct(frame, gamma=2.0)
    assert corrected.mean() > frame.mean()


def test_gamma_correct_darkens_with_gamma_below_one():
    frame = solid_frame(200)
    corrected = gamma_correct(frame, gamma=0.5)
    assert corrected.mean() < frame.mean()


def test_apply_clahe_preserves_shape_and_dtype():
    gray = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    out = apply_clahe(gray)
    assert out.shape == gray.shape
    assert out.dtype == gray.dtype


def test_enhance_for_ocr_brightens_dark_frames():
    dark = solid_frame(20)
    enhanced = enhance_for_ocr(dark)
    assert enhanced.mean() > dark.mean()


def test_enhance_for_ocr_darkens_washed_out_frames():
    bright = solid_frame(250)
    enhanced = enhance_for_ocr(bright)
    assert enhanced.mean() < bright.mean()


def test_enhance_for_ocr_keeps_frame_shape_and_dtype():
    frame = solid_frame(130, shape=(64, 96, 3))
    enhanced = enhance_for_ocr(frame)
    assert enhanced.shape == frame.shape
    assert enhanced.dtype == frame.dtype
