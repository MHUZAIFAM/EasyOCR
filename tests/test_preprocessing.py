import cv2
import numpy as np
import pytest

from preprocessing import (
    apply_clahe,
    enhance_for_ocr,
    estimate_brightness,
    gamma_correct,
    pick_sharpest,
    sharpness,
    upscale_for_ocr,
)


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


def textured_frame(blur_len: int = 0) -> np.ndarray:
    """A frame with real edges, optionally motion-blurred."""
    img = np.full((160, 640, 3), 240, dtype=np.uint8)
    cv2.putText(img, "SHARPNESS TEST 123", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    if blur_len > 1:
        kernel = np.zeros((blur_len, blur_len))
        kernel[blur_len // 2, :] = 1 / blur_len
        img = cv2.filter2D(img, -1, kernel)
    return img


def test_sharpness_scores_a_blurred_frame_lower_than_a_sharp_one():
    assert sharpness(textured_frame(blur_len=0)) > sharpness(textured_frame(blur_len=11))


def test_sharpness_of_a_flat_frame_is_near_zero():
    assert sharpness(solid_frame(128)) < 1.0


def test_pick_sharpest_finds_the_unblurred_frame_in_a_burst():
    burst = [textured_frame(11), textured_frame(9), textured_frame(0), textured_frame(7)]
    chosen = pick_sharpest(burst)
    assert chosen is burst[2]


def test_pick_sharpest_rejects_an_empty_burst():
    with pytest.raises(ValueError):
        pick_sharpest([])


def test_upscale_for_ocr_enlarges_small_frames_preserving_aspect():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = upscale_for_ocr(frame, min_width=1280)
    assert out.shape[1] == 1280
    assert out.shape[0] == 960  # 480 * (1280/640)


def test_upscale_for_ocr_leaves_already_large_frames_alone():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = upscale_for_ocr(frame, min_width=1280)
    assert out.shape == frame.shape


def test_upscaled_detection_boxes_map_back_to_original_coordinates():
    """The capture path runs OCR on an upscaled frame but draws boxes on the
    original, so the inverse scaling has to line up."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    upscaled = upscale_for_ocr(frame, min_width=1280)

    scale = upscaled.shape[1] / frame.shape[1]
    assert scale == 2.0

    box_on_upscaled = np.array([[200, 200], [400, 200], [400, 300], [200, 300]])
    mapped_back = np.round(box_on_upscaled / scale).astype(int)
    assert mapped_back.tolist() == [[100, 100], [200, 100], [200, 150], [100, 150]]
