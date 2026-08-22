import numpy as np

from realtime_ocr import downscale_for_detection, draw_results


def test_downscale_leaves_small_frames_untouched():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out, scale = downscale_for_detection(frame, max_width=640)
    assert out.shape == frame.shape
    assert scale == 1.0


def test_downscale_shrinks_wide_frames_and_reports_scale():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out, scale = downscale_for_detection(frame, max_width=640)
    assert out.shape[1] == 640
    assert scale == 0.5
    assert out.shape[0] == 360  # height scaled by the same factor


def test_downscale_disabled_with_zero_max_width():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out, scale = downscale_for_detection(frame, max_width=0)
    assert out.shape == frame.shape
    assert scale == 1.0


def test_detection_boxes_map_back_to_full_resolution():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, scale = downscale_for_detection(frame, max_width=640)

    box_in_small_frame = np.array([[100, 100], [200, 100], [200, 150], [100, 150]])
    mapped_back = np.round(box_in_small_frame / scale).astype(int)

    assert mapped_back.tolist() == [[200, 200], [400, 200], [400, 300], [200, 300]]


def test_draw_results_annotates_without_raising():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = np.array([[10, 10], [50, 10], [50, 30], [10, 30]])
    annotated = draw_results(frame.copy(), [(box, "hello", 0.9)])

    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated, frame)  # something was actually drawn


def test_draw_results_handles_empty_detections():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated = draw_results(frame.copy(), [])
    assert np.array_equal(annotated, frame)
