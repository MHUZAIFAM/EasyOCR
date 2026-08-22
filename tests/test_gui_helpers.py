import numpy as np
import pytest

from gui_helpers import fit_frame_to_box, resize_for_display, select_new_detections


def make_detection(text: str, conf: float = 0.9):
    return (np.zeros((4, 2), dtype=int), text, conf)


def test_select_new_detections_returns_all_when_nothing_logged_yet():
    results = [make_detection("A"), make_detection("B")]
    new_results, logged = select_new_detections(results, already_logged=set())
    assert {r[1] for r in new_results} == {"A", "B"}
    assert logged == {"A", "B"}


def test_select_new_detections_skips_already_logged_text():
    results = [make_detection("A"), make_detection("B")]
    new_results, logged = select_new_detections(results, already_logged={"A"})
    assert [r[1] for r in new_results] == ["B"]
    assert logged == {"A", "B"}


def test_select_new_detections_handles_empty_results():
    new_results, logged = select_new_detections([], already_logged={"A"})
    assert new_results == []
    assert logged == set()


def test_resize_for_display_leaves_narrow_frames_untouched():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = resize_for_display(frame, max_width=800)
    assert out.shape == frame.shape


def test_resize_for_display_shrinks_wide_frames():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = resize_for_display(frame, max_width=800)
    assert out.shape[1] == 800
    assert out.shape[0] == 450  # 1080 * (800/1920)


def test_fit_frame_to_box_scales_up_to_fill_a_larger_box():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = fit_frame_to_box(frame, box_width=1280, box_height=960)
    assert out.shape[1] == 1280
    assert out.shape[0] == 960


def test_fit_frame_to_box_scales_down_to_fit_a_smaller_box():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = fit_frame_to_box(frame, box_width=640, box_height=640)
    # width is the limiting dimension (1920 -> 640 needs a bigger shrink than 1080 -> 640)
    assert out.shape[1] == 640
    assert out.shape[0] == 360


def test_fit_frame_to_box_preserves_aspect_ratio_when_box_ratio_differs():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 4:3
    out = fit_frame_to_box(frame, box_width=1000, box_height=200)  # very wide, short box
    # height is the limiting dimension
    assert out.shape[0] == 200
    assert out.shape[1] == pytest.approx(266, abs=1)


def test_fit_frame_to_box_handles_degenerate_box_sizes():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert fit_frame_to_box(frame, box_width=0, box_height=480).shape == frame.shape
    assert fit_frame_to_box(frame, box_width=640, box_height=0).shape == frame.shape
