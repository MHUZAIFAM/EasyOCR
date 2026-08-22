import numpy as np

from gui_helpers import resize_for_display, select_new_detections


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
