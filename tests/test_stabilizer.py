import numpy as np

from stabilizer import DetectionStabilizer


def box_at(x, y, w=100, h=40):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


def det(x, y, text, conf=0.9, **kw):
    return (box_at(x, y, **kw), text, conf)


def test_single_sighting_is_withheld_until_it_has_support():
    """One-frame flukes shouldn't reach the display."""
    s = DetectionStabilizer(min_votes=2)
    assert s.update([det(10, 10, "GARBAGE")], now=0.0) == []


def test_repeated_sighting_is_reported():
    s = DetectionStabilizer(min_votes=2)
    s.update([det(10, 10, "HELLO")], now=0.0)
    out = s.update([det(12, 11, "HELLO")], now=0.1)
    assert [t for _, t, _ in out] == ["HELLO"]


def test_majority_reading_wins_over_intermittent_misreads():
    """The clock case: '8/22' seen consistently, with occasional garbage."""
    s = DetectionStabilizer(min_votes=2)
    readings = ["8/22", "822", "8/22", "8122", "8/22", "8*22", "8/22"]
    out = []
    for i, text in enumerate(readings):
        out = s.update([det(100, 50, text, conf=0.9)], now=i * 0.1)

    assert [t for _, t, _ in out] == ["8/22"]


def test_confidence_weighting_beats_raw_frequency():
    """A reading seen twice at high confidence should beat one seen three
    times but hesitantly."""
    s = DetectionStabilizer(min_votes=2)
    for i, (text, conf) in enumerate(
        [("332", 0.30), ("3:32", 0.95), ("332", 0.30), ("3:32", 0.95), ("332", 0.30)]
    ):
        out = s.update([det(100, 50, text, conf=conf)], now=i * 0.1)

    assert [t for _, t, _ in out] == ["3:32"]


def test_detections_in_different_places_stay_separate():
    s = DetectionStabilizer(min_votes=2)
    for i in range(3):
        out = s.update(
            [det(0, 0, "LEFT"), det(500, 300, "RIGHT")], now=i * 0.1
        )
    assert sorted(t for _, t, _ in out) == ["LEFT", "RIGHT"]


def test_stale_detections_fall_out_of_the_window():
    s = DetectionStabilizer(window_seconds=1.0, min_votes=2)
    s.update([det(10, 10, "OLD")], now=0.0)
    s.update([det(10, 10, "OLD")], now=0.1)
    assert [t for _, t, _ in s.update([], now=0.5)] == ["OLD"]

    # Well past the window with nothing new -- it should be forgotten.
    assert s.update([], now=5.0) == []


def test_text_that_moves_across_the_frame_is_still_one_cluster():
    """Slow drift (hand-held object) shouldn't split into separate readings."""
    s = DetectionStabilizer(min_votes=3)
    out = []
    for i in range(5):
        out = s.update([det(100 + i * 8, 50, "DRIFTING")], now=i * 0.1)
    assert [t for _, t, _ in out] == ["DRIFTING"]


def test_reset_clears_history():
    s = DetectionStabilizer(min_votes=2)
    s.update([det(10, 10, "A")], now=0.0)
    s.update([det(10, 10, "A")], now=0.1)
    s.reset()
    assert s.update([det(10, 10, "A")], now=0.2) == []


def test_empty_input_is_safe():
    s = DetectionStabilizer()
    assert s.update([], now=0.0) == []
