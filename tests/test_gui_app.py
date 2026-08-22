"""GUI integration tests. Skipped automatically wherever Tk isn't usable --
either because the tkinter module isn't built into this Python at all
(common on minimal CI images), or because it is but there's no display for
it to attach to (also common on CI)."""

import time
from unittest.mock import patch

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")  # skip this whole file if Tk isn't installed
ttkb = pytest.importorskip("ttkbootstrap")  # and if ttkbootstrap isn't installed

import gui_app  # noqa: E402 (must come after the importorskip guards above)


class FakeCapture:
    def __init__(self, frame):
        self._frame = frame

    def isOpened(self):
        return True

    def read(self):
        return True, self._frame.copy()

    def get(self, prop):
        return self._frame.shape[0]  # CAP_PROP_FRAME_HEIGHT

    def release(self):
        pass


class FakeEngine:
    def __init__(self, detections):
        self._detections = detections

    def read(self, frame):
        return self._detections


@pytest.fixture
def tk_root():
    try:
        root = ttkb.Window(themename="tokyo-night-dark")
        root.withdraw()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    yield root
    try:
        root.destroy()
    except tk.TclError:
        pass


def pump(root, seconds=0.3, step=0.05):
    """Process the Tk event queue for a bit, so scheduled after()/worker
    thread hand-offs (via the pending_results queue) get a chance to run."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        time.sleep(step)


def test_live_detection_appears_in_log(tk_root):
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    detection = (np.array([[10, 10], [50, 10], [50, 30], [10, 30]]), "LIVE TEXT", 0.9)

    with patch("gui_app.cv2.VideoCapture", return_value=FakeCapture(frame)):
        app = gui_app.OCRApp(tk_root, FakeEngine([detection]), camera_index=0, enhance=True, max_width=640)
        try:
            pump(tk_root, seconds=0.4)
            log_content = app.log.get("1.0", tk.END)
            assert "LIVE TEXT" in log_content
            assert "(live)" in log_content
        finally:
            app.on_close()


def test_capture_button_uses_full_resolution_and_logs_via_queue(tk_root):
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    detection = (np.array([[10, 10], [50, 10], [50, 30], [10, 30]]), "CAPTURED TEXT", 0.95)

    with patch("gui_app.cv2.VideoCapture", return_value=FakeCapture(frame)):
        app = gui_app.OCRApp(tk_root, FakeEngine([detection]), camera_index=0, enhance=True, max_width=640)
        try:
            pump(tk_root, seconds=0.2)  # let latest_frame populate
            app.capture_and_run_ocr()
            pump(tk_root, seconds=0.4)

            log_content = app.log.get("1.0", tk.END)
            assert "(capture)" in log_content
            assert "CAPTURED TEXT" in log_content
        finally:
            app.on_close()
