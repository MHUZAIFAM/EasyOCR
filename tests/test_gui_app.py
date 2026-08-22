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


def test_video_panel_size_does_not_track_the_rendered_image(tk_root):
    """Regression test for the window creeping larger on every frame.

    The loop was: the rendered image sets the label's requested size, which
    sets the video frame's requested size, which grows an auto-sizing
    window, which enlarges the panel, which enlarges the next image.

    Asserting on the window's size directly is unreliable -- an explicit
    geometry() pins it and hides the bug, while without one the growth
    saturates against the screen before a test can sample it. So this
    asserts the underlying invariant instead: with geometry propagation
    off, the panel's *requested* size must stay independent of whatever
    image is currently displayed in it.
    """
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    with patch("gui_app.cv2.VideoCapture", return_value=FakeCapture(frame)):
        app = gui_app.OCRApp(tk_root, FakeEngine([]), camera_index=0, enhance=True, max_width=640)
        try:
            pump(tk_root, seconds=0.3, step=0.01)
            tk_root.update_idletasks()

            image_width = app.video_label.imgtk.width()
            requested_width = app.video_frame.winfo_reqwidth()

            # Pre-fix this was image_width + the frame's 8px of padding.
            assert requested_width < image_width, (
                f"video panel requests {requested_width}px to hold a {image_width}px "
                "image -- its size is tracking the image again, which re-creates "
                "the window-growth feedback loop"
            )
        finally:
            app.on_close()


def test_video_image_scales_up_when_the_window_grows(tk_root):
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    with patch("gui_app.cv2.VideoCapture", return_value=FakeCapture(frame)):
        app = gui_app.OCRApp(tk_root, FakeEngine([]), camera_index=0, enhance=True, max_width=640)
        try:
            tk_root.state("normal")
            tk_root.geometry("1000x650")
            pump(tk_root, seconds=0.3, step=0.01)
            small_width = app.video_label.imgtk.width()

            tk_root.geometry("1400x900")
            pump(tk_root, seconds=0.3, step=0.01)
            large_width = app.video_label.imgtk.width()

            assert large_width > small_width, (
                f"video did not scale up: {small_width}px -> {large_width}px"
            )
        finally:
            app.on_close()
