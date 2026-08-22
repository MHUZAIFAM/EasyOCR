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
        import cv2

        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._frame.shape[1]
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._frame.shape[0]
        return 0

    def set(self, prop, value):
        return True  # a real driver may ignore the request too

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

    with patch("realtime_ocr.cv2.VideoCapture", return_value=FakeCapture(frame)):
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

    with patch("realtime_ocr.cv2.VideoCapture", return_value=FakeCapture(frame)):
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


def test_feed_renders_at_a_fixed_size_regardless_of_window_size(tk_root):
    """The feed renders at a size fixed at startup, not one derived from the
    panel's current geometry.

    This is also the regression test for the window creeping larger on every
    frame. That bug was a feedback loop -- rendered image size set the
    label's requested size, which set the panel's, which grew an auto-sizing
    window, which enlarged the panel, which enlarged the next image. Because
    the render size is now a startup constant, the loop cannot close: a
    bigger window simply cannot produce a bigger image.

    Asserting on the window's size directly doesn't work -- an explicit
    geometry() pins it and hides the bug, and without one the growth
    saturates against the screen before a test can sample it.
    """
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    with patch("realtime_ocr.cv2.VideoCapture", return_value=FakeCapture(frame)):
        app = gui_app.OCRApp(tk_root, FakeEngine([]), camera_index=0, enhance=True, max_width=640)
        try:
            tk_root.state("normal")
            tk_root.geometry("1000x650")
            pump(tk_root, seconds=0.3, step=0.01)
            width_in_small_window = app.video_label.imgtk.width()

            tk_root.geometry("1600x1000")
            pump(tk_root, seconds=0.3, step=0.01)
            width_in_large_window = app.video_label.imgtk.width()

            assert width_in_small_window == width_in_large_window == app.display_width, (
                f"feed size followed the window ({width_in_small_window}px -> "
                f"{width_in_large_window}px); it must stay at the fixed "
                f"{app.display_width}px or the growth feedback loop is back"
            )
        finally:
            app.on_close()


def test_video_panel_hugs_the_feed_without_letterbox_bars(tk_root):
    """The panel is sized to the feed, so there are no dead bars around it --
    only the intended few pixels of border padding."""
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    with patch("realtime_ocr.cv2.VideoCapture", return_value=FakeCapture(frame)):
        app = gui_app.OCRApp(tk_root, FakeEngine([]), camera_index=0, enhance=True, max_width=640)
        try:
            pump(tk_root, seconds=0.3, step=0.01)
            tk_root.update_idletasks()

            allowed = 2 * gui_app.VIDEO_PADDING_PX
            width_gap = app.video_frame.winfo_width() - app.video_label.imgtk.width()
            height_gap = app.video_frame.winfo_height() - app.video_label.imgtk.height()

            assert width_gap <= allowed, f"{width_gap / 2:.0f}px bars either side of the feed"
            assert height_gap <= allowed, f"{height_gap / 2:.0f}px bars above/below the feed"
        finally:
            app.on_close()


class CyclingCapture:
    """Yields frames from a list in turn, imitating a camera whose subject is
    moving (blurred) most of the time and briefly still (sharp)."""

    def __init__(self, frames):
        self._frames = frames
        self._i = 0

    def isOpened(self):
        return True

    def read(self):
        frame = self._frames[self._i % len(self._frames)]
        self._i += 1
        return True, frame.copy()

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._frames[0].shape[1]
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._frames[0].shape[0]
        return 0

    def set(self, prop, value):
        return True

    def release(self):
        pass


class RecordingEngine:
    """Records every frame it is asked to read.

    Both the live loop and the capture button call read(), so tests have to
    keep all of them and pick out the one they mean -- the capture path is
    identifiable by its upscaled width.
    """

    def __init__(self):
        self.frames = []

    def read(self, frame):
        self.frames.append(frame)
        return []


def test_capture_picks_the_sharpest_recent_frame_not_the_current_one(tk_root):
    """Motion blur wrecks recognition, so 'Get OCR' should reach back for the
    sharpest of the last few frames rather than whatever was on screen when
    the button happened to be clicked."""
    import cv2

    from preprocessing import sharpness

    def make(blur_len):
        img = np.full((480, 640, 3), 240, dtype=np.uint8)
        cv2.putText(img, "CAPTURE ME 123", (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3)
        if blur_len > 1:
            kernel = np.zeros((blur_len, blur_len))
            kernel[blur_len // 2, :] = 1 / blur_len
            img = cv2.filter2D(img, -1, kernel)
        return img

    sharp, blurry = make(0), make(13)
    # Mostly blurred frames with a single sharp one in the middle.
    frames = [blurry, blurry, sharp, blurry, blurry]
    engine = RecordingEngine()

    # max_width=320 makes the live loop downscale, so its reads are 320px
    # wide while the capture path's full-resolution read is 640px -- that's
    # how we pick the capture's frame out of the recorded reads.
    with patch("realtime_ocr.cv2.VideoCapture", return_value=CyclingCapture(frames)):
        app = gui_app.OCRApp(tk_root, engine, camera_index=0, enhance=False, max_width=320)
        try:
            pump(tk_root, seconds=0.4, step=0.01)  # fill the rolling buffer
            assert len(app.recent_frames) > 1, "rolling frame buffer never filled"

            app.capture_and_run_ocr()
            pump(tk_root, seconds=0.5, step=0.01)

            captured = [f for f in engine.frames if f.shape[1] == 640]
            assert captured, "capture never reached the engine at full resolution"

            assert sharpness(captured[-1]) > sharpness(blurry), (
                "capture used a motion-blurred frame instead of the sharp one"
            )
        finally:
            app.on_close()


def test_display_size_follows_the_camera_aspect_ratio(tk_root):
    """A 16:9 camera must not be squashed into a 4:3 panel (or vice versa)."""
    widescreen = np.full((720, 1280, 3), 100, dtype=np.uint8)

    with patch("realtime_ocr.cv2.VideoCapture", return_value=FakeCapture(widescreen)):
        app = gui_app.OCRApp(
            tk_root, FakeEngine([]), camera_index=0, enhance=True, max_width=640, display_width=960
        )
        try:
            assert (app.display_width, app.display_height) == (960, 540)
        finally:
            app.on_close()
