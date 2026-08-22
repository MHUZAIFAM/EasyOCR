"""Tkinter UI for real-time OCR (styled with ttkbootstrap).

Layout: a live annotated camera feed in the center, with FPS and a running
detection log on the right. Two extra actions live in the sidebar:

- "Get OCR": freezes the current frame and runs OCR on it at full
  resolution (no downscaling), trading the live loop's speed for accuracy
  on that one capture.
- "Load Image...": runs the same full-accuracy OCR on a file from disk.

The EasyOCR model loads in the background behind a splash screen, since
loading it synchronously before the window appears made the app look frozen
for several seconds on startup.

Usage:
    python gui_app.py
    python gui_app.py --camera 1 --gpu
"""

import argparse
import queue
import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime
from tkinter import filedialog, scrolledtext

import cv2
import numpy as np
import ttkbootstrap as ttkb
from PIL import Image, ImageTk

from gui_helpers import fit_frame_to_box, resize_for_display, select_new_detections
from ocr_engine import Detections, OCREngine
from ocr_worker import OCRWorker
from preprocessing import enhance_for_ocr, pick_sharpest
from realtime_ocr import downscale_for_detection, draw_results, open_camera
from stabilizer import DetectionStabilizer

FRAME_POLL_MS = 10
THEME = "tokyo-night-dark"
SIDEBAR_WIDTH_PX = 360
DISPLAY_WIDTH_PX = 960  # fixed feed width; height follows the camera's aspect ratio
VIDEO_PADDING_PX = 4
# "Get OCR" picks the sharpest of the last few frames rather than whatever
# frame happened to be on screen, so a moment of hand movement doesn't ruin
# the capture. Kept short so it stays "the last instant", not stale footage.
CAPTURE_BUFFER_FRAMES = 8
# The capture button can afford EasyOCR's slower, more thorough detection
# pass; the live loop cannot (it roughly triples inference time).
CAPTURE_MAG_RATIO = 2.0

LOG_TAG_COLORS = {
    "live": None,  # falls back to the theme's default foreground
    "capture": "success",
    "image": "info",
    "system": "secondary",
}


def maximize_window(root: tk.Misc) -> None:
    """Open maximized, filling the screen. 'zoomed' is the Windows/macOS
    spelling; X11 builds of Tk expose it as a -zoomed attribute instead, and
    if neither works we fall back to sizing to the screen manually."""
    try:
        root.state("zoomed")
        return
    except tk.TclError:
        pass
    try:
        root.attributes("-zoomed", True)
        return
    except tk.TclError:
        root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")


class OCRApp:
    def __init__(self, root: ttkb.Window, engine: OCREngine, camera_index: int, enhance: bool,
                 max_width: int, display_width: int = DISPLAY_WIDTH_PX, stabilize: bool = True):
        self.root = root
        self.engine = engine
        self.enhance = enhance
        self.max_width = max_width
        # Live results are averaged over a short window so a stationary sign
        # reads steadily instead of flickering between per-frame guesses.
        self.stabilizer = DetectionStabilizer() if stabilize else None
        self.colors = ttkb.Style().colors

        self.cap = open_camera(camera_index)

        # Fix the display size up front, from the camera's aspect ratio. The
        # panel is then sized to exactly this, so the feed fills it edge to
        # edge with no letterbox/pillarbox bars. Computing it once (rather
        # than per frame from the panel's current size) also keeps the
        # render path independent of widget geometry entirely.
        cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self.display_width = display_width
        self.display_height = max(round(display_width * cam_h / cam_w), 1)

        self.latest_frame: np.ndarray = None
        # A short rolling window of recent frames, so "Get OCR" can pick the
        # sharpest one instead of whatever frame was on screen at click time.
        # Reuses frames the live loop already grabbed -- no extra camera
        # reads, no blocking the UI to collect a burst on demand.
        self.recent_frames: deque = deque(maxlen=CAPTURE_BUFFER_FRAMES)
        self.prev_time = time.time()
        self.logged_texts: set = set()
        self._last_results_version = -1
        self._stable_results: Detections = []
        # One-off capture/image results cross from a worker thread to the
        # main thread here. Tkinter itself isn't thread-safe, so results are
        # only ever consumed from _update_loop, which already runs on the
        # main thread via root.after().
        self.pending_results: "queue.Queue" = queue.Queue()

        self.worker = OCRWorker(self._process_live_frame)
        self.worker.start()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._update_loop()

    # -- OCR plumbing ---------------------------------------------------------

    def _process_live_frame(self, frame: np.ndarray) -> Detections:
        small, scale = downscale_for_detection(frame, self.max_width)
        processed = enhance_for_ocr(small) if self.enhance else small
        results = self.engine.read(processed)
        if scale != 1.0:
            inv = 1.0 / scale
            results = [(np.round(box * inv).astype(int), text, conf) for box, text, conf in results]
        return results

    # -- UI construction --------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.title("Real-Time OCR (EasyOCR)")
        self.root.resizable(True, True)  # the splash screen locks this off
        self.root.minsize(900, 600)
        maximize_window(self.root)

        header = ttkb.Frame(self.root, padding=(16, 12))
        header.pack(fill=tk.X)
        ttkb.Label(header, text="Real-Time OCR", font=("Segoe UI", 16, "bold")).pack(side=tk.LEFT)
        ttkb.Label(
            header, text="EasyOCR + OpenCV", bootstyle="secondary", font=("Segoe UI", 10)
        ).pack(side=tk.LEFT, padx=(10, 0), pady=(4, 0))

        ttkb.Separator(self.root).pack(fill=tk.X)

        main = ttkb.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)  # video panel takes the remaining width
        main.columnconfigure(1, minsize=SIDEBAR_WIDTH_PX)  # sidebar keeps a fixed width
        # Geometry propagation OFF is what keeps this layout stable. Otherwise
        # the rendered image's size becomes the label's requested size, which
        # becomes the frame's, which grows the window, which enlarges the
        # panel, which enlarges the next rendered image -- the window creeps
        # bigger every frame. With propagation off, sizes flow strictly one
        # way: window -> panels -> image.
        main.grid_propagate(False)

        # Sized to exactly fit the feed (plus its padding), and centered in
        # the cell -- so the bordered panel hugs the video instead of
        # stretching wide and leaving dark bars either side of it.
        self.video_frame = ttkb.Frame(
            main, bootstyle="dark", padding=VIDEO_PADDING_PX,
            width=self.display_width + 2 * VIDEO_PADDING_PX,
            height=self.display_height + 2 * VIDEO_PADDING_PX,
        )
        self.video_frame.grid(row=0, column=0, padx=(0, 12))
        self.video_frame.pack_propagate(False)
        self.video_label = ttkb.Label(self.video_frame, anchor="center")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.sidebar = ttkb.Frame(main, width=SIDEBAR_WIDTH_PX)
        self.sidebar.grid(row=0, column=1, sticky="nsew")
        self.sidebar.pack_propagate(False)
        sidebar = self.sidebar

        status_row = ttkb.Frame(sidebar)
        status_row.pack(fill=tk.X, pady=(0, 12))
        ttkb.Label(status_row, text="●", bootstyle="success", font=("Segoe UI", 12)).pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="FPS: --")
        ttkb.Label(status_row, textvariable=self.fps_var, font=("Segoe UI", 14, "bold")).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        ttkb.Button(
            sidebar, text="Get OCR (capture frame)", bootstyle="primary", command=self.capture_and_run_ocr
        ).pack(fill=tk.X, pady=3)
        ttkb.Button(
            sidebar, text="Load Image...", bootstyle="secondary-outline", command=self.load_image_and_run_ocr
        ).pack(fill=tk.X, pady=3)

        ttkb.Label(sidebar, text="Detections", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(14, 4))

        # width/height here are just a minimum request -- pack(expand=True)
        # stretches the log to fill whatever height the sidebar has. Deriving
        # a line count from the video panel's height (an earlier approach)
        # fed the sidebar's requested size back into the window size, which
        # made the window creep larger on every frame.
        self.log = scrolledtext.ScrolledText(
            sidebar, width=10, height=5, state="disabled", wrap="word",
            font=("Consolas", 9), borderwidth=0,
            bg=self.colors.bg, fg=self.colors.fg, insertbackground=self.colors.fg,
        )
        self.log.pack(fill=tk.BOTH, expand=True)
        for tag, color_name in LOG_TAG_COLORS.items():
            if color_name:
                self.log.tag_configure(tag, foreground=getattr(self.colors, color_name))

    def _append_log(self, text: str, tag: str = "live") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert(tk.END, f"[{timestamp}] {text}\n", tag)
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    # -- live loop ----------------------------------------------------------------

    def _update_loop(self) -> None:
        ok, frame = self.cap.read()
        if ok:
            self.latest_frame = frame
            self.recent_frames.append(frame)
            self.worker.submit(frame)

            # Only feed the stabilizer when OCR has actually finished a new
            # pass. This loop polls ~100x/sec while OCR completes ~1x/sec, so
            # voting on every poll would just count one result many times.
            version, raw_results = self.worker.latest_versioned_results()
            if self.stabilizer is None:
                results = raw_results
            else:
                if version != self._last_results_version:
                    self._last_results_version = version
                    self._stable_results = self.stabilizer.update(raw_results)
                results = self._stable_results

            now = time.time()
            fps = 1.0 / max(now - self.prev_time, 1e-6)
            self.prev_time = now
            self.fps_var.set(f"FPS: {fps:.1f}")

            new_results, self.logged_texts = select_new_detections(results, self.logged_texts)
            for _, text, conf in new_results:
                self._append_log(f"(live) {text} ({conf:.2f})", tag="live")

            annotated = draw_results(frame.copy(), results)
            self._render(annotated)

        self._drain_pending_results()
        self.root.after(FRAME_POLL_MS, self._update_loop)

    def _drain_pending_results(self) -> None:
        while True:
            try:
                handler, args = self.pending_results.get_nowait()
            except queue.Empty:
                break
            handler(*args)

    def _render(self, frame: np.ndarray) -> None:
        # Scales to the fixed display size chosen at startup, which the panel
        # is sized to match exactly. Deliberately does not consult the
        # panel's current geometry: doing so is what previously let the
        # rendered size feed back into the layout and grow the window.
        display_frame = fit_frame_to_box(frame, self.display_width, self.display_height)
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.imgtk = imgtk  # keep a reference so it isn't garbage collected
        self.video_label.configure(image=imgtk)

    # -- one-off high-accuracy actions -------------------------------------------

    def capture_and_run_ocr(self) -> None:
        if not self.recent_frames:
            return
        # Pick the sharpest of the last few frames rather than the current
        # one -- motion blur is the single worst thing for accuracy, and a
        # click often lands while the subject is still moving.
        frame = pick_sharpest(list(self.recent_frames)).copy()
        self._append_log("Capturing frame for high-accuracy OCR...", tag="system")

        def work():
            results = self._read_at_full_accuracy(frame)
            self.pending_results.put((self._show_capture_results, (frame, results)))

        threading.Thread(target=work, daemon=True).start()

    def _read_at_full_accuracy(self, frame: np.ndarray) -> Detections:
        """OCR a single frame with no downscaling and EasyOCR's slower,
        higher-magnification detection pass -- used by the capture and
        load-image buttons, which are one-offs and can spend the time the
        live loop can't."""
        processed = enhance_for_ocr(frame) if self.enhance else frame
        return self.engine.read(processed, mag_ratio=CAPTURE_MAG_RATIO)

    def _show_capture_results(self, frame: np.ndarray, results: Detections) -> None:
        if not results:
            self._append_log("(capture) no text detected", tag="capture")
            return
        for _, text, conf in results:
            self._append_log(f"(capture) {text} ({conf:.2f})", tag="capture")
        self._show_popup(draw_results(frame.copy(), results), title="Captured Frame Result")

    def load_image_and_run_ocr(self) -> None:
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            self._append_log(f"could not read {path}", tag="system")
            return

        self._append_log(f"Running OCR on {path}...", tag="system")

        def work():
            results = self._read_at_full_accuracy(frame)
            self.pending_results.put((self._show_image_results, (frame, results, path)))

        threading.Thread(target=work, daemon=True).start()

    def _show_image_results(self, frame: np.ndarray, results: Detections, path: str) -> None:
        if not results:
            self._append_log(f"(image) no text detected in {path}", tag="image")
            return
        for _, text, conf in results:
            self._append_log(f"(image) {text} ({conf:.2f})", tag="image")
        self._show_popup(draw_results(frame.copy(), results), title=f"Image Result: {path}")

    def _show_popup(self, frame: np.ndarray, title: str) -> None:
        popup = ttkb.Toplevel(self.root)
        popup.title(title)
        display_frame = resize_for_display(frame, max_width=900)
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        label = ttkb.Label(popup, image=imgtk)
        label.imgtk = imgtk
        label.pack()

    # -- shutdown -------------------------------------------------------------

    def on_close(self) -> None:
        self.worker.stop()
        self.cap.release()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time OCR GUI (EasyOCR)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--lang", nargs="+", default=["en"], help="Languages for EasyOCR (default: en)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--min-confidence", type=float, default=0.4, help="Drop detections below this confidence")
    parser.add_argument("--no-enhance", dest="enhance", action="store_false", help="Disable lighting preprocessing")
    parser.add_argument(
        "--max-width", type=int, default=640,
        help="Downscale live frames to this width before OCR (0 disables). Captures/images always run full-res.",
    )
    parser.add_argument(
        "--display-width", type=int, default=DISPLAY_WIDTH_PX,
        help=f"On-screen width of the feed in pixels (default: {DISPLAY_WIDTH_PX}). "
             "Height follows the camera's aspect ratio. Display only -- does not affect OCR.",
    )
    parser.add_argument(
        "--allowlist", type=str, default=None,
        help="Restrict recognition to these characters. A large accuracy win "
             "when the text has a known format -- e.g. --allowlist '0123456789:/' "
             "for a clock read '3:32' at 1.00 confidence where the unrestricted "
             "model returned '3.32'.",
    )
    parser.add_argument(
        "--no-stabilize", dest="stabilize", action="store_false",
        help="Show each frame's raw OCR result instead of a consensus over the "
             "last couple of seconds. Noisier, but responds instantly.",
    )
    parser.set_defaults(enhance=True, stabilize=True)
    return parser.parse_args()


def _show_splash(root: ttkb.Window) -> ttkb.Frame:
    root.title("Real-Time OCR (EasyOCR)")
    root.resizable(False, False)

    splash = ttkb.Frame(root, padding=48)
    splash.pack(fill=tk.BOTH, expand=True)
    ttkb.Label(splash, text="Real-Time OCR", font=("Segoe UI", 20, "bold")).pack(pady=(0, 8))
    ttkb.Label(
        splash, text="Loading EasyOCR model...\nThis can take a while on first run.",
        justify=tk.CENTER, bootstyle="secondary",
    ).pack(pady=(0, 16))
    progress = ttkb.Progressbar(splash, mode="indeterminate", bootstyle="info-striped", length=260)
    progress.pack()
    progress.start(12)
    return splash


def main() -> None:
    args = parse_args()

    root = ttkb.Window(themename=THEME)
    splash = _show_splash(root)

    state = {}

    def load_engine():
        state["engine"] = OCREngine(
            languages=args.lang, use_gpu=args.gpu,
            min_confidence=args.min_confidence, allowlist=args.allowlist,
        )
        root.after(0, launch_app)

    def launch_app():
        splash.destroy()
        OCRApp(root, state["engine"], args.camera, args.enhance, args.max_width,
               args.display_width, args.stabilize)

    threading.Thread(target=load_engine, daemon=True).start()
    root.mainloop()


if __name__ == "__main__":
    main()
