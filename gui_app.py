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
import tkinter.font as tkfont
from datetime import datetime
from tkinter import filedialog, scrolledtext

import cv2
import numpy as np
import ttkbootstrap as ttkb
from PIL import Image, ImageTk

from gui_helpers import fit_frame_to_box, resize_for_display, select_new_detections
from ocr_engine import Detections, OCREngine
from ocr_worker import OCRWorker
from preprocessing import enhance_for_ocr
from realtime_ocr import downscale_for_detection, draw_results

FRAME_POLL_MS = 10
THEME = "tokyo-night-dark"

LOG_TAG_COLORS = {
    "live": None,  # falls back to the theme's default foreground
    "capture": "success",
    "image": "info",
    "system": "secondary",
}


class OCRApp:
    def __init__(self, root: ttkb.Window, engine: OCREngine, camera_index: int, enhance: bool, max_width: int):
        self.root = root
        self.engine = engine
        self.enhance = enhance
        self.max_width = max_width
        self.colors = ttkb.Style().colors

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")
        self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        self.latest_frame: np.ndarray = None
        self.prev_time = time.time()
        self.logged_texts: set = set()
        # One-off capture/image results cross from a worker thread to the
        # main thread here. Tkinter itself isn't thread-safe, so results are
        # only ever consumed from _update_loop, which already runs on the
        # main thread via root.after().
        self.pending_results: "queue.Queue" = queue.Queue()

        self.worker = OCRWorker(self._process_live_frame)
        self.worker.start()

        self._resize_after_id: str = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Configure>", self._on_resize)
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
        self.root.resizable(True, True)

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
        main.columnconfigure(0, weight=3)  # video panel gets most of any extra space
        main.columnconfigure(1, weight=1)  # sidebar grows too, but stays narrower

        self.video_frame = ttkb.Frame(main, bootstyle="dark", padding=4)
        self.video_frame.grid(row=0, column=0, padx=(0, 12), sticky="nsew")
        self.video_label = ttkb.Label(self.video_frame)
        self.video_label.pack(expand=True)

        self.sidebar = ttkb.Frame(main)
        self.sidebar.grid(row=0, column=1, sticky="nsew")
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

        # A modest placeholder height; _sync_log_height() below replaces this
        # with a value measured against the actual rendered video panel, once
        # there's a real frame to measure against.
        self.log_font = ("Consolas", 9)
        self.log = scrolledtext.ScrolledText(
            sidebar, width=44, height=12, state="disabled", wrap="word",
            font=self.log_font, borderwidth=0,
            bg=self.colors.bg, fg=self.colors.fg, insertbackground=self.colors.fg,
        )
        self.log.pack(fill=tk.BOTH, expand=True)
        for tag, color_name in LOG_TAG_COLORS.items():
            if color_name:
                self.log.tag_configure(tag, foreground=getattr(self.colors, color_name))
        self._log_height_synced = False

    def _sync_log_height(self, attempts_left: int = 20) -> None:
        """Resize the log to match the video panel's actual rendered height.

        Estimating this from font metrics alone was fragile -- DPI scaling
        and widget padding aren't predictable in advance -- so instead this
        measures the real, already-laid-out geometry once a frame exists.

        winfo_height() reports 1 (not real geometry) until a widget has
        actually been mapped on screen at least once, so this retries for a
        bit rather than risk computing a target height from that placeholder.
        """
        self.root.update_idletasks()
        video_px = self.video_frame.winfo_height()
        sidebar_px = self.sidebar.winfo_height()
        log_px = self.log.winfo_height()

        if attempts_left > 0 and (video_px <= 1 or sidebar_px <= 1 or log_px <= 1):
            self.root.after(30, lambda: self._sync_log_height(attempts_left - 1))
            return

        overhead_px = sidebar_px - log_px
        target_log_px = max(video_px - overhead_px, 100)
        line_height_px = tkfont.Font(font=self.log_font).metrics("linespace")
        self.log.configure(height=max(target_log_px // line_height_px, 6))

    def _on_resize(self, event: tk.Event) -> None:
        if event.widget is not self.root:
            return
        # <Configure> fires continuously while dragging a resize handle;
        # debounce so we only recompute once the window settles.
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(150, self._sync_log_height)

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
            self.worker.submit(frame)
            results = self.worker.latest_results()

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
        # Scale into whatever size the video panel currently has, so the
        # feed grows to fill the window on maximize/resize instead of
        # staying pinned at its original capture resolution. Before the
        # panel has ever been mapped, winfo_width/height() report 1 (a
        # Tk placeholder, not real geometry); fall back to a sane default
        # for that first frame or two.
        box_w, box_h = self.video_frame.winfo_width(), self.video_frame.winfo_height()
        if box_w <= 1 or box_h <= 1:
            box_w, box_h = 800, 600
        display_frame = fit_frame_to_box(frame, box_w - 8, box_h - 8)  # minus the frame's own padding
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.imgtk = imgtk  # keep a reference so it isn't garbage collected
        self.video_label.configure(image=imgtk)

        if not self._log_height_synced:
            self._log_height_synced = True
            self.root.after_idle(self._sync_log_height)

    # -- one-off high-accuracy actions -------------------------------------------

    def capture_and_run_ocr(self) -> None:
        if self.latest_frame is None:
            return
        frame = self.latest_frame.copy()
        self._append_log("Capturing frame for high-accuracy OCR...", tag="system")

        def work():
            processed = enhance_for_ocr(frame) if self.enhance else frame
            results = self.engine.read(processed)  # full resolution, no downscale
            self.pending_results.put((self._show_capture_results, (frame, results)))

        threading.Thread(target=work, daemon=True).start()

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
            processed = enhance_for_ocr(frame) if self.enhance else frame
            results = self.engine.read(processed)
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
    parser.set_defaults(enhance=True)
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
        state["engine"] = OCREngine(languages=args.lang, use_gpu=args.gpu, min_confidence=args.min_confidence)
        root.after(0, launch_app)

    def launch_app():
        splash.destroy()
        OCRApp(root, state["engine"], args.camera, args.enhance, args.max_width)

    threading.Thread(target=load_engine, daemon=True).start()
    root.mainloop()


if __name__ == "__main__":
    main()
