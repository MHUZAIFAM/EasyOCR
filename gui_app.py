"""Tkinter UI for real-time OCR.

Layout: a live annotated camera feed in the center, with FPS and a running
detection log on the right. Two extra actions live in the sidebar:

- "Get OCR": freezes the current frame and runs OCR on it at full
  resolution (no downscaling), trading the live loop's speed for accuracy
  on that one capture.
- "Load Image...": runs the same full-accuracy OCR on a file from disk.

Usage:
    python gui_app.py
    python gui_app.py --camera 1 --gpu
"""

import argparse
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, scrolledtext, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from gui_helpers import resize_for_display, select_new_detections
from ocr_engine import Detections, OCREngine
from ocr_worker import OCRWorker
from preprocessing import enhance_for_ocr
from realtime_ocr import downscale_for_detection, draw_results

FRAME_POLL_MS = 10


class OCRApp:
    def __init__(self, root: tk.Tk, engine: OCREngine, camera_index: int, enhance: bool, max_width: int):
        self.root = root
        self.engine = engine
        self.enhance = enhance
        self.max_width = max_width

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

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

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._update_loop()

    # -- OCR plumbing -----------------------------------------------------

    def _process_live_frame(self, frame: np.ndarray) -> Detections:
        small, scale = downscale_for_detection(frame, self.max_width)
        processed = enhance_for_ocr(small) if self.enhance else small
        results = self.engine.read(processed)
        if scale != 1.0:
            inv = 1.0 / scale
            results = [(np.round(box * inv).astype(int), text, conf) for box, text, conf in results]
        return results

    # -- UI construction ----------------------------------------------------

    def _build_ui(self) -> None:
        self.root.title("Real-Time OCR (EasyOCR)")

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(main)
        self.video_label.grid(row=0, column=0, padx=(0, 10), sticky="n")

        sidebar = ttk.Frame(main)
        sidebar.grid(row=0, column=1, sticky="n")

        self.fps_var = tk.StringVar(value="FPS: --")
        ttk.Label(sidebar, textvariable=self.fps_var, font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 10))

        ttk.Button(sidebar, text="Get OCR (capture frame)", command=self.capture_and_run_ocr).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(sidebar, text="Load Image...", command=self.load_image_and_run_ocr).pack(fill=tk.X, pady=2)

        ttk.Label(sidebar, text="Detections:").pack(anchor="w", pady=(12, 2))
        self.log = scrolledtext.ScrolledText(sidebar, width=40, height=26, state="disabled", wrap="word")
        self.log.pack(fill=tk.BOTH, expand=True)

    def _append_log(self, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert(tk.END, f"[{timestamp}] {text}\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    # -- live loop ----------------------------------------------------------

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
                self._append_log(f"(live) {text} ({conf:.2f})")

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
        display_frame = resize_for_display(frame)
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.imgtk = imgtk  # keep a reference so it isn't garbage collected
        self.video_label.configure(image=imgtk)

    # -- one-off high-accuracy actions ---------------------------------------

    def capture_and_run_ocr(self) -> None:
        if self.latest_frame is None:
            return
        frame = self.latest_frame.copy()
        self._append_log("Capturing frame for high-accuracy OCR...")

        def work():
            processed = enhance_for_ocr(frame) if self.enhance else frame
            results = self.engine.read(processed)  # full resolution, no downscale
            self.pending_results.put((self._show_capture_results, (frame, results)))

        threading.Thread(target=work, daemon=True).start()

    def _show_capture_results(self, frame: np.ndarray, results: Detections) -> None:
        if not results:
            self._append_log("(capture) no text detected")
            return
        for _, text, conf in results:
            self._append_log(f"(capture) {text} ({conf:.2f})")
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
            self._append_log(f"(image) could not read {path}")
            return

        self._append_log(f"Running OCR on {path}...")

        def work():
            processed = enhance_for_ocr(frame) if self.enhance else frame
            results = self.engine.read(processed)
            self.pending_results.put((self._show_image_results, (frame, results, path)))

        threading.Thread(target=work, daemon=True).start()

    def _show_image_results(self, frame: np.ndarray, results: Detections, path: str) -> None:
        if not results:
            self._append_log(f"(image) no text detected in {path}")
            return
        for _, text, conf in results:
            self._append_log(f"(image) {text} ({conf:.2f})")
        self._show_popup(draw_results(frame.copy(), results), title=f"Image Result: {path}")

    def _show_popup(self, frame: np.ndarray, title: str) -> None:
        popup = tk.Toplevel(self.root)
        popup.title(title)
        display_frame = resize_for_display(frame, max_width=900)
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        label = ttk.Label(popup, image=imgtk)
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


def main() -> None:
    args = parse_args()
    engine = OCREngine(languages=args.lang, use_gpu=args.gpu, min_confidence=args.min_confidence)

    root = tk.Tk()
    OCRApp(root, engine, args.camera, args.enhance, args.max_width)
    root.mainloop()


if __name__ == "__main__":
    main()
