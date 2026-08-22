"""Real-time OCR from a live camera feed, built to stay usable under
extreme lighting and poor visibility.

Usage:
    python realtime_ocr.py                 # live webcam (camera 0)
    python realtime_ocr.py --camera 1      # a different camera
    python realtime_ocr.py --image path.jpg  # single image, no webcam needed
    python realtime_ocr.py --no-enhance    # disable lighting preprocessing

OCR runs on a background thread (see ocr_worker.py) so the video stays
smooth instead of freezing every time a detection pass runs.
"""

import argparse
import time
from typing import Tuple

import cv2
import numpy as np

from ocr_engine import Detections, OCREngine
from ocr_worker import OCRWorker
from preprocessing import enhance_for_ocr


def downscale_for_detection(frame: np.ndarray, max_width: int) -> Tuple[np.ndarray, float]:
    """Shrink the frame before OCR to cut inference time; EasyOCR's cost
    scales with pixel count, so this matters far more than skipping frames.
    Returns the (possibly) resized frame and the scale factor to map
    detected boxes back onto the original frame.
    """
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame, 1.0
    scale = max_width / w
    resized = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def draw_results(frame: np.ndarray, results: Detections) -> np.ndarray:
    for box, text, conf in results:
        cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)
        x, y = box[0]
        label = f"{text} ({conf:.2f})"
        cv2.putText(
            frame, label, (int(x), max(int(y) - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
        )
    return frame


def run_on_image(engine: OCREngine, path: str, enhance: bool, output: str = None) -> None:
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    processed = enhance_for_ocr(frame) if enhance else frame
    results = engine.read(processed)

    for box, text, conf in results:
        print(f"[{conf:.2f}] {text}")

    annotated = draw_results(frame.copy(), results)
    out_path = output or "output.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"Saved annotated result to {out_path}")


def run_on_webcam(engine: OCREngine, camera_index: int, enhance: bool, max_width: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    def process_frame(frame):
        small, scale = downscale_for_detection(frame, max_width)
        processed = enhance_for_ocr(small) if enhance else small
        results = engine.read(processed)
        if scale != 1.0:
            inv = 1.0 / scale
            results = [(np.round(box * inv).astype(int), text, conf) for box, text, conf in results]
        return results

    worker = OCRWorker(process_frame)
    worker.start()

    prev_time = time.time()
    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed, stopping.")
                break

            worker.submit(frame)
            annotated = draw_results(frame.copy(), worker.latest_results())

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(
                annotated, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA,
            )

            cv2.imshow("Real-Time OCR (EasyOCR)", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time OCR with EasyOCR")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--image", type=str, default=None, help="Run on a single image instead of the webcam")
    parser.add_argument("--output", type=str, default=None, help="Where to save annotated output for --image mode")
    parser.add_argument("--lang", nargs="+", default=["en"], help="Languages for EasyOCR (default: en)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--min-confidence", type=float, default=0.4, help="Drop detections below this confidence")
    parser.add_argument("--no-enhance", dest="enhance", action="store_false", help="Disable lighting preprocessing")
    parser.add_argument(
        "--max-width", type=int, default=640,
        help="Downscale frames to this width before OCR to speed up CPU inference (0 disables)",
    )
    parser.set_defaults(enhance=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = OCREngine(languages=args.lang, use_gpu=args.gpu, min_confidence=args.min_confidence)

    if args.image:
        run_on_image(engine, args.image, args.enhance, args.output)
    else:
        run_on_webcam(engine, args.camera, args.enhance, args.max_width)


if __name__ == "__main__":
    main()
