# Real-Time OCR with EasyOCR

[![CI](https://github.com/MHUZAIFAM/EasyOCR/actions/workflows/ci.yml/badge.svg)](https://github.com/MHUZAIFAM/EasyOCR/actions/workflows/ci.yml)

A real-time text recognition system that reads text straight from a live camera feed using [EasyOCR](https://github.com/JaidedAI/EasyOCR) and OpenCV, with adaptive preprocessing so it keeps working under extreme lighting and poor visibility (very dark rooms, backlighting, glare/washout).

## Features

- **Live webcam OCR** — detects and overlays text with bounding boxes and confidence scores in real time, with the video feed staying smooth instead of freezing during detection.
- **Desktop GUI** — a live feed panel, an FPS + running detection log sidebar, a one-click high-accuracy frame capture, and a static image loader (see [GUI mode](#gui-mode) below).
- **Lighting-robust preprocessing** — automatic gamma correction based on measured frame brightness, plus CLAHE contrast enhancement and denoising for low-light frames.
- **Single-image mode** — run the same pipeline on a static image, useful for testing or batch processing without a camera.
- **Configurable** — camera index, OCR languages, GPU usage, confidence threshold, and detection resolution are all CLI flags.

## How it works

1. Grab a frame from the camera (or load an image).
2. Estimate frame brightness and apply gamma correction to pull very dark or overexposed frames back toward a usable range.
3. Apply CLAHE (adaptive histogram equalization) to boost local contrast, and denoise if the frame is still dark.
4. Run the corrected frame through EasyOCR's detector + recognizer.
5. Draw bounding boxes, recognized text, and confidence scores back on the original frame.

EasyOCR's CPU inference (roughly 1 FPS) is much slower than camera capture (tens to hundreds of FPS). Running both on the same thread would make the video stutter every time a detection pass ran, so OCR instead runs on a background thread ([`ocr_worker.py`](ocr_worker.py)) that always works on the most recent frame and publishes results the display loop reads without blocking — video stays smooth, and the on-screen text just refreshes whenever the next detection finishes.

See [`preprocessing.py`](preprocessing.py) for the enhancement pipeline and [`ocr_engine.py`](ocr_engine.py) for the EasyOCR wrapper.

## Installation

```bash
git clone https://github.com/MHUZAIFAM/EasyOCR.git
cd EasyOCR
pip install -r requirements.txt
pip uninstall -y opencv-python-headless
pip install --force-reinstall opencv-python
```

> EasyOCR pulls in PyTorch. For CPU-only machines, installing `torch` from the [CPU wheel index](https://pytorch.org/get-started/locally/) first is usually faster than the default GPU build.

> **Why the extra two lines:** EasyOCR depends on `opencv-python-headless`, which has no GUI backend. Installing it can silently overwrite the GUI-capable `opencv-python` this project needs for the live webcam window (both packages ship a `cv2` module under the same name, so pip has no idea they conflict). Re-running these two lines after *any* reinstall of `requirements.txt` fixes it. If you skip this and the window fails to open, `realtime_ocr.py` will detect it and print this exact fix instead of a cryptic OpenCV traceback.

## Usage

Live webcam:

```bash
python realtime_ocr.py
```

A different camera, with GPU acceleration:

```bash
python realtime_ocr.py --camera 1 --gpu
```

Single image instead of a webcam:

```bash
python realtime_ocr.py --image sample.jpg --output result.jpg
```

Disable the lighting preprocessing (to compare raw vs. enhanced accuracy):

```bash
python realtime_ocr.py --no-enhance
```

Trade detection accuracy for speed by shrinking frames further before OCR (default is 640px wide, 0 disables downscaling):

```bash
python realtime_ocr.py --max-width 960
```

Press `q` to quit the live window.

## GUI mode

```bash
python gui_app.py
python gui_app.py --camera 1 --gpu
```

A styled, resizable desktop window ([ttkbootstrap](https://ttkbootstrap.readthedocs.io/), Tokyo Night dark theme) with:

- **Center**: the live annotated feed, same detection loop as the CLI. Scales to fill the window on resize/maximize (preserving aspect ratio) — this is purely cosmetic and independent of the resolution OCR actually processes at, so resizing the window doesn't change detection speed or accuracy.
- **Sidebar**: current FPS with a status indicator, and a timestamped, color-coded log of everything detected (green for captures, blue for loaded images, default for live).
- **Get OCR (capture frame)**: freezes the current frame and re-runs OCR on it at full resolution — no downscaling — trading the live loop's speed for the best accuracy that single frame can get. Result opens in a popup and is logged as `(capture)`.
- **Load Image...**: pick any file from disk and run the same full-accuracy OCR on it, independent of the camera. Result opens in a popup and is logged as `(image)`.

The EasyOCR model (and the PyTorch it pulls in) can take several seconds to load. Rather than leave the window looking frozen during that, a loading screen appears immediately while the model loads on a background thread, then swaps in the real UI once it's ready.

The GUI reuses the same `OCREngine`/`OCRWorker`/`preprocessing` modules as the CLI — it's a different front-end on the same pipeline, not a separate implementation. It also sidesteps the `opencv-python-headless` GUI conflict entirely, since Tkinter never touches OpenCV's own window backend.

## Testing

The preprocessing, downscaling, and threading logic are covered by unit tests that don't need a webcam or GPU:

```bash
pip install -r requirements-dev.txt
pytest -v
```

CI runs this same suite on every push (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## Project structure

```
realtime_ocr.py    CLI entry point: webcam/image loop, downscaling, drawing
gui_app.py          Tkinter desktop UI on top of the same pipeline
gui_helpers.py       Pure GUI logic (log de-duplication, display resizing) kept testable
ocr_engine.py       EasyOCR wrapper (detector + recognizer)
ocr_worker.py       Background thread that decouples OCR speed from display FPS
preprocessing.py    Lighting-robust preprocessing (gamma correction, CLAHE, denoising)
tests/              Unit tests for the above
```

## Tech Stack

Python, EasyOCR, PyTorch, OpenCV, Tkinter, ttkbootstrap, Pillow, NumPy

## License

MIT — see [LICENSE](LICENSE).
