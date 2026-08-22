# Real-Time OCR with EasyOCR

[![CI](https://github.com/MHUZAIFAM/EasyOCR/actions/workflows/ci.yml/badge.svg)](https://github.com/MHUZAIFAM/EasyOCR/actions/workflows/ci.yml)

A real-time text recognition system that reads text straight from a live camera feed using [EasyOCR](https://github.com/JaidedAI/EasyOCR) and OpenCV, with adaptive preprocessing so it keeps working under extreme lighting and poor visibility (very dark rooms, backlighting, glare/washout).

## Features

- **Live webcam OCR** — detects and overlays text with bounding boxes and confidence scores in real time, with the video feed staying smooth instead of freezing during detection.
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
```

> EasyOCR pulls in PyTorch. For CPU-only machines, installing `torch` from the [CPU wheel index](https://pytorch.org/get-started/locally/) first is usually faster than the default GPU build.

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
ocr_engine.py       EasyOCR wrapper (detector + recognizer)
ocr_worker.py       Background thread that decouples OCR speed from display FPS
preprocessing.py    Lighting-robust preprocessing (gamma correction, CLAHE, denoising)
tests/              Unit tests for the above
```

## Tech Stack

Python, EasyOCR, PyTorch, OpenCV, NumPy

## License

MIT — see [LICENSE](LICENSE).
