# Real-Time OCR with EasyOCR

A real-time text recognition system that reads text straight from a live camera feed using [EasyOCR](https://github.com/JaidedAI/EasyOCR) and OpenCV, with adaptive preprocessing so it keeps working under extreme lighting and poor visibility (very dark rooms, backlighting, glare/washout).

## Features

- **Live webcam OCR** — detects and overlays text with bounding boxes and confidence scores in real time.
- **Lighting-robust preprocessing** — automatic gamma correction based on measured frame brightness, plus CLAHE contrast enhancement and denoising for low-light frames.
- **Single-image mode** — run the same pipeline on a static image, useful for testing or batch processing without a camera.
- **Configurable** — camera index, OCR languages, GPU usage, confidence threshold, and detection frequency (to trade accuracy for FPS) are all CLI flags.

## How it works

1. Grab a frame from the camera (or load an image).
2. Estimate frame brightness and apply gamma correction to pull very dark or overexposed frames back toward a usable range.
3. Apply CLAHE (adaptive histogram equalization) to boost local contrast, and denoise if the frame is still dark.
4. Run the corrected frame through EasyOCR's detector + recognizer.
5. Draw bounding boxes, recognized text, and confidence scores back on the original frame.

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

Run detection every 3rd frame for higher FPS on slower machines:

```bash
python realtime_ocr.py --detect-every 3
```

Press `q` to quit the live window.

## Tech Stack

Python, EasyOCR, PyTorch, OpenCV, NumPy

## License

MIT — see [LICENSE](LICENSE).
