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

- **Center**: the live annotated feed, same detection loop as the CLI. Rendered at a fixed size (`--display-width`, default 960px wide, height following the camera's aspect ratio) with the panel sized to match it exactly, so there are no letterbox bars around the video. Display only — independent of the resolution OCR actually processes at, so it doesn't affect detection speed or accuracy.
- **Sidebar**: current FPS with a status indicator, and a timestamped, color-coded log of everything detected (green for captures, blue for loaded images, default for live). Live readings are stabilized over a short window so they don't flicker — see [Accuracy notes](#accuracy-notes).
- **Get OCR (capture frame)**: picks the **sharpest of the last few frames** and re-runs OCR on it at full resolution with a higher `mag_ratio` — no downscaling — trading the live loop's speed for the best accuracy that moment can give. Result opens in a popup and is logged as `(capture)`.
- **Load Image...**: pick any file from disk and run the same full-accuracy OCR on it, independent of the camera. Result opens in a popup and is logged as `(image)`.

The EasyOCR model (and the PyTorch it pulls in) can take several seconds to load. Rather than leave the window looking frozen during that, a loading screen appears immediately while the model loads on a background thread, then swaps in the real UI once it's ready.

The GUI reuses the same `OCREngine`/`OCRWorker`/`preprocessing` modules as the CLI — it's a different front-end on the same pipeline, not a separate implementation. It also sidesteps the `opencv-python-headless` GUI conflict entirely, since Tkinter never touches OpenCV's own window backend.

## Accuracy notes

Some measured findings, since a few of them are counterintuitive:

**Motion blur is the single biggest accuracy killer.** On the same text, a motion-blurred frame scored 0.03 confidence and returned `'Po1*8i9? (30r0i@)'`, while a sharp frame scored 0.71 and returned `"POND'S: $5.99 (50% offl)"`. This is why a webcam capture often reads far worse than a saved photo — the photo was taken while holding still. "Get OCR" therefore picks the sharpest of the last several frames (by variance of the Laplacian) instead of whatever frame was on screen when you clicked.

**Camera resolution matters and OpenCV under-requests it.** OpenCV opens most webcams at 640×480 even when the hardware supports more, which is a hard ceiling on recognition. Both entry points now ask for 720p and fall back silently if the driver refuses.

**Manual upscaling before OCR is not a reliable win** — measured at ~12px character height it helped (0.51 → 0.75), but at ~22px it *hurt* (0.85 → 0.68). EasyOCR rescales internally and interpolating on top fights it, so this was tried and dropped in favour of `mag_ratio` below, which is the mechanism EasyOCR provides for the same goal and works considerably better.

**`mag_ratio` recovers small glyphs, at a cost.** Raising EasyOCR's pre-detection magnification is what makes the difference between a clock reading `332` and `3:32` — the colon is small enough to be dropped entirely at the default. It costs time roughly in proportion (1.7s → 5.8s on a 1280px frame), so the live loop leaves it at 1.0 and only the one-off **Get OCR** / **Load Image** paths raise it to 2.0. On the CLI it is `--mag-ratio`, which suits `--image` far more than the webcam.

**Restricting the character set is the single biggest win for known formats.** Unconstrained, a clock face read `3.32` (colon → period, 0.78). With `--allowlist '0123456789:'` it read `3:32` at **1.00**. Use it whenever you know the shape of the text — clocks, licence plates, meter readings, serial numbers.

**Punctuation is supported.** The English model's character set includes ``!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`` and it does read them (`POND'S: $5.99 (50% off!)` comes back essentially intact on a clean frame). If punctuation seems to go missing the causes are usually one of the above — a mark too small to survive the default `mag_ratio`, or an unconstrained charset picking a similar-looking character — plus the confidence threshold, since punctuated strings tend to score lower overall (try `--min-confidence 0.25`). For other scripts, pass the language: `--lang en ur`, `--lang ch_sim`, etc.

**Per-frame OCR disagrees with itself, so the live view is smoothed.** Each frame is recognised independently, so a stationary clock produced a different reading roughly every second — `822`, `8/22`, `8122`, `8*22`, `8 22` — none of them wrong exactly, just inconsistent. The live view now aggregates detections over a ~2 second window, groups them by position on screen, and shows the reading that wins a confidence-weighted vote. Steady text reads steadily, and one-frame garbage drops out for lack of support. `--no-stabilize` turns it off for raw, instant, noisier output.

## Testing

The preprocessing, downscaling, and threading logic are covered by unit tests that don't need a webcam or GPU:

```bash
pip install -r requirements-dev.txt
pytest -v
```

CI runs this same suite on every push (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## Project structure

```
realtime_ocr.py     CLI entry point: webcam/image loop, camera setup, downscaling, drawing
gui_app.py          Tkinter desktop UI on top of the same pipeline
gui_helpers.py      Pure GUI logic (log de-duplication, display scaling) kept testable
ocr_engine.py       EasyOCR wrapper (detector + recognizer), charset and magnification
ocr_worker.py       Background thread that decouples OCR speed from display FPS
stabilizer.py       Temporal voting so live readings stay steady between frames
preprocessing.py    Frame quality: lighting correction, sharpness scoring, frame selection
tests/              Unit tests for the above
```

## Tech Stack

Python, EasyOCR, PyTorch, OpenCV, Tkinter, ttkbootstrap, Pillow, NumPy

## License

MIT — see [LICENSE](LICENSE).
