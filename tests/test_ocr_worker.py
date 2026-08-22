import time

import numpy as np

from ocr_worker import OCRWorker


def test_worker_returns_empty_results_before_any_frame_processed():
    worker = OCRWorker(process_frame=lambda frame: [])
    assert worker.latest_results() == []


def test_worker_processes_submitted_frame_and_publishes_results():
    def fake_process(frame):
        return [(np.zeros((4, 2)), f"value-{int(frame[0, 0, 0])}", 1.0)]

    worker = OCRWorker(fake_process)
    worker.start()
    try:
        worker.submit(np.full((2, 2, 3), 7, dtype=np.uint8))
        deadline = time.time() + 2
        while worker.latest_results() == [] and time.time() < deadline:
            time.sleep(0.01)

        results = worker.latest_results()
        assert len(results) == 1
        assert results[0][1] == "value-7"
    finally:
        worker.stop()


def test_worker_drops_stale_frames_under_slow_processing():
    call_count = {"n": 0}

    def slow_process(frame):
        call_count["n"] += 1
        time.sleep(0.1)
        return [(np.zeros((4, 2)), f"frame-{int(frame[0, 0, 0])}", 1.0)]

    worker = OCRWorker(slow_process)
    worker.start()
    try:
        for i in range(50):
            worker.submit(np.full((2, 2, 3), i % 256, dtype=np.uint8))
            time.sleep(0.01)  # simulate a much faster capture loop

        time.sleep(0.2)  # let the worker finish whatever it's mid-processing
        assert call_count["n"] < 50  # far fewer OCR calls than frames submitted
    finally:
        worker.stop()
