"""Temporal smoothing for live detections.

Each frame is OCR'd independently, so a stationary sign produces a different
reading every second or so -- a clock face cycling through "822", "8/22",
"8122", "8*22", "8 22". Nothing is wrong with any single pass; they just
disagree, and showing the newest one makes the display look unreliable.

This aggregates detections over a short time window instead: group them by
where they are on screen, and report the reading that wins a
confidence-weighted vote within each group. Steady text then reads steadily,
and one-off garbage detections drop out for lack of support.
"""

import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

import numpy as np

from ocr_engine import Detections

DEFAULT_WINDOW_SECONDS = 2.0
DEFAULT_MIN_VOTES = 2


def box_center(box: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(box[:, 0])), float(np.mean(box[:, 1]))


def box_size(box: np.ndarray) -> Tuple[float, float]:
    return (float(np.ptp(box[:, 0])), float(np.ptp(box[:, 1])))


class DetectionStabilizer:
    def __init__(self, window_seconds: float = DEFAULT_WINDOW_SECONDS,
                 min_votes: int = DEFAULT_MIN_VOTES):
        self.window_seconds = window_seconds
        self.min_votes = min_votes
        self._history: Deque[Tuple[float, Detections]] = deque()

    def update(self, detections: Detections, now: float = None) -> Detections:
        """Add this frame's detections and return the stabilized reading."""
        now = time.time() if now is None else now
        self._history.append((now, detections))

        cutoff = now - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        return self._consensus()

    def reset(self) -> None:
        self._history.clear()

    def _consensus(self) -> Detections:
        clusters: List[Dict] = []

        # Oldest first, so the newest box ends up as each cluster's position.
        for _, detections in self._history:
            for box, text, conf in detections:
                cluster = self._find_cluster(clusters, box)
                if cluster is None:
                    clusters.append({"box": box, "votes": [(text, conf)]})
                else:
                    cluster["box"] = box
                    cluster["votes"].append((text, conf))

        stabilized: Detections = []
        for cluster in clusters:
            if len(cluster["votes"]) < self.min_votes:
                continue  # too little support to trust -- likely a one-frame fluke

            # Weight each candidate by summed confidence, so a reading seen
            # often and confidently beats one seen often but hesitantly.
            weights: Dict[str, float] = defaultdict(float)
            counts: Dict[str, int] = defaultdict(int)
            for text, conf in cluster["votes"]:
                weights[text] += conf
                counts[text] += 1

            winner = max(weights, key=weights.get)
            mean_conf = weights[winner] / counts[winner]
            stabilized.append((cluster["box"], winner, mean_conf))

        return stabilized

    def _find_cluster(self, clusters: List[Dict], box: np.ndarray):
        """Match a box to an existing cluster by centre proximity, scaled to
        the box's own size so it works at any text size or camera distance."""
        cx, cy = box_center(box)
        w, h = box_size(box)
        tol_x = max(w * 0.5, 10.0)
        tol_y = max(h * 0.5, 10.0)

        for cluster in clusters:
            ox, oy = box_center(cluster["box"])
            if abs(cx - ox) <= tol_x and abs(cy - oy) <= tol_y:
                return cluster
        return None
