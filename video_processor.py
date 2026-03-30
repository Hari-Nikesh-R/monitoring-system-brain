import time

import cv2
import numpy as np

from config import (
    MAX_DISAPPEARED_FRAMES,
    MAX_MATCH_DISTANCE,
    DB_LOG_INTERVAL_FRAMES,
)
from database import Database
from detector import CowDetector
from health_monitor import HealthMonitor
from tracker import CentroidTracker


class VideoProcessor:
    """Orchestrates detection → tracking → health analysis → storage."""

    def __init__(
        self,
        detect_all: bool = False,
        confidence: float = 0.45,
        max_disappeared: int = MAX_DISAPPEARED_FRAMES,
        max_distance: float = MAX_MATCH_DISTANCE,
    ):
        self.detector = CowDetector(
            confidence=confidence, detect_all=detect_all
        )
        self.tracker = CentroidTracker(
            max_disappeared=max_disappeared, max_distance=max_distance
        )
        self.health = HealthMonitor()
        self.db = Database()

        self._total_frames = 0
        self._fps_counter = 0
        self._fps_ts = time.time()
        self.fps = 0.0

    # -- main entry -----------------------------------------------------------

    def process_frame(self, frame: np.ndarray):
        """Run the full pipeline on one BGR frame.

        Returns (annotated_frame, health_results_dict).
        """
        self._tick_fps()
        self._total_frames += 1

        detections, _ = self.detector.detect(frame)
        tracked = self.tracker.update(detections)
        health_results = self.health.update(tracked)

        self._persist(health_results)

        annotated = self._draw(frame, health_results)
        return annotated, health_results

    # -- drawing --------------------------------------------------------------

    def _draw(self, frame, results):
        out = frame.copy()
        for cow_id, info in results.items():
            x1, y1, x2, y2 = (int(v) for v in info["bbox"])
            cx, cy = int(info["centroid"][0]), int(info["centroid"][1])
            color = _status_color(info["health_status"])

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"Cow {cow_id} | {info['movement_status']} | {info['health_status']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.circle(out, (cx, cy), 4, color, -1)

        cv2.putText(out, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(out, f"Tracked: {len(results)}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        return out

    # -- persistence ----------------------------------------------------------

    def _persist(self, results):
        for cow_id, info in results.items():
            if self._total_frames % DB_LOG_INTERVAL_FRAMES == 0:
                self.db.log_status(
                    cow_id,
                    info["centroid"][0],
                    info["centroid"][1],
                    info["movement_status"],
                    info["health_status"],
                )
            for msg in info["alerts"]:
                self.db.log_alert(cow_id, info["health_status"], msg)

    # -- fps ------------------------------------------------------------------

    def _tick_fps(self):
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_ts
        if elapsed >= 1.0:
            self.fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_ts = now


def _status_color(status: str) -> tuple[int, int, int]:
    return {
        "Healthy": (0, 200, 0),
        "Warning": (0, 180, 255),
        "Alert": (0, 0, 255),
    }.get(status, (200, 200, 200))
