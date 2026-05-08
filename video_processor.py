import time

import cv2
import numpy as np

from config import (
    ABNORMALITY_CONFIDENCE,
    ABNORMALITY_MIN_HITS,
    ABNORMALITY_MODEL,
    MAX_DISAPPEARED_FRAMES,
    MAX_MATCH_DISTANCE,
    DB_LOG_INTERVAL_FRAMES,
    MARK_UNHEALTHY_ON_WEAK,
    MARK_UNHEALTHY_ON_WOUND,
    UNHEALTHY_CLASS_KEYWORDS,
    UNHEALTHY_CLASS_THRESHOLD,
    WEAK_CLASS_KEYWORDS,
    WOUND_CLASS_KEYWORDS,
)
from database import Database
from detector import CowDetector
from health_monitor import HealthMonitor
from tracker import CentroidTracker
from abnormality_detector import AbnormalityDetector, summarize_hits


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
        self.abnormality = (
            AbnormalityDetector(ABNORMALITY_MODEL, confidence=ABNORMALITY_CONFIDENCE)
            if ABNORMALITY_MODEL
            else None
        )

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

        if self.abnormality is not None and health_results:
            self._apply_abnormality(frame, health_results)

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

            appearance = info.get("appearance_flags") or []
            extra = f" | {'/'.join(appearance)}" if appearance else ""
            label = f"Cow {cow_id} | {info['movement_status']} | {info['health_status']}{extra}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.circle(out, (cx, cy), 4, color, -1)

            for hit in info.get("abnormality_hits", []) or []:
                ax1, ay1, ax2, ay2 = (int(v) for v in hit["bbox"])
                cv2.rectangle(out, (ax1, ay1), (ax2, ay2), (255, 0, 255), 2)
                txt = f"{hit['label']} {hit['conf']:.2f}"
                cv2.putText(
                    out,
                    txt,
                    (ax1, max(0, ay1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

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

    # -- abnormality ----------------------------------------------------------

    def _apply_abnormality(self, frame: np.ndarray, results: dict):
        for cow_id, info in results.items():
            x1, y1, x2, y2 = (int(v) for v in info["bbox"])
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2c <= x1c or y2c <= y1c:
                continue

            crop = frame[y1c:y2c, x1c:x2c]
            # Classification model: decide unhealthy directly from label/prob
            if self.abnormality.task == "classify":
                out = self.abnormality.classify(crop)
                if out is None:
                    continue
                label, prob = out
                info["appearance_flags"] = [f"{label} {prob:.2f}"]

                is_unhealthy = (
                    prob >= UNHEALTHY_CLASS_THRESHOLD
                    and any(k.lower() in label.lower() for k in UNHEALTHY_CLASS_KEYWORDS)
                )
                if is_unhealthy:
                    info["health_status"] = "Alert"
                    info["alerts"].append(
                        f"Cow {cow_id} abnormality: {label} ({prob:.2f})"
                    )
                continue

            # Detection model: use keyword-matching on detected labels
            hits = self.abnormality.detect(crop)
            if not hits:
                continue

            summary = summarize_hits(hits, WOUND_CLASS_KEYWORDS, WEAK_CLASS_KEYWORDS)
            wound_hits = summary["wound_hits"]
            weak_hits = summary["weak_hits"]

            appearance_flags: list[str] = []
            if wound_hits:
                appearance_flags.append("Wound")
            if weak_hits:
                appearance_flags.append("Weak")

            mapped_hits: list[dict] = []
            for h in hits:
                mapped_hits.append(
                    {
                        "label": h.label,
                        "conf": h.conf,
                        "bbox": (
                            float(x1c + h.x1),
                            float(y1c + h.y1),
                            float(x1c + h.x2),
                            float(y1c + h.y2),
                        ),
                    }
                )

            info["appearance_flags"] = appearance_flags
            info["abnormality_hits"] = mapped_hits

            hits_count = len(wound_hits) + len(weak_hits)
            should_alert = hits_count >= ABNORMALITY_MIN_HITS and (
                (MARK_UNHEALTHY_ON_WOUND and len(wound_hits) > 0)
                or (MARK_UNHEALTHY_ON_WEAK and len(weak_hits) > 0)
            )
            if should_alert:
                info["health_status"] = "Alert"
                if wound_hits:
                    info["alerts"].append(f"Cow {cow_id} abnormality: wound detected")
                if weak_hits:
                    info["alerts"].append(
                        f"Cow {cow_id} abnormality: weak body condition detected"
                    )

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
