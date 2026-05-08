from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class AbnormalityHit:
    """One abnormality prediction on a cow crop (crop coordinates)."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    label: str


class AbnormalityDetector:
    """Runs a custom YOLO model on cow crops.

    Supports:
    - detect: returns bounding boxes (wound/lesion/weak labels, etc.)
    - classify: returns top class + probability (healthy/unhealthy)
    """

    def __init__(self, model_path: str, confidence: float = 0.35):
        self.model = YOLO(model_path)
        self.confidence = confidence

    @property
    def task(self) -> str | None:
        return getattr(self.model, "task", None)

    def detect(self, crop_bgr: np.ndarray) -> list[AbnormalityHit]:
        """Return abnormality detections (detect task) on a BGR crop."""
        if crop_bgr is None or crop_bgr.size == 0:
            return []

        results = self.model(crop_bgr, conf=self.confidence, verbose=False)
        if not results:
            return []

        r0 = results[0]
        if getattr(r0, "boxes", None) is None:
            return []

        hits: list[AbnormalityHit] = []
        for box in r0.boxes:
            cls_id = int(box.cls[0])
            label = str(self.model.names.get(cls_id, cls_id))
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            hits.append(
                AbnormalityHit(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    conf=conf,
                    label=label,
                )
            )
        return hits

    def classify(self, crop_bgr: np.ndarray) -> tuple[str, float] | None:
        """Return (label, probability) for classification models, else None."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        results = self.model(crop_bgr, verbose=False)
        if not results:
            return None

        r0 = results[0]
        probs = getattr(r0, "probs", None)
        if probs is None:
            return None

        # ultralytics Probs exposes .top1 and .top1conf
        top1 = getattr(probs, "top1", None)
        top1conf = getattr(probs, "top1conf", None)
        if top1 is None or top1conf is None:
            return None

        label = str(self.model.names.get(int(top1), str(top1)))
        conf = float(top1conf)
        return label, conf


def _label_matches(label: str, keywords: Iterable[str]) -> bool:
    l = label.strip().lower()
    return any(k.lower() in l for k in keywords)


def summarize_hits(
    hits: list[AbnormalityHit],
    wound_keywords: Iterable[str],
    weak_keywords: Iterable[str],
) -> dict:
    """Return a compact summary used by the pipeline/UI."""
    wound = [h for h in hits if _label_matches(h.label, wound_keywords)]
    weak = [h for h in hits if _label_matches(h.label, weak_keywords)]
    other = [h for h in hits if h not in wound and h not in weak]
    return {
        "wound_hits": wound,
        "weak_hits": weak,
        "other_hits": other,
    }

