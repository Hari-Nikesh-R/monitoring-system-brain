from ultralytics import YOLO
from config import YOLO_MODEL, DETECTION_CONFIDENCE, COW_CLASS_NAME


class CowDetector:
    """Wrapper around YOLO for detecting cows (or all objects in demo mode)."""

    def __init__(
        self,
        model_path: str = YOLO_MODEL,
        confidence: float = DETECTION_CONFIDENCE,
        detect_all: bool = False,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.detect_all = detect_all

        self.cow_class_id: int | None = None
        for cls_id, name in self.model.names.items():
            if name == COW_CLASS_NAME:
                self.cow_class_id = cls_id
                break

    def detect(self, frame):
        """Return (detections, raw_result).

        detections – list of (x1, y1, x2, y2, confidence, class_name)
        raw_result – the first ultralytics Result for annotation reuse
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections: list[tuple] = []

        if not results:
            return detections, None

        result = results[0]
        for box in result.boxes:
            cls_id = int(box.cls[0])

            if not self.detect_all:
                if self.cow_class_id is not None and cls_id != self.cow_class_id:
                    continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            class_name = self.model.names.get(cls_id, str(cls_id))
            detections.append((x1, y1, x2, y2, conf, class_name))

        return detections, result
