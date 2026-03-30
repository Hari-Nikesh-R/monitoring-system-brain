from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist
from config import MAX_DISAPPEARED_FRAMES, MAX_MATCH_DISTANCE


class CentroidTracker:
    """Simple centroid-based multi-object tracker.

    Assigns persistent IDs to detected objects across frames using
    nearest-neighbour matching on centroids.
    """

    def __init__(
        self,
        max_disappeared: int = MAX_DISAPPEARED_FRAMES,
        max_distance: float = MAX_MATCH_DISTANCE,
    ):
        self.next_id = 0
        self.objects: OrderedDict[int, np.ndarray] = OrderedDict()
        self.bboxes: OrderedDict[int, tuple] = OrderedDict()
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    # -- internal helpers -----------------------------------------------------

    def _register(self, centroid: np.ndarray, bbox: tuple):
        self.objects[self.next_id] = centroid
        self.bboxes[self.next_id] = bbox
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, oid: int):
        del self.objects[oid]
        del self.bboxes[oid]
        del self.disappeared[oid]

    # -- public ---------------------------------------------------------------

    def update(self, detections: list[tuple]) -> dict:
        """Accept detections [(x1,y1,x2,y2,conf,...)] and return
        {id: (centroid_array, bbox_tuple)} for currently-tracked objects.
        """
        # No detections → increment disappeared counters
        if len(detections) == 0:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self._snapshot()

        input_centroids = np.zeros((len(detections), 2), dtype="float")
        input_bboxes: list[tuple] = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det[:4]
            input_centroids[i] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            input_bboxes.append((x1, y1, x2, y2))

        # First frame – register everything
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self._register(input_centroids[i], input_bboxes[i])
            return self._snapshot()

        # Match existing tracks to new detections
        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        D = dist.cdist(object_centroids, input_centroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows: set[int] = set()
        used_cols: set[int] = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            oid = object_ids[row]
            self.objects[oid] = input_centroids[col]
            self.bboxes[oid] = input_bboxes[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing tracks
        for row in set(range(D.shape[0])) - used_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self._deregister(oid)

        # Register brand-new detections
        for col in set(range(D.shape[1])) - used_cols:
            self._register(input_centroids[col], input_bboxes[col])

        return self._snapshot()

    def _snapshot(self) -> dict:
        return {
            oid: (self.objects[oid].copy(), self.bboxes[oid])
            for oid in self.objects
        }
