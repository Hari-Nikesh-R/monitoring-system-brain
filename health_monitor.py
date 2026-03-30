import time
from collections import defaultdict

import numpy as np

from config import (
    IDLE_SPEED_THRESHOLD,
    IDLE_WARNING_SECONDS,
    IDLE_ALERT_SECONDS,
    ISOLATION_DISTANCE_PX,
    ABNORMAL_SPEED_THRESHOLD,
    ALERT_COOLDOWN_SECONDS,
    POSITION_HISTORY_SECONDS,
)


class HealthMonitor:
    """Analyse tracked-cow positions to infer movement status & health."""

    def __init__(self):
        self.history: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        self._last_alert_time: dict[int, float] = defaultdict(lambda: 0.0)
        self.statuses: dict[int, dict] = {}

    # -- public ---------------------------------------------------------------

    def update(self, tracked: dict) -> dict:
        """Consume tracker output {id: (centroid, bbox)} and return analysis.

        Returns {id: {movement_status, health_status, alerts, centroid, bbox}}.
        """
        now = time.time()
        results: dict[int, dict] = {}
        all_centroids: list[tuple[float, float]] = []

        for cow_id, (centroid, bbox) in tracked.items():
            cx, cy = float(centroid[0]), float(centroid[1])
            self.history[cow_id].append((cx, cy, now))
            all_centroids.append((cx, cy))

        self._prune_old_history(now)

        herd_centroid = np.mean(all_centroids, axis=0) if len(all_centroids) > 1 else None

        for cow_id, (centroid, bbox) in tracked.items():
            cx, cy = float(centroid[0]), float(centroid[1])
            movement, health, alerts = self._analyse_cow(cow_id, cx, cy, herd_centroid, now)

            self.statuses[cow_id] = {
                "movement_status": movement,
                "health_status": health,
            }
            results[cow_id] = {
                "movement_status": movement,
                "health_status": health,
                "alerts": alerts,
                "centroid": (cx, cy),
                "bbox": bbox,
            }

        self._cleanup_stale(tracked, now)
        return results

    def get_all_statuses(self) -> dict:
        return dict(self.statuses)

    # -- internals ------------------------------------------------------------

    def _analyse_cow(self, cow_id, cx, cy, herd_centroid, now):
        movement = "Active"
        health = "Healthy"
        alerts: list[str] = []

        speed, idle_duration = self._movement_metrics(cow_id, now)

        if speed is not None:
            if speed < IDLE_SPEED_THRESHOLD:
                if idle_duration >= IDLE_ALERT_SECONDS:
                    movement = "Idle"
                    health = "Alert"
                    alerts.append(
                        f"Cow {cow_id} inactive for {int(idle_duration)}s"
                    )
                elif idle_duration >= IDLE_WARNING_SECONDS:
                    movement = "Idle"
                    health = "Warning"

            if speed > ABNORMAL_SPEED_THRESHOLD:
                health = "Alert"
                alerts.append(
                    f"Cow {cow_id} moving abnormally fast ({speed:.0f} px/s)"
                )

        if herd_centroid is not None:
            d = np.sqrt((cx - herd_centroid[0]) ** 2 + (cy - herd_centroid[1]) ** 2)
            if d > ISOLATION_DISTANCE_PX:
                health = "Alert"
                alerts.append(f"Cow {cow_id} is isolated from herd")

        alerts = self._apply_cooldown(cow_id, alerts, now)
        return movement, health, alerts

    def _movement_metrics(self, cow_id, now):
        """Return (speed_px_per_sec, idle_duration_sec) or (None, 0)."""
        pts = self.history.get(cow_id, [])
        if len(pts) < 2:
            return None, 0.0

        window = [p for p in pts if p[2] > now - IDLE_ALERT_SECONDS]
        if len(window) < 2:
            return None, 0.0

        total_dist = sum(
            np.sqrt((window[i][0] - window[i - 1][0]) ** 2 +
                     (window[i][1] - window[i - 1][1]) ** 2)
            for i in range(1, len(window))
        )
        elapsed = window[-1][2] - window[0][2]
        if elapsed <= 0:
            return 0.0, 0.0

        speed = total_dist / elapsed

        idle_dur = 0.0
        if speed < IDLE_SPEED_THRESHOLD:
            idle_dur = elapsed

        return speed, idle_dur

    def _apply_cooldown(self, cow_id, alerts, now):
        if not alerts:
            return alerts
        if now - self._last_alert_time[cow_id] < ALERT_COOLDOWN_SECONDS:
            return []
        self._last_alert_time[cow_id] = now
        return alerts

    def _prune_old_history(self, now):
        cutoff = now - POSITION_HISTORY_SECONDS
        for cow_id in list(self.history):
            self.history[cow_id] = [
                p for p in self.history[cow_id] if p[2] > cutoff
            ]

    def _cleanup_stale(self, tracked, now):
        tracked_ids = set(tracked)
        for cow_id in list(self.history):
            if cow_id not in tracked_ids:
                pts = self.history[cow_id]
                if pts and now - pts[-1][2] > POSITION_HISTORY_SECONDS:
                    del self.history[cow_id]
