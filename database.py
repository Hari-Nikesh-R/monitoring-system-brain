import sqlite3
import threading
from config import DATABASE_PATH


class Database:
    """Thread-safe SQLite wrapper for cattle monitoring data."""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._local = threading.local()
        self._init_tables()

    # -- connection handling --------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_tables(self):
        conn = self._conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cow_status (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                cow_id          INTEGER NOT NULL,
                x               REAL,
                y               REAL,
                movement_status TEXT,
                health_status   TEXT,
                timestamp       DATETIME DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                cow_id      INTEGER NOT NULL,
                alert_type  TEXT,
                message     TEXT,
                timestamp   DATETIME DEFAULT (datetime('now','localtime'))
            );
            """
        )
        conn.commit()

    # -- writes ---------------------------------------------------------------

    def log_status(self, cow_id: int, x: float, y: float, movement: str, health: str):
        conn = self._conn()
        conn.execute(
            "INSERT INTO cow_status (cow_id, x, y, movement_status, health_status) "
            "VALUES (?, ?, ?, ?, ?)",
            (cow_id, x, y, movement, health),
        )
        conn.commit()

    def log_alert(self, cow_id: int, alert_type: str, message: str):
        conn = self._conn()
        conn.execute(
            "INSERT INTO alerts (cow_id, alert_type, message) VALUES (?, ?, ?)",
            (cow_id, alert_type, message),
        )
        conn.commit()

    # -- reads ----------------------------------------------------------------

    def get_latest_cow_statuses(self) -> list[dict]:
        cur = self._conn().execute(
            """
            SELECT cow_id, x, y, movement_status, health_status,
                   MAX(timestamp) AS timestamp
            FROM cow_status
            GROUP BY cow_id
            ORDER BY cow_id
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def get_recent_alerts(self, limit: int = 30) -> list[dict]:
        cur = self._conn().execute(
            "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in cur.fetchall()]

    def clear_all(self):
        conn = self._conn()
        conn.execute("DELETE FROM cow_status")
        conn.execute("DELETE FROM alerts")
        conn.commit()
