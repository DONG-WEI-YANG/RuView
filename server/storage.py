"""SQLite persistent storage for poses, vitals, and fall alerts."""
from __future__ import annotations

import json
import sqlite3
import time
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/wifi_body.db"


class Storage:
    """Lightweight SQLite store for historical data.

    Tables:
      - poses: timestamped 24-joint snapshots
      - vitals: breathing/HR/HRV readings
      - fall_alerts: fall detection events
      - calibration: node spatial calibration data
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("Storage opened: %s", db_path)

    def _create_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS poses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                joints_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS vitals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                breathing_bpm REAL,
                heart_bpm REAL,
                hrv_rmssd REAL,
                stress_index REAL,
                motion_intensity REAL,
                data_json TEXT
            );
            CREATE TABLE IF NOT EXISTS fall_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                confidence REAL,
                head_height REAL,
                velocity REAL,
                notified INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                profile_id TEXT,
                node_positions_json TEXT,
                reference_csi_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_poses_ts ON poses(ts);
            CREATE INDEX IF NOT EXISTS idx_vitals_ts ON vitals(ts);
            CREATE INDEX IF NOT EXISTS idx_fall_ts ON fall_alerts(ts);
        """)
        self.db.commit()

    # ── Poses ────────────────────────────────────────────────

    def save_pose(self, joints: np.ndarray):
        self.db.execute(
            "INSERT INTO poses (ts, joints_json) VALUES (?, ?)",
            (time.time(), json.dumps(joints.tolist())),
        )
        self.db.commit()

    def get_recent_poses(self, limit: int = 100) -> list[dict]:
        rows = self.db.execute(
            "SELECT ts, joints_json FROM poses ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"ts": r["ts"], "joints": json.loads(r["joints_json"])}
            for r in reversed(rows)
        ]

    # ── Vitals ───────────────────────────────────────────────

    def save_vitals(self, vitals: dict):
        self.db.execute(
            "INSERT INTO vitals (ts, breathing_bpm, heart_bpm, hrv_rmssd, "
            "stress_index, motion_intensity, data_json) VALUES (?,?,?,?,?,?,?)",
            (
                time.time(),
                vitals.get("breathing_bpm"),
                vitals.get("heart_bpm"),
                vitals.get("hrv_rmssd"),
                vitals.get("stress_index"),
                vitals.get("motion_intensity"),
                json.dumps(vitals),
            ),
        )
        self.db.commit()

    def get_recent_vitals(self, limit: int = 100) -> list[dict]:
        rows = self.db.execute(
            "SELECT ts, data_json FROM vitals ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"ts": r["ts"], **json.loads(r["data_json"])}
            for r in reversed(rows)
        ]

    # ── Fall Alerts ──────────────────────────────────────────

    def save_fall_alert(
        self, confidence: float, head_height: float, velocity: float
    ) -> int:
        cur = self.db.execute(
            "INSERT INTO fall_alerts (ts, confidence, head_height, velocity) "
            "VALUES (?,?,?,?)",
            (time.time(), confidence, head_height, velocity),
        )
        self.db.commit()
        return cur.lastrowid

    def mark_notified(self, alert_id: int):
        self.db.execute(
            "UPDATE fall_alerts SET notified = 1 WHERE id = ?", (alert_id,)
        )
        self.db.commit()

    def get_fall_alerts(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, ts, confidence, head_height, velocity, notified "
            "FROM fall_alerts ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_unnotified_alerts(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, ts, confidence, head_height, velocity "
            "FROM fall_alerts WHERE notified = 0 ORDER BY ts"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Calibration ──────────────────────────────────────────

    def save_calibration(
        self, profile_id: str, node_positions: dict, reference_csi: dict
    ):
        self.db.execute(
            "INSERT INTO calibration (ts, profile_id, node_positions_json, "
            "reference_csi_json) VALUES (?,?,?,?)",
            (
                time.time(),
                profile_id,
                json.dumps(node_positions),
                json.dumps(reference_csi),
            ),
        )
        self.db.commit()

    def get_latest_calibration(self, profile_id: str) -> dict | None:
        row = self.db.execute(
            "SELECT ts, profile_id, node_positions_json, reference_csi_json "
            "FROM calibration WHERE profile_id = ? ORDER BY ts DESC LIMIT 1",
            (profile_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "ts": row["ts"],
            "profile_id": row["profile_id"],
            "node_positions": json.loads(row["node_positions_json"]),
            "reference_csi": json.loads(row["reference_csi_json"]),
        }

    # ── Stats ────────────────────────────────────────────────

    def get_stats(self) -> dict:
        counts = {}
        for table in ("poses", "vitals", "fall_alerts", "calibration"):
            row = self.db.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()
            counts[table] = row["c"]
        return counts

    def close(self):
        self.db.close()
