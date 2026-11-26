"""Lightweight SQLite persistence for analyses and candidate records."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class ResultStore:
    """Persist analysis history, candidate records, and model status."""

    def __init__(self, database: str | Path = "data/exoplanet_discovery.db") -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    target_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cadence_count INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    target_id TEXT NOT NULL,
                    candidate_name TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    confirmed_exoplanet INTEGER NOT NULL,
                    probability REAL,
                    period_days REAL,
                    transit_depth_ppm REAL,
                    transit_duration_hours REAL,
                    snr REAL,
                    confidence REAL,
                    notes TEXT
                );
                """
            )

    def add_analysis(
        self,
        target_id: str,
        source: str,
        status: str,
        cadence_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO analyses
                (target_id, source, status, cadence_count, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (target_id, source, status, cadence_count, json.dumps(metadata or {})),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an analysis row identifier.")
            return int(cursor.lastrowid)

    def add_candidates(self, candidates: list[dict[str, Any]]) -> int:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO candidates
                (target_id, candidate_name, classification, confirmed_exoplanet,
                 probability, period_days, transit_depth_ppm,
                 transit_duration_hours, snr, confidence, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["target_id"],
                        item["candidate_name"],
                        item["classification"],
                        int(item["confirmed_exoplanet"]),
                        item.get("probability"),
                        item.get("period_days"),
                        item.get("transit_depth_ppm"),
                        item.get("transit_duration_hours"),
                        item.get("snr"),
                        item.get("confidence"),
                        item.get("notes"),
                    )
                    for item in candidates
                ],
            )
            return len(candidates)

    def analyses(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM analyses ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def candidates(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM candidates ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def candidate_csv(self, path: str | Path) -> Path:
        import pandas as pd

        frame = pd.DataFrame(self.candidates())
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
        return target
