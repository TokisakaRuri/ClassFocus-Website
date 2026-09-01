from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterator

from app_api.core.config import ensure_directories, settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS course (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_name TEXT,
    teacher_name TEXT,
    class_name TEXT,
    classroom TEXT,
    lesson_date TEXT,
    lesson_section TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS video (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id INTEGER,
    video_name TEXT,
    video_path TEXT,
    duration REAL,
    fps REAL,
    resolution TEXT,
    upload_time TEXT,
    analysis_status TEXT,
    FOREIGN KEY(course_id) REFERENCES course(id)
);

CREATE TABLE IF NOT EXISTS analysis_task (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER,
    model_path TEXT,
    model_config_path TEXT,
    model_repo_path TEXT,
    status TEXT,
    progress REAL,
    frame_interval REAL,
    confidence_threshold REAL,
    segment_seconds INTEGER,
    start_time TEXT,
    end_time TEXT,
    error_message TEXT,
    result_path TEXT,
    analysis_mode TEXT,
    worker_id TEXT,
    claimed_at TEXT,
    heartbeat_at TEXT,
    attempt_count INTEGER DEFAULT 0,
    cancel_requested INTEGER DEFAULT 0,
    created_at TEXT,
    FOREIGN KEY(video_id) REFERENCES video(id)
);

CREATE TABLE IF NOT EXISTS worker_heartbeat (
    worker_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL,
    status TEXT NOT NULL,
    current_task_id INTEGER,
    FOREIGN KEY(current_task_id) REFERENCES analysis_task(id)
);

CREATE TABLE IF NOT EXISTS detection_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER,
    frame_id INTEGER,
    timestamp REAL,
    label TEXT,
    confidence REAL,
    x1 REAL,
    y1 REAL,
    x2 REAL,
    y2 REAL,
    image_path TEXT,
    FOREIGN KEY(task_id) REFERENCES analysis_task(id)
);

CREATE TABLE IF NOT EXISTS segment_statistic (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER,
    start_time REAL,
    end_time REAL,
    listening_count INTEGER,
    writing_count INTEGER,
    reading_count INTEGER,
    using_phone_count INTEGER,
    bowing_head_count INTEGER,
    sleeping_count INTEGER,
    total_count INTEGER,
    attention_rate REAL,
    abnormal_rate REAL,
    risk_level TEXT,
    FOREIGN KEY(task_id) REFERENCES analysis_task(id)
);

CREATE TABLE IF NOT EXISTS behavior_report (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER,
    course_id INTEGER,
    attention_rate REAL,
    abnormal_rate REAL,
    main_problem TEXT,
    ai_summary TEXT,
    ai_suggestion TEXT,
    risk_level TEXT,
    report_path TEXT,
    created_at TEXT,
    FOREIGN KEY(task_id) REFERENCES analysis_task(id),
    FOREIGN KEY(course_id) REFERENCES course(id)
);

CREATE TABLE IF NOT EXISTS teacher_review (
    task_id INTEGER PRIMARY KEY,
    owner TEXT NOT NULL,
    due TEXT NOT NULL,
    actions TEXT NOT NULL,
    status TEXT NOT NULL,
    review_conclusion TEXT NOT NULL,
    context_notes TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(task_id) REFERENCES analysis_task(id) ON DELETE CASCADE
);

"""

MIGRATION_COLUMNS: dict[str, dict[str, str]] = {
    "analysis_task": {
        "model_config_path": "TEXT",
        "model_repo_path": "TEXT",
        "analysis_mode": "TEXT DEFAULT 'pending'",
        "worker_id": "TEXT",
        "claimed_at": "TEXT",
        "heartbeat_at": "TEXT",
        "attempt_count": "INTEGER DEFAULT 0",
        "cancel_requested": "INTEGER DEFAULT 0",
    },
}

_INIT_LOCK = threading.Lock()
_INITIALIZED_DATABASES: set[Path] = set()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_db() -> None:
    ensure_directories()
    database_path = settings.database_path.resolve()
    with _INIT_LOCK:
        with sqlite3.connect(database_path, timeout=30) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)
            _apply_migrations(connection)
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_analysis_task_video_id ON analysis_task(video_id);
                CREATE INDEX IF NOT EXISTS idx_analysis_task_status ON analysis_task(status);
                CREATE INDEX IF NOT EXISTS idx_analysis_task_waiting ON analysis_task(id) WHERE status = 'waiting';
                CREATE INDEX IF NOT EXISTS idx_analysis_task_worker ON analysis_task(worker_id, status);
                CREATE INDEX IF NOT EXISTS idx_detection_result_task_frame ON detection_result(task_id, frame_id);
                CREATE INDEX IF NOT EXISTS idx_segment_statistic_task_id ON segment_statistic(task_id);
                CREATE INDEX IF NOT EXISTS idx_behavior_report_task_id ON behavior_report(task_id);
                CREATE INDEX IF NOT EXISTS idx_teacher_review_updated_at ON teacher_review(updated_at);
                """
            )
            connection.execute("PRAGMA optimize")
            connection.commit()
        _INITIALIZED_DATABASES.add(database_path)


def _apply_migrations(connection: sqlite3.Connection) -> None:
    for table, columns in MIGRATION_COLUMNS.items():
        existing = {
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for column, definition in columns.items():
            if column not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def get_connection() -> sqlite3.Connection:
    database_path = settings.database_path.resolve()
    if database_path not in _INITIALIZED_DATABASES:
        init_db()
    connection = sqlite3.connect(database_path, timeout=30)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.row_factory = sqlite3.Row
    return connection


def reset_initialization_state() -> None:
    """Clear process-local initialization state for isolated tests."""
    with _INIT_LOCK:
        _INITIALIZED_DATABASES.clear()


def dict_from_row(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def rows_to_dicts(rows: Iterator[sqlite3.Row]) -> list[dict]:
    return [dict(row) for row in rows]


def relative_or_absolute(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return str(candidate.relative_to(settings.database_path.parent))
    except ValueError:
        return str(candidate)
