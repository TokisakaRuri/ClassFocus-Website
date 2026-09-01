from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app_api.core.config import current_model_config_path, current_model_repo_path, settings
from app_api.core.exceptions import TaskOwnershipLost
from app_api.db.database import dict_from_row, get_connection, now_iso, rows_to_dicts


LOGGER = logging.getLogger(__name__)
TASK_UPDATE_FIELDS = {
    "status",
    "progress",
    "start_time",
    "end_time",
    "error_message",
    "result_path",
    "analysis_mode",
    "worker_id",
    "claimed_at",
    "heartbeat_at",
    "attempt_count",
    "cancel_requested",
}


def create_course(data: dict[str, Any]) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO course (
                course_name, teacher_name, class_name, classroom,
                lesson_date, lesson_section, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("course_name", ""),
                data.get("teacher_name", ""),
                data.get("class_name", ""),
                data.get("classroom", ""),
                data.get("lesson_date", ""),
                data.get("lesson_section", ""),
                now_iso(),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def create_video(data: dict[str, Any]) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO video (
                course_id, video_name, video_path, duration, fps,
                resolution, upload_time, analysis_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("course_id"),
                data.get("video_name", ""),
                data.get("video_path", ""),
                data.get("duration", 0),
                data.get("fps", 0),
                data.get("resolution", ""),
                now_iso(),
                data.get("analysis_status", "uploaded"),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def create_course_with_video(course: dict[str, Any], video: dict[str, Any]) -> tuple[int, int]:
    with get_connection() as connection:
        course_cursor = connection.execute(
            """
            INSERT INTO course (
                course_name, teacher_name, class_name, classroom,
                lesson_date, lesson_section, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                course.get("course_name", ""),
                course.get("teacher_name", ""),
                course.get("class_name", ""),
                course.get("classroom", ""),
                course.get("lesson_date", ""),
                course.get("lesson_section", ""),
                now_iso(),
            ),
        )
        course_id = int(course_cursor.lastrowid)
        video_cursor = connection.execute(
            """
            INSERT INTO video (
                course_id, video_name, video_path, duration, fps,
                resolution, upload_time, analysis_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                course_id,
                video.get("video_name", ""),
                video.get("video_path", ""),
                video.get("duration", 0),
                video.get("fps", 0),
                video.get("resolution", ""),
                now_iso(),
                video.get("analysis_status", "uploaded"),
            ),
        )
        connection.commit()
        return course_id, int(video_cursor.lastrowid)


def _insert_task(connection: Any, data: dict[str, Any]) -> int:
    cursor = connection.execute(
        """
        INSERT INTO analysis_task (
            video_id, model_path, model_config_path, model_repo_path,
            status, progress, frame_interval,
            confidence_threshold, segment_seconds, start_time, end_time,
            error_message, result_path, analysis_mode, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data.get("video_id"),
            data.get("model_path", str(settings.model_path)),
            data.get("model_config_path", str(current_model_config_path())),
            data.get("model_repo_path", str(current_model_repo_path())),
            data.get("status", "waiting"),
            data.get("progress", 0),
            data.get("frame_interval", settings.frame_sample_seconds),
            data.get("confidence_threshold", settings.confidence_threshold),
            data.get("segment_seconds", settings.segment_seconds),
            data.get("start_time"),
            data.get("end_time"),
            data.get("error_message"),
            data.get("result_path"),
            data.get("analysis_mode", "pending"),
            now_iso(),
        ),
    )
    return int(cursor.lastrowid)


def create_task(data: dict[str, Any]) -> int:
    with get_connection() as connection:
        task_id = _insert_task(connection, data)
        connection.commit()
        return task_id


def create_task_if_no_active(data: dict[str, Any]) -> tuple[int, bool]:
    """Atomically reuse an identical waiting/running task instead of duplicating GPU work."""
    model_path = str(data.get("model_path", settings.model_path))
    model_config_path = str(data.get("model_config_path", current_model_config_path()))
    model_repo_path = str(data.get("model_repo_path", current_model_repo_path()))
    frame_interval = float(data.get("frame_interval", settings.frame_sample_seconds))
    confidence_threshold = float(data.get("confidence_threshold", settings.confidence_threshold))
    segment_seconds = int(data.get("segment_seconds", settings.segment_seconds))
    with get_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            """
            SELECT id
            FROM analysis_task
            WHERE video_id = ?
              AND model_path = ?
              AND model_config_path = ?
              AND model_repo_path = ?
              AND ABS(COALESCE(frame_interval, 0) - ?) < 0.000001
              AND ABS(COALESCE(confidence_threshold, 0) - ?) < 0.000001
              AND segment_seconds = ?
              AND status IN ('waiting', 'running')
              AND COALESCE(cancel_requested, 0) = 0
            ORDER BY id ASC
            LIMIT 1
            """,
            (
                data.get("video_id"),
                model_path,
                model_config_path,
                model_repo_path,
                frame_interval,
                confidence_threshold,
                segment_seconds,
            ),
        ).fetchone()
        if existing is not None:
            connection.commit()
            return int(existing["id"]), False
        task_id = _insert_task(connection, data)
        connection.commit()
        return task_id, True


def update_task(task_id: int, **fields: Any) -> bool:
    updates = [(key, value) for key, value in fields.items() if key in TASK_UPDATE_FIELDS]
    if not updates:
        return False

    sql = ", ".join(f"{key} = ?" for key, _ in updates)
    values = [value for _, value in updates]
    values.append(task_id)

    with get_connection() as connection:
        cursor = connection.execute(f"UPDATE analysis_task SET {sql} WHERE id = ?", values)
        connection.commit()
        return cursor.rowcount == 1


def update_task_owned(task_id: int, worker_id: str, **fields: Any) -> bool:
    updates = [(key, value) for key, value in fields.items() if key in TASK_UPDATE_FIELDS]
    if not updates:
        return False
    sql = ", ".join(f"{key} = ?" for key, _ in updates)
    values = [value for _, value in updates]
    values.extend((task_id, worker_id))
    with get_connection() as connection:
        cursor = connection.execute(
            f"UPDATE analysis_task SET {sql} WHERE id = ? AND worker_id = ? AND status = 'running'",
            values,
        )
        connection.commit()
        return cursor.rowcount == 1


def _require_task_ownership(connection: Any, task_id: int, worker_id: str | None) -> None:
    if worker_id is None:
        return
    owned = connection.execute(
        "SELECT 1 FROM analysis_task WHERE id = ? AND worker_id = ? AND status = 'running'",
        (task_id, worker_id),
    ).fetchone()
    if owned is None:
        raise TaskOwnershipLost(f"Worker {worker_id} 已失去任务 {task_id} 的租约")


def update_video_status(video_id: int | None, status: str) -> None:
    if video_id is None:
        return
    with get_connection() as connection:
        connection.execute(
            "UPDATE video SET analysis_status = ? WHERE id = ?",
            (status, video_id),
        )
        connection.commit()


def update_video_metadata(video_id: int | None, *, duration: float, fps: float, resolution: str) -> None:
    if video_id is None:
        return
    with get_connection() as connection:
        connection.execute(
            "UPDATE video SET duration = ?, fps = ?, resolution = ? WHERE id = ?",
            (duration, fps, resolution, video_id),
        )
        connection.commit()


def get_video(video_id: int) -> dict | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT video.*, course.course_name, course.teacher_name, course.class_name,
                   course.classroom, course.lesson_date, course.lesson_section
            FROM video
            LEFT JOIN course ON course.id = video.course_id
            WHERE video.id = ?
            """,
            (video_id,),
        ).fetchone()
        return dict_from_row(row)


def get_task(task_id: int) -> dict | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT analysis_task.*, video.video_name, video.video_path, video.course_id,
                   video.duration, video.fps, video.resolution,
                   course.course_name, course.teacher_name, course.class_name,
                   course.classroom, course.lesson_date, course.lesson_section
            FROM analysis_task
            LEFT JOIN video ON video.id = analysis_task.video_id
            LEFT JOIN course ON course.id = video.course_id
            WHERE analysis_task.id = ?
            """,
            (task_id,),
        ).fetchone()
        return dict_from_row(row)


def list_tasks(limit: int = 50, real_only: bool = False) -> list[dict]:
    where = ""
    if real_only:
        where = """
            WHERE analysis_task.video_id IS NOT NULL
              AND COALESCE(analysis_task.analysis_mode, '') != 'demo'
        """
    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT analysis_task.*, video.video_name, video.video_path, video.course_id,
                   video.duration, video.fps, video.resolution,
                   course.course_name, course.teacher_name, course.class_name,
                   course.classroom, course.lesson_date, course.lesson_section
            FROM analysis_task
            LEFT JOIN video ON video.id = analysis_task.video_id
            LEFT JOIN course ON course.id = video.course_id
            {where}
            ORDER BY analysis_task.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        items = rows_to_dicts(rows)
        for index, item in enumerate(items, start=1):
            item["display_id"] = index
        return items


def list_tasks_by_status(statuses: set[str]) -> list[dict[str, Any]]:
    if not statuses:
        return []
    placeholders = ",".join("?" for _ in statuses)
    with get_connection() as connection:
        rows = connection.execute(
            f"SELECT * FROM analysis_task WHERE status IN ({placeholders}) ORDER BY id ASC",
            tuple(sorted(statuses)),
        )
        return rows_to_dicts(rows)


def recover_expired_tasks(lease_seconds: int = 45) -> int:
    cutoff = (datetime.now() - timedelta(seconds=max(10, lease_seconds))).isoformat(timespec="seconds")
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE video
            SET analysis_status = 'waiting'
            WHERE id IN (
                SELECT video_id FROM analysis_task
                WHERE status = 'running'
                  AND (heartbeat_at IS NULL OR heartbeat_at < ?)
                  AND COALESCE(cancel_requested, 0) = 0
            )
            """,
            (cutoff,),
        )
        cursor = connection.execute(
            """
            UPDATE analysis_task
            SET status = 'waiting', progress = 0, worker_id = NULL,
                claimed_at = NULL, heartbeat_at = NULL,
                start_time = NULL, end_time = NULL,
                error_message = 'Worker 中断，任务已自动恢复到等待队列'
            WHERE status = 'running'
              AND (heartbeat_at IS NULL OR heartbeat_at < ?)
              AND COALESCE(cancel_requested, 0) = 0
            """,
            (cutoff,),
        )
        connection.execute(
            """
            UPDATE video
            SET analysis_status = 'canceled'
            WHERE id IN (
                SELECT video_id FROM analysis_task
                WHERE status IN ('waiting', 'running')
                  AND COALESCE(cancel_requested, 0) = 1
                  AND (status = 'waiting' OR heartbeat_at IS NULL OR heartbeat_at < ?)
            )
            """,
            (cutoff,),
        )
        connection.execute(
            """
            UPDATE analysis_task
            SET status = 'canceled', progress = 100, end_time = ?,
                worker_id = NULL, claimed_at = NULL, heartbeat_at = NULL
            WHERE status IN ('waiting', 'running')
              AND COALESCE(cancel_requested, 0) = 1
              AND (status = 'waiting' OR heartbeat_at IS NULL OR heartbeat_at < ?)
            """,
            (now_iso(), cutoff),
        )
        connection.commit()
        return int(cursor.rowcount or 0)


def claim_next_task(worker_id: str, lease_seconds: int = 45) -> dict[str, Any] | None:
    recover_expired_tasks(lease_seconds)
    claimed_at = now_iso()
    with get_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT id
            FROM analysis_task
            WHERE status = 'waiting' AND COALESCE(cancel_requested, 0) = 0
            ORDER BY id ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            connection.commit()
            return None
        task_id = int(row["id"])
        cursor = connection.execute(
            """
            UPDATE analysis_task
            SET status = 'running', progress = 1, worker_id = ?,
                claimed_at = ?, heartbeat_at = ?, start_time = ?, end_time = NULL,
                error_message = NULL, attempt_count = COALESCE(attempt_count, 0) + 1
            WHERE id = ? AND status = 'waiting' AND COALESCE(cancel_requested, 0) = 0
            """,
            (worker_id, claimed_at, claimed_at, claimed_at, task_id),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            return None
        connection.commit()
    task = get_task(task_id)
    if task and task.get("video_id") is not None:
        update_video_status(int(task["video_id"]), "running")
    return task


def heartbeat_task(task_id: int, worker_id: str) -> bool:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE analysis_task
            SET heartbeat_at = ?
            WHERE id = ? AND worker_id = ? AND status = 'running'
            """,
            (now_iso(), task_id, worker_id),
        )
        connection.commit()
        return cursor.rowcount == 1


def is_task_cancel_requested(task_id: int) -> bool:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT cancel_requested FROM analysis_task WHERE id = ?",
            (task_id,),
        ).fetchone()
        return bool(row and row["cancel_requested"])


def get_task_control_state(task_id: int) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT status, worker_id, cancel_requested FROM analysis_task WHERE id = ?",
            (task_id,),
        ).fetchone()
    return dict_from_row(row)


def request_task_cancellation(task_id: int) -> str | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT status, video_id FROM analysis_task WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        status = str(row["status"] or "")
        if status not in {"waiting", "running"}:
            return status
        if status == "waiting":
            connection.execute(
                """
                UPDATE analysis_task
                SET status = 'canceled', progress = 100, cancel_requested = 1,
                    end_time = ?, worker_id = NULL, claimed_at = NULL, heartbeat_at = NULL
                WHERE id = ?
                """,
                (now_iso(), task_id),
            )
        else:
            connection.execute(
                "UPDATE analysis_task SET cancel_requested = 1 WHERE id = ?",
                (task_id,),
            )
        connection.commit()
    if row["video_id"] is not None and status == "waiting":
        update_video_status(int(row["video_id"]), "canceled")
    return "canceling" if status == "running" else "canceled"


def heartbeat_worker(
    worker_id: str,
    *,
    started_at: str,
    status: str,
    current_task_id: int | None,
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO worker_heartbeat (worker_id, started_at, heartbeat_at, status, current_task_id)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
                heartbeat_at = excluded.heartbeat_at,
                status = excluded.status,
                current_task_id = excluded.current_task_id
            """,
            (worker_id, started_at, now_iso(), status, current_task_id),
        )
        connection.commit()


def get_worker_health(stale_seconds: int = 15) -> dict[str, Any]:
    cutoff = (datetime.now() - timedelta(seconds=max(5, stale_seconds))).isoformat(timespec="seconds")
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT worker_id, started_at, heartbeat_at, status, current_task_id
            FROM worker_heartbeat
            ORDER BY heartbeat_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return {"online": False, "status": "offline", "current_task_id": None}
    result = dict(row)
    result["online"] = bool(
        result.get("heartbeat_at")
        and str(result["heartbeat_at"]) >= cutoff
        and result.get("status") != "stopped"
    )
    if not result["online"]:
        result["status"] = "offline"
    return result


def _stored_file_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = settings.database_path.parent / candidate
    return candidate


def _safe_generated_path(path: Path) -> bool:
    try:
        resolved = path.resolve()
        root = settings.database_path.parent.resolve()
        uploads = (root / "uploads").resolve()
        return resolved.is_relative_to(uploads)
    except (OSError, ValueError):
        return False


def _restore_staged_paths(staged: list[tuple[Path, Path]]) -> None:
    for source, staged_path in reversed(staged):
        if not staged_path.exists():
            continue
        try:
            source.parent.mkdir(parents=True, exist_ok=True)
            staged_path.replace(source)
        except OSError:
            LOGGER.exception("Unable to restore staged task artifact %s", source)


def _stage_generated_paths(paths: set[Path | None], task_id: int) -> tuple[Path | None, list[tuple[Path, Path]]]:
    upload_root = (settings.database_path.parent / "uploads").resolve()
    existing: list[Path] = []
    for path in paths:
        if path is None or not path.exists() or not _safe_generated_path(path):
            continue
        resolved = path.resolve()
        if resolved not in existing:
            existing.append(resolved)
    if not existing:
        return None, []

    trash_dir = upload_root / ".trash" / f"task_{task_id}_{uuid.uuid4().hex}"
    staged: list[tuple[Path, Path]] = []
    try:
        for source in existing:
            relative = source.relative_to(upload_root)
            staged_path = trash_dir / relative
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            source.replace(staged_path)
            staged.append((source, staged_path))
    except OSError:
        _restore_staged_paths(staged)
        shutil.rmtree(trash_dir, ignore_errors=True)
        raise
    return trash_dir, staged


def delete_task(task_id: int) -> dict | None:
    task = get_task(task_id)
    if task is None:
        return None

    with get_connection() as connection:
        report_rows = connection.execute(
            "SELECT report_path FROM behavior_report WHERE task_id = ?",
            (task_id,),
        ).fetchall()
    paths = {_stored_file_path(task.get("result_path")), settings.frame_dir / str(task_id)}
    paths.update(_stored_file_path(row["report_path"]) for row in report_rows)
    trash_dir, staged = _stage_generated_paths(paths, task_id)

    with get_connection() as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM detection_result WHERE task_id = ?", (task_id,))
            connection.execute("DELETE FROM segment_statistic WHERE task_id = ?", (task_id,))
            connection.execute("DELETE FROM behavior_report WHERE task_id = ?", (task_id,))
            connection.execute("DELETE FROM teacher_review WHERE task_id = ?", (task_id,))
            connection.execute(
                "UPDATE worker_heartbeat SET current_task_id = NULL WHERE current_task_id = ?",
                (task_id,),
            )
            connection.execute("DELETE FROM analysis_task WHERE id = ?", (task_id,))
            remaining_tasks = connection.execute("SELECT COUNT(*) AS count FROM analysis_task").fetchone()["count"]
            if int(remaining_tasks or 0) == 0:
                connection.execute("DELETE FROM sqlite_sequence WHERE name = ?", ("analysis_task",))

            if task.get("video_id") is not None:
                remaining = connection.execute(
                    "SELECT COUNT(*) AS count FROM analysis_task WHERE video_id = ?",
                    (task["video_id"],),
                ).fetchone()["count"]
                if int(remaining or 0) == 0:
                    connection.execute(
                        "UPDATE video SET analysis_status = ? WHERE id = ?",
                        ("uploaded", task["video_id"]),
                    )
            connection.commit()
        except Exception:
            connection.rollback()
            _restore_staged_paths(staged)
            if trash_dir is not None:
                shutil.rmtree(trash_dir, ignore_errors=True)
            raise

    if trash_dir is not None:
        try:
            shutil.rmtree(trash_dir)
        except OSError:
            LOGGER.warning("Task %s deleted; staged artifacts remain for later cleanup at %s", task_id, trash_dir)
    return task


def list_videos(limit: int = 50) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT video.*, course.course_name, course.teacher_name, course.class_name,
                   course.classroom, course.lesson_date, course.lesson_section
            FROM video
            LEFT JOIN course ON course.id = video.course_id
            ORDER BY video.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return rows_to_dicts(rows)


def save_result_json(task_id: int, payload: dict[str, Any], *, artifact_key: str | None = None) -> Path:
    settings.result_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{artifact_key}" if artifact_key else ""
    result_path = settings.result_dir / f"task_{task_id}{suffix}.json"
    compact = dict(payload)
    detections = compact.pop("detections", [])
    compact["detection_count"] = int(compact.get("detection_count") or len(detections))
    _write_json_atomic(result_path, compact)
    return result_path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def load_result_json(path: str | None, *, include_detections: bool = False) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = settings.database_path.parent / candidate
    if not candidate.exists():
        return None
    try:
        with candidate.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not include_detections:
        detections = payload.pop("detections", [])
        if detections and "detection_count" not in payload:
            payload["detection_count"] = len(detections)
    return payload


def compact_legacy_result_files() -> dict[str, int]:
    """Remove duplicated detection arrays after confirming they are stored in SQLite."""
    compacted = 0
    reclaimed_bytes = 0
    for task in list_tasks(limit=10_000):
        path = _stored_file_path(task.get("result_path"))
        if path is None or not path.is_file():
            continue
        try:
            before = path.stat().st_size
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        detections = payload.get("detections") if isinstance(payload, dict) else None
        if not isinstance(detections, list) or not detections:
            continue
        with get_connection() as connection:
            stored_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM detection_result WHERE task_id = ?",
                    (task["id"],),
                ).fetchone()[0]
            )
        if stored_count != len(detections):
            continue
        payload.pop("detections", None)
        payload["detection_count"] = stored_count
        _write_json_atomic(path, payload)
        compacted += 1
        reclaimed_bytes += max(0, before - path.stat().st_size)
    return {"compacted": compacted, "reclaimed_bytes": reclaimed_bytes}


def replace_detections(task_id: int, detections: list[dict[str, Any]], *, worker_id: str | None = None) -> None:
    with get_connection() as connection:
        if worker_id is not None:
            connection.execute("BEGIN IMMEDIATE")
        _require_task_ownership(connection, task_id, worker_id)
        connection.execute("DELETE FROM detection_result WHERE task_id = ?", (task_id,))
        connection.executemany(
            """
            INSERT INTO detection_result (
                task_id, frame_id, timestamp, label, confidence,
                x1, y1, x2, y2, image_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    task_id,
                    det.get("frame_id"),
                    det.get("timestamp"),
                    det.get("label"),
                    det.get("confidence"),
                    det.get("x1"),
                    det.get("y1"),
                    det.get("x2"),
                    det.get("y2"),
                    det.get("image_path"),
                )
                for det in detections
            ],
        )
        connection.commit()


def list_frame_summaries(task_id: int) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                frame_id,
                MIN(timestamp) AS timestamp,
                MAX(image_path) AS image_path,
                COUNT(*) AS target_count,
                COUNT(DISTINCT label) AS class_count,
                AVG(confidence) AS average_confidence,
                SUM(CASE WHEN label IN ('using phone', 'sleeping') THEN 1 ELSE 0 END) AS review_cue_count
            FROM detection_result
            WHERE task_id = ?
            GROUP BY frame_id
            ORDER BY timestamp ASC, frame_id ASC
            """,
            (task_id,),
        )
        return rows_to_dicts(rows)


def list_detections(task_id: int) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT frame_id, timestamp, label, confidence, x1, y1, x2, y2, image_path
            FROM detection_result
            WHERE task_id = ?
            ORDER BY frame_id ASC, id ASC
            """,
            (task_id,),
        )
        return rows_to_dicts(rows)


def prune_task_frames(task_id: int, keep: int = 24) -> dict[str, int]:
    summaries = list_frame_summaries(task_id)
    ranked = sorted(
        summaries,
        key=lambda row: (
            int(row.get("class_count") or 0),
            int(row.get("target_count") or 0),
            float(row.get("average_confidence") or 0),
        ),
        reverse=True,
    )
    keep_ids = {int(row["frame_id"]) for row in ranked[: max(0, keep)]}
    with get_connection() as connection:
        stored_rows = connection.execute(
            """
            SELECT frame_id, MAX(image_path) AS image_path
            FROM detection_result
            WHERE task_id = ? AND COALESCE(image_path, '') != ''
            GROUP BY frame_id
            """,
            (task_id,),
        ).fetchall()
        remove_paths = [
            _stored_file_path(row["image_path"])
            for row in stored_rows
            if int(row["frame_id"]) not in keep_ids
        ]
        if keep_ids:
            placeholders = ",".join("?" for _ in keep_ids)
            connection.execute(
                f"UPDATE detection_result SET image_path = NULL WHERE task_id = ? AND frame_id NOT IN ({placeholders})",
                (task_id, *sorted(keep_ids)),
            )
        else:
            connection.execute(
                "UPDATE detection_result SET image_path = NULL WHERE task_id = ?",
                (task_id,),
            )
        connection.commit()

    removed = 0
    reclaimed = 0
    for path in remove_paths:
        if path is None or not path.is_file() or not _safe_generated_path(path):
            continue
        try:
            reclaimed += path.stat().st_size
            path.unlink()
            removed += 1
        except OSError:
            continue
    return {"removed": removed, "reclaimed_bytes": reclaimed}


def list_detections_for_frames(task_id: int, frame_ids: list[int]) -> list[dict[str, Any]]:
    if not frame_ids:
        return []
    placeholders = ",".join("?" for _ in frame_ids)
    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT id, frame_id, timestamp, label, confidence, x1, y1, x2, y2, image_path
            FROM detection_result
            WHERE task_id = ? AND frame_id IN ({placeholders})
            ORDER BY timestamp ASC, frame_id ASC, id ASC
            """,
            (task_id, *frame_ids),
        )
        return rows_to_dicts(rows)


def replace_segments(task_id: int, segments: list[dict[str, Any]], *, worker_id: str | None = None) -> None:
    with get_connection() as connection:
        if worker_id is not None:
            connection.execute("BEGIN IMMEDIATE")
        _require_task_ownership(connection, task_id, worker_id)
        connection.execute("DELETE FROM segment_statistic WHERE task_id = ?", (task_id,))
        connection.executemany(
            """
            INSERT INTO segment_statistic (
                task_id, start_time, end_time, listening_count, writing_count,
                reading_count, using_phone_count, bowing_head_count,
                sleeping_count, total_count, attention_rate, abnormal_rate, risk_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    task_id,
                    segment.get("start_time"),
                    segment.get("end_time"),
                    segment.get("listening_count", 0),
                    segment.get("writing_count", 0),
                    segment.get("reading_count", 0),
                    segment.get("using_phone_count", 0),
                    segment.get("bowing_head_count", 0),
                    segment.get("sleeping_count", 0),
                    segment.get("total_count", 0),
                    segment.get("attention_rate", 0),
                    segment.get("abnormal_rate", 0),
                    segment.get("risk_level", ""),
                )
                for segment in segments
            ],
        )
        connection.commit()


def upsert_report(data: dict[str, Any], *, worker_id: str | None = None) -> int:
    task_id = data["task_id"]
    with get_connection() as connection:
        if worker_id is not None:
            connection.execute("BEGIN IMMEDIATE")
        _require_task_ownership(connection, task_id, worker_id)
        existing = connection.execute(
            "SELECT id FROM behavior_report WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        values = (
            data.get("course_id"),
            data.get("attention_rate", 0),
            data.get("abnormal_rate", 0),
            data.get("main_problem", ""),
            data.get("ai_summary", ""),
            data.get("ai_suggestion", ""),
            data.get("risk_level", ""),
            data.get("report_path", ""),
            now_iso(),
        )

        if existing:
            connection.execute(
                """
                UPDATE behavior_report
                SET course_id = ?, attention_rate = ?, abnormal_rate = ?,
                    main_problem = ?, ai_summary = ?, ai_suggestion = ?,
                    risk_level = ?, report_path = ?, created_at = ?
                WHERE task_id = ?
                """,
                (*values, task_id),
            )
            report_id = int(existing["id"])
        else:
            cursor = connection.execute(
                """
                INSERT INTO behavior_report (
                    task_id, course_id, attention_rate, abnormal_rate,
                    main_problem, ai_summary, ai_suggestion, risk_level,
                    report_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (task_id, *values),
            )
            report_id = int(cursor.lastrowid)

        connection.commit()
        return report_id


def get_report_by_task(task_id: int) -> dict | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM behavior_report WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        return dict_from_row(row)


def get_teacher_review(task_id: int) -> dict | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM teacher_review WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        return dict_from_row(row)


def upsert_teacher_review(task_id: int, data: dict[str, Any]) -> dict:
    updated_at = now_iso()
    values = (
        str(data.get("owner") or "").strip(),
        str(data.get("due") or "").strip(),
        str(data.get("actions") or "").strip(),
        str(data.get("status") or "").strip(),
        str(data.get("review_conclusion") or "").strip(),
        str(data.get("context_notes") or "").strip(),
        updated_at,
    )
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO teacher_review (
                task_id, owner, due, actions, status,
                review_conclusion, context_notes, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                owner = excluded.owner,
                due = excluded.due,
                actions = excluded.actions,
                status = excluded.status,
                review_conclusion = excluded.review_conclusion,
                context_notes = excluded.context_notes,
                updated_at = excluded.updated_at
            """,
            (task_id, *values),
        )
        connection.commit()
    return get_teacher_review(task_id) or {"task_id": task_id, "updated_at": updated_at}


def get_summary() -> dict[str, Any]:
    with get_connection() as connection:
        video_count = connection.execute("SELECT COUNT(*) AS count FROM video").fetchone()["count"]
        completed_count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM analysis_task
            WHERE status = 'completed'
              AND video_id IS NOT NULL
              AND COALESCE(analysis_mode, '') != 'demo'
            """
        ).fetchone()["count"]
        task_count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM analysis_task
            WHERE video_id IS NOT NULL
              AND COALESCE(analysis_mode, '') != 'demo'
            """
        ).fetchone()["count"]
        latest_report = connection.execute(
            """
            SELECT AVG(attention_rate) AS avg_attention,
                   AVG(abnormal_rate) AS avg_abnormal,
                   SUM(CASE WHEN main_problem LIKE '%using phone%' THEN 1 ELSE 0 END) AS phone_alerts
            FROM behavior_report
            LEFT JOIN analysis_task ON analysis_task.id = behavior_report.task_id
            WHERE analysis_task.video_id IS NOT NULL
              AND COALESCE(analysis_task.analysis_mode, '') != 'demo'
            """
        ).fetchone()
        status_rows = connection.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM analysis_task
            WHERE video_id IS NOT NULL
              AND COALESCE(analysis_mode, '') != 'demo'
            GROUP BY status
            """
        )
        reports = connection.execute(
            """
            SELECT behavior_report.*, analysis_task.created_at, analysis_task.analysis_mode,
                   course.course_name, course.class_name
            FROM behavior_report
            LEFT JOIN analysis_task ON analysis_task.id = behavior_report.task_id
            LEFT JOIN course ON course.id = behavior_report.course_id
            WHERE analysis_task.video_id IS NOT NULL
              AND COALESCE(analysis_task.analysis_mode, '') != 'demo'
            ORDER BY behavior_report.id DESC
            LIMIT 12
            """
        )

        return {
            "video_count": video_count,
            "task_count": task_count,
            "completed_count": completed_count,
            "avg_attention": round(latest_report["avg_attention"] or 0, 2),
            "avg_abnormal": round(latest_report["avg_abnormal"] or 0, 2),
            "phone_alerts": int(latest_report["phone_alerts"] or 0),
            "status_counts": rows_to_dicts(status_rows),
            "recent_reports": rows_to_dicts(reports),
        }


def get_video_collection_stamp() -> tuple[int, int, str]:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count, COALESCE(MAX(id), 0) AS max_id, COALESCE(MAX(upload_time), '') AS latest FROM video"
        ).fetchone()
    return int(row["count"] or 0), int(row["max_id"] or 0), str(row["latest"] or "")
