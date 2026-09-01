from __future__ import annotations

import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

from app_api.core.config import ROOT_DIR, current_model_config_path, current_model_path, current_model_repo_path, settings
from app_api.core.exceptions import AnalysisCanceled, TaskOwnershipLost
from app_api.db import crud
from app_api.db.database import now_iso
from app_api.services.agent_service import generate_multi_agent_analysis, generate_quality_report, generate_rule_based_report
from app_api.services.report_service import export_report
from app_api.services.statistics_service import aggregate_by_segment, calculate_overall_statistics, detect_warnings
from app_api.services.yolo_service import ClassroomYOLOService


LOGGER = logging.getLogger(__name__)
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


class AnalysisInputError(ValueError):
    pass


class ModelServiceCache:
    """Keep one model resident in the dedicated worker process."""

    def __init__(self) -> None:
        self._key: tuple[str, str, str, str] | None = None
        self._service: ClassroomYOLOService | None = None

    def get(self, task: dict[str, Any]) -> ClassroomYOLOService:
        model_path = str(task.get("model_path") or current_model_path())
        config_path = str(task.get("model_config_path") or current_model_config_path())
        repo_path = str(task.get("model_repo_path") or current_model_repo_path())
        key = (model_path, config_path, repo_path, settings.device)
        if self._service is not None and self._key == key:
            return self._service

        self.clear()
        self._service = ClassroomYOLOService(
            model_path=model_path,
            config_path=config_path,
            repo_path=repo_path,
            device=settings.device,
        )
        self._key = key
        return self._service

    def clear(self) -> None:
        self._service = None
        self._key = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def resolve_video_path(path: str | None) -> Path:
    if not path:
        raise AnalysisInputError("缺少视频路径")
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    try:
        resolved = candidate.resolve(strict=True)
        upload_root = settings.upload_dir.resolve(strict=True)
    except OSError as exc:
        raise AnalysisInputError("视频文件不存在") from exc
    if not resolved.is_file():
        raise AnalysisInputError("视频文件不存在")
    if not resolved.is_relative_to(upload_root):
        raise AnalysisInputError("视频路径必须位于受管上传目录中")
    if resolved.suffix.lower() not in VIDEO_SUFFIXES:
        raise AnalysisInputError("不支持的视频格式")
    return resolved


def _clear_partial_frame_dir(frame_dir: Path) -> None:
    try:
        resolved = frame_dir.resolve()
        root = settings.frame_dir.resolve()
    except OSError:
        return
    if resolved.is_relative_to(root) and resolved != root and resolved.exists():
        shutil.rmtree(resolved)


def _raise_if_stopped(task_id: int, worker_id: str) -> None:
    state = crud.get_task_control_state(task_id)
    if state and state.get("cancel_requested"):
        raise AnalysisCanceled("任务已由用户取消")
    if not state or state.get("status") != "running" or state.get("worker_id") != worker_id:
        raise TaskOwnershipLost(f"Worker {worker_id} 已失去任务 {task_id} 的租约")


def _update_owned_or_raise(task_id: int, worker_id: str, **fields: Any) -> None:
    if not crud.update_task_owned(task_id, worker_id, **fields):
        raise TaskOwnershipLost(f"Worker {worker_id} 已失去任务 {task_id} 的租约")


def _run_artifact_key(task_id: int, task: dict[str, Any], worker_id: str) -> str:
    safe_worker = re.sub(r"[^A-Za-z0-9_-]+", "-", worker_id).strip("-")[:64] or "worker"
    attempt = max(1, int(task.get("attempt_count") or 1))
    return f"task-{task_id}-attempt-{attempt}-{safe_worker}"


def _cleanup_run_artifacts(frame_dir: Path, *files: Path | None) -> None:
    _clear_partial_frame_dir(frame_dir)
    for path in files:
        if path is None:
            continue
        try:
            resolved = path.resolve()
            root = settings.database_path.parent.resolve()
            if resolved.is_relative_to(root / "uploads") and resolved.is_file():
                resolved.unlink()
        except OSError:
            LOGGER.warning("Unable to clean partial analysis artifact %s", path)


def run_analysis(
    task_id: int,
    *,
    worker_id: str,
    model_cache: ModelServiceCache,
) -> None:
    task = crud.get_task(task_id)
    if not task or task.get("status") != "running" or task.get("worker_id") != worker_id:
        return
    video_id = task.get("video_id")
    video = crud.get_video(int(video_id)) if video_id is not None else None
    artifact_key = _run_artifact_key(task_id, task, worker_id)
    frame_dir = settings.frame_dir / str(task_id) / artifact_key
    report_path: Path | None = None
    result_path: Path | None = None

    try:
        _raise_if_stopped(task_id, worker_id)
        video_path = resolve_video_path(video["video_path"] if video else task.get("video_path"))
        _clear_partial_frame_dir(frame_dir)
        _update_owned_or_raise(task_id, worker_id, status="running", progress=5, error_message=None)
        crud.update_video_status(video_id, "running")

        last_saved_progress = -1.0
        last_saved_at = 0.0

        def progress_callback(value: float) -> None:
            nonlocal last_saved_progress, last_saved_at
            _raise_if_stopped(task_id, worker_id)
            progress = round(5 + min(value, 95) * 0.65, 2)
            now = time.monotonic()
            if progress - last_saved_progress < 1 and now - last_saved_at < 2:
                return
            _update_owned_or_raise(task_id, worker_id, progress=progress, heartbeat_at=now_iso())
            last_saved_progress = progress
            last_saved_at = now

        service = model_cache.get(task)
        _raise_if_stopped(task_id, worker_id)
        analysis = service.analyze_video(
            video_path=video_path,
            confidence_threshold=float(task.get("confidence_threshold") or settings.confidence_threshold),
            frame_sample_seconds=float(task.get("frame_interval") or settings.frame_sample_seconds),
            progress_callback=progress_callback,
        )
        _raise_if_stopped(task_id, worker_id)
        _update_owned_or_raise(
            task_id,
            worker_id,
            progress=75,
            analysis_mode=analysis.get("mode", "unknown"),
            heartbeat_at=now_iso(),
        )

        detections = analysis.get("detections", [])
        overall = calculate_overall_statistics(detections)
        segments = aggregate_by_segment(
            detections=detections,
            segment_seconds=int(task.get("segment_seconds") or settings.segment_seconds),
            duration=analysis.get("duration"),
        )
        if settings.save_key_frames and settings.max_key_frames > 0:
            service.save_key_frames(
                video_path=video_path,
                detections=detections,
                frame_dir=frame_dir,
                limit=settings.max_key_frames,
            )
        _raise_if_stopped(task_id, worker_id)
        warnings = detect_warnings(overall, segments)
        course = task or {}
        agent_report = generate_rule_based_report(overall, segments, course=course, warnings=warnings)
        quality_report = generate_quality_report(overall, segments, course=course, warnings=warnings)
        multi_agent = generate_multi_agent_analysis(overall, segments, course=course, warnings=warnings)

        video_info = {
            "fps": analysis.get("fps", 0),
            "total_frames": analysis.get("total_frames", 0),
            "duration": analysis.get("duration", 0),
            "resolution": analysis.get("resolution", ""),
            "mode": analysis.get("mode", ""),
            "message": analysis.get("message", ""),
            "video_path": str(video_path),
        }
        report_path = export_report(
            task_id=task_id,
            course=course,
            video_info=video_info,
            overall=overall,
            segments=segments,
            agent_report=agent_report,
            quality_report=quality_report,
            multi_agent=multi_agent,
            artifact_key=artifact_key,
        )
        payload = {
            "video_info": video_info,
            "overall": overall,
            "segments": segments,
            "warnings": warnings,
            "agent_report": agent_report,
            "quality_report": quality_report,
            "multi_agent": multi_agent,
            "detections": detections,
            "report_path": str(report_path),
        }
        _raise_if_stopped(task_id, worker_id)
        result_path = crud.save_result_json(task_id, payload, artifact_key=artifact_key)
        crud.replace_segments(task_id, segments, worker_id=worker_id)
        crud.replace_detections(task_id, detections, worker_id=worker_id)
        crud.upsert_report(
            {
                "task_id": task_id,
                "course_id": task.get("course_id") if task else None,
                "attention_rate": 0,
                "abnormal_rate": 0,
                "main_problem": agent_report.get("main_problem", ""),
                "ai_summary": agent_report.get("summary", ""),
                "ai_suggestion": agent_report.get("suggestion", ""),
                "risk_level": overall.get("evidence_status", "行为证据已生成"),
                "report_path": str(report_path),
            },
            worker_id=worker_id,
        )
        _update_owned_or_raise(
            task_id,
            worker_id,
            status="completed",
            progress=100,
            end_time=now_iso(),
            result_path=str(result_path),
            analysis_mode=analysis.get("mode", "unknown"),
            cancel_requested=0,
        )
        crud.update_video_metadata(
            video_id,
            duration=float(video_info["duration"] or 0),
            fps=float(video_info["fps"] or 0),
            resolution=str(video_info["resolution"] or ""),
        )
        crud.update_video_status(video_id, "completed")
    except AnalysisCanceled as exc:
        LOGGER.info("Analysis task %s canceled", task_id)
        _cleanup_run_artifacts(frame_dir, report_path, result_path)
        updated = crud.update_task_owned(
            task_id,
            worker_id,
            status="canceled",
            progress=100,
            end_time=now_iso(),
            error_message=str(exc),
            cancel_requested=1,
        )
        current = crud.get_task_control_state(task_id)
        if updated or (current and current.get("status") == "canceled"):
            crud.update_video_status(video_id, "canceled")
    except TaskOwnershipLost:
        LOGGER.warning("Worker %s stopped writing task %s after losing its lease", worker_id, task_id)
        _cleanup_run_artifacts(frame_dir, report_path, result_path)
    except Exception as exc:
        LOGGER.exception("Analysis failed for task %s", task_id)
        _cleanup_run_artifacts(frame_dir, report_path, result_path)
        updated = crud.update_task_owned(
            task_id,
            worker_id,
            status="failed",
            progress=100,
            end_time=now_iso(),
            error_message=str(getattr(exc, "detail", None) or exc)[:2000],
        )
        if updated:
            crud.update_video_status(video_id, "failed")
