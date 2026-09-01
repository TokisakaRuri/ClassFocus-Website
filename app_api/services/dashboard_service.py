from __future__ import annotations

from pathlib import Path
from typing import Any

from app_api.core.config import current_model_config_path, current_model_name, current_model_path
from app_api.db import crud
from app_api.db.database import now_iso
from app_api.services.agent_service import generate_multi_agent_analysis, generate_quality_report
from app_api.services.statistics_service import BEHAVIOR_LABELS, derive_overall_evidence, derive_segment_evidence, detect_warnings


SCHEMA_VERSION = 3
_cache_key: tuple[Any, ...] | None = None
_cache_payload: dict[str, Any] | None = None


def _result_stamp(task: dict[str, Any]) -> tuple[Any, ...]:
    path_value = task.get("result_path")
    if not path_value:
        return task.get("id"), task.get("status"), task.get("progress"), None
    path = Path(str(path_value))
    try:
        modified = path.stat().st_mtime_ns
    except OSError:
        modified = None
    return task.get("id"), task.get("status"), task.get("progress"), modified


def _file_stamp(path: Path) -> tuple[str, int | None]:
    try:
        return str(path), path.stat().st_mtime_ns
    except OSError:
        return str(path), None


def _warning_texts(warnings: Any) -> list[str]:
    if not isinstance(warnings, list):
        return []
    result: list[str] = []
    for warning in warnings:
        if isinstance(warning, dict):
            detail = warning.get("detail") or warning.get("type")
            if detail:
                result.append(str(detail))
        elif warning:
            result.append(str(warning))
    return result


def _segment_rows(segments: Any) -> list[dict[str, Any]]:
    if not isinstance(segments, list):
        return []
    rows: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        evidence = derive_segment_evidence(segment)
        rows.append({
            "label": str(segment.get("time_range") or ""),
            "start": float(segment.get("start_time") or 0),
            "total": evidence["total_count"],
            "classCount": evidence["class_count"],
            "dominantBehavior": BEHAVIOR_LABELS.get(evidence["dominant_behavior"], "暂无"),
            "dominantBehaviorKey": evidence["dominant_behavior"],
            "dominantBehaviorRate": evidence["dominant_behavior_rate"],
            "reviewCueCount": evidence["review_cue_count"],
            "reviewCueRate": evidence["review_cue_rate"],
            "reviewPriority": evidence["review_priority"],
            "reviewReason": evidence["review_reason"],
            "requiresReview": evidence["requires_review"],
            "distribution": evidence["behavior_distribution"],
            "counts": evidence["raw_counts"],
        })
    return rows


def _task_row(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(task.get("id") or ""),
        "displayId": str(task.get("display_id") or ""),
        "course": str(task.get("course_name") or "未命名课程"),
        "video": str(task.get("video_name") or "-"),
        "status": str(task.get("status") or "unknown"),
        "progress": round(float(task.get("progress") or 0), 2),
        "mode": str(task.get("analysis_mode") or "pending").upper(),
        "createdAt": str(task.get("created_at") or "")[:10],
        "startedAt": str(task.get("start_time") or ""),
        "endedAt": str(task.get("end_time") or ""),
        "errorMessage": str(task.get("error_message") or ""),
        "workerId": str(task.get("worker_id") or ""),
        "heartbeatAt": str(task.get("heartbeat_at") or ""),
        "attemptCount": int(task.get("attempt_count") or 0),
    }


def _report_row(task: dict[str, Any]) -> dict[str, Any] | None:
    payload = crud.load_result_json(task.get("result_path")) or {}
    overall = payload.get("overall") or {}
    if not overall:
        return None

    evidence = derive_overall_evidence(overall)
    segment_rows = _segment_rows(payload.get("segments"))
    review_segment_count = sum(1 for row in segment_rows if row["requiresReview"])
    stored_report = crud.get_report_by_task(int(task["id"])) or {}
    segments = payload.get("segments") or []
    warnings = detect_warnings(overall, segments)
    evidence_report = generate_quality_report(overall, segments, course=task, warnings=warnings)
    multi_agent = generate_multi_agent_analysis(overall, segments, course=task, warnings=warnings)
    video_info = payload.get("video_info") or {}
    recommendations = evidence_report.get("recommendations") or []
    suggestion = stored_report.get("ai_suggestion") or ""
    if not suggestion and recommendations:
        suggestion = " ".join(
            str(item.get("content") or "") for item in recommendations[:2] if isinstance(item, dict)
        )

    return {
        "id": int(task["id"]),
        "course": str(task.get("course_name") or "未命名课程"),
        "teacher": str(task.get("teacher_name") or ""),
        "className": str(task.get("class_name") or ""),
        "classroom": str(task.get("classroom") or ""),
        "lessonDate": str(task.get("lesson_date") or task.get("created_at") or "")[:10],
        "lessonSection": str(task.get("lesson_section") or ""),
        "videoName": str(task.get("video_name") or ""),
        "duration": round(float(video_info.get("duration") or 0), 2),
        "totalCount": evidence["total_count"],
        "classCount": evidence["class_count"],
        "dominantBehavior": BEHAVIOR_LABELS.get(evidence["dominant_behavior"], "暂无"),
        "dominantBehaviorKey": evidence["dominant_behavior"],
        "dominantBehaviorRate": evidence["dominant_behavior_rate"],
        "reviewCueCount": evidence["review_cue_count"],
        "reviewCueRate": evidence["review_cue_rate"],
        "reviewSegmentCount": review_segment_count,
        "evidenceCompleteness": round(float(evidence_report.get("evidence_completeness") or 0), 2),
        "evidenceStatus": evidence["evidence_status"],
        "reviewFocus": str(stored_report.get("main_problem") or f"待复核片段 {review_segment_count} 个"),
        "distribution": evidence["behavior_distribution"],
        "counts": evidence["raw_counts"],
        "dimensions": evidence_report.get("dimension_scores") or [],
        "segments": segment_rows,
        "warnings": _warning_texts(warnings),
        "agents": multi_agent.get("agents") or [],
        "consensus": str(multi_agent.get("consensus") or stored_report.get("ai_summary") or ""),
        "suggestion": str(suggestion),
    }


def build_dashboard_payload(limit: int = 50, *, use_cache: bool = True) -> dict[str, Any]:
    global _cache_key, _cache_payload

    tasks = crud.list_tasks(limit=limit, real_only=True)
    model_path = current_model_path()
    config_path = current_model_config_path()
    cache_key = (
        tuple(_result_stamp(task) for task in tasks),
        crud.get_video_collection_stamp(),
        _file_stamp(model_path),
        _file_stamp(config_path),
    )
    if use_cache and _cache_key == cache_key and _cache_payload is not None:
        return _cache_payload

    reports = [row for task in tasks if (row := _report_row(task)) is not None]
    raw_summary = crud.get_summary()
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": now_iso(),
        "summary": {
            "videoCount": int(raw_summary.get("video_count") or 0),
            "taskCount": int(raw_summary.get("task_count") or 0),
            "completedCount": int(raw_summary.get("completed_count") or 0),
            "reportCount": len(reports),
            "reviewSegmentCount": sum(int(report["reviewSegmentCount"]) for report in reports),
        },
        "reports": reports,
        "tasks": [_task_row(task) for task in tasks],
        "model": {
            "name": current_model_name(),
            "family": "DEIM" if model_path.suffix.lower() == ".pth" else "YOLO",
            "weight": model_path.name,
            "config": config_path.name,
            "ready": model_path.is_file() and config_path.is_file(),
        },
    }
    _cache_key = cache_key
    _cache_payload = payload
    return payload
