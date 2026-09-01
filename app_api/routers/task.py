from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse

from app_api.core.config import (
    current_model_config_path,
    current_model_path,
    current_model_repo_path,
)
from app_api.db import crud
from app_api.schemas.task_schema import AnalyzeRequest, TeacherReviewRequest
from app_api.services.agent_service import (
    generate_multi_agent_analysis,
    generate_quality_report,
    generate_rule_based_report,
)
from app_api.services.analysis_service import AnalysisInputError, resolve_video_path
from app_api.services.dashboard_service import build_dashboard_payload
from app_api.services.frame_service import (
    build_frame_analysis,
    build_occlusion_payload,
    extract_clean_frame,
    get_frame_source,
    render_annotated_frame,
)
from app_api.services.llm_service import assess_frame_occlusion
from app_api.services.statistics_service import (
    detect_warnings,
)


router = APIRouter()


def _resolve_path(path: str | None) -> Path:
    try:
        return resolve_video_path(path)
    except AnalysisInputError as exc:
        status_code = 404 if str(exc) == "视频文件不存在" else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


def _slim_result(payload: dict[str, Any] | None, include_detections: bool = False) -> dict[str, Any] | None:
    if payload is None:
        return None
    if include_detections:
        return payload
    result = dict(payload)
    detections = result.pop("detections", None)
    if detections is not None:
        result["detection_count"] = len(detections)
    else:
        result.setdefault("detection_count", 0)
    return result


def _refresh_report_sections(payload: dict[str, Any] | None, task: dict[str, Any]) -> dict[str, Any] | None:
    if payload is None:
        return None
    result = dict(payload)
    overall = result.get("overall") or {}
    segments = result.get("segments") or []
    warnings = result.get("warnings") or detect_warnings(overall, segments)
    if overall:
        result["warnings"] = warnings
        result["agent_report"] = generate_rule_based_report(overall, segments, course=task, warnings=warnings)
        result["quality_report"] = generate_quality_report(overall, segments, course=task, warnings=warnings)
        result["multi_agent"] = generate_multi_agent_analysis(overall, segments, course=task, warnings=warnings)
    return result


def _task_response(task: dict[str, Any], include_detections: bool = False) -> dict[str, Any]:
    result = _refresh_report_sections(crud.load_result_json(task.get("result_path")), task)
    if include_detections and result is not None:
        result = {**result, "detections": crud.list_detections(int(task["id"]))}
    report = crud.get_report_by_task(int(task["id"]))
    return {
        **task,
        "result": _slim_result(result, include_detections=include_detections),
        "report": report,
    }


def _event_task_items(limit: int) -> list[dict[str, Any]]:
    return [
        {
            "id": str(item.get("id") or ""),
            "status": str(item.get("status") or "unknown"),
            "progress": round(float(item.get("progress") or 0), 2),
            "workerId": str(item.get("worker_id") or ""),
            "heartbeatAt": str(item.get("heartbeat_at") or ""),
        }
        for item in crud.list_tasks(limit=limit, real_only=True)
    ]


@router.get("/summary")
async def get_summary():
    return crud.get_summary()


@router.get("")
async def list_tasks(
    limit: int = Query(50, ge=1, le=200),
    real_only: bool = Query(False),
):
    return {"items": crud.list_tasks(limit=limit, real_only=real_only)}


@router.get("/dashboard")
async def get_dashboard(limit: int = Query(50, ge=1, le=100)):
    return build_dashboard_payload(limit=limit)


@router.get("/events")
async def stream_task_events(request: Request, limit: int = Query(50, ge=1, le=100)):
    async def event_stream():
        previous = ""
        keepalive = 0
        while not await request.is_disconnected():
            payload = {
                "tasks": _event_task_items(limit),
                "worker": crud.get_worker_health(),
            }
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if encoded != previous:
                yield f"event: tasks\ndata: {encoded}\n\n"
                previous = encoded
                keepalive = 0
            else:
                keepalive += 1
                if keepalive >= 10:
                    yield ": keepalive\n\n"
                    keepalive = 0
            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/analyze", status_code=202)
async def analyze_video(req: AnalyzeRequest):
    video = crud.get_video(req.video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="视频不存在")

    _resolve_path(video["video_path"])
    task_id, created = crud.create_task_if_no_active(
        {
            "video_id": req.video_id,
            "model_path": str(current_model_path()),
            "model_config_path": str(current_model_config_path()),
            "model_repo_path": str(current_model_repo_path()),
            "status": "waiting",
            "progress": 0,
            "frame_interval": req.frame_sample_seconds,
            "confidence_threshold": req.confidence_threshold,
            "segment_seconds": req.segment_seconds,
        }
    )
    if created:
        crud.update_video_status(req.video_id, "waiting")
    return {
        "message": "分析任务已进入持久队列" if created else "相同分析任务已在队列中",
        "task_id": task_id,
        "created": created,
    }


@router.post("/{task_id}/cancel", status_code=202)
async def cancel_task(task_id: int):
    status = crud.request_task_cancellation(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if status not in {"canceling", "canceled"}:
        raise HTTPException(status_code=409, detail=f"任务当前状态为 {status}，无法取消")
    return {"message": "取消请求已提交", "task_id": task_id, "status": status}


@router.get("/{task_id}/frames")
async def get_task_frames(task_id: int, limit: int = Query(8, ge=4, le=24)):
    task = crud.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return build_frame_analysis(task, limit=limit)


@router.get("/{task_id}/frames/{frame_id}/image")
async def get_task_frame_image(
    task_id: int,
    frame_id: int,
    variant: str = Query("annotated", pattern="^(annotated|clean|clean_strict)$"),
):
    task = crud.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    source = get_frame_source(task, frame_id)
    if source is None:
        raise HTTPException(status_code=404, detail="帧截图不存在")
    if variant in {"clean", "clean_strict"}:
        clean = extract_clean_frame(task, frame_id)
        if clean:
            return Response(content=clean, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
        if variant == "clean_strict":
            raise HTTPException(status_code=404, detail="无法从原视频提取干净帧")
    annotated = render_annotated_frame(task, frame_id)
    if annotated:
        return Response(content=annotated, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
    if source[0] is not None:
        return FileResponse(source[0], headers={"Cache-Control": "private, max-age=3600"})
    raise HTTPException(status_code=404, detail="帧截图不存在")


@router.post("/{task_id}/frames/{frame_id}/occlusion")
def review_task_frame_occlusion(task_id: int, frame_id: int):
    task = crud.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    payload = build_occlusion_payload(task, frame_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="帧截图或检测结果不存在")
    return assess_frame_occlusion(payload)


def _default_teacher_review(task: dict[str, Any]) -> dict[str, Any]:
    report = crud.get_report_by_task(int(task["id"])) or {}
    return {
        "task_id": int(task["id"]),
        "owner": str(task.get("teacher_name") or "任课教师"),
        "due": "下节课前",
        "actions": str(report.get("ai_suggestion") or ""),
        "status": "待提交",
        "review_conclusion": "尚未复核",
        "context_notes": "",
        "updated_at": "",
    }


@router.get("/{task_id}/review")
def get_teacher_review(task_id: int):
    task = crud.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return crud.get_teacher_review(task_id) or _default_teacher_review(task)


@router.post("/{task_id}/review")
def save_teacher_review(task_id: int, req: TeacherReviewRequest):
    task = crud.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return crud.upsert_teacher_review(task_id, req.model_dump())


@router.get("/{task_id}")
async def get_task(task_id: int, include_detections: bool = Query(False)):
    task = crud.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _task_response(task, include_detections=include_detections)


@router.delete("/{task_id}")
async def delete_task(task_id: int):
    task = crud.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.get("status") in {"waiting", "running", "canceling"}:
        raise HTTPException(status_code=409, detail="任务正在分析中，无法删除")

    try:
        deleted = crud.delete_task(task_id)
    except OSError as exc:
        raise HTTPException(status_code=409, detail="任务文件正在使用，未删除任何数据库记录；请关闭报告或稍后重试") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"message": "任务记录已删除", "task_id": task_id}
