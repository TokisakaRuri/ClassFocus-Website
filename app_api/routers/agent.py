from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app_api.db import crud
from app_api.schemas.task_schema import AgentReportRequest
from app_api.services.agent_service import (
    generate_multi_agent_analysis,
    generate_quality_report,
    generate_rule_based_report,
)
from app_api.services.llm_service import assess_frame_occlusion, generate_llm_diagnosis, get_llm_status
from app_api.services.statistics_service import detect_warnings


router = APIRouter()


class OcclusionReviewRequest(BaseModel):
    task_id: int | None = None
    frame_id: int | None = None
    timestamp: float | None = None
    image_size: dict[str, float] = Field(default_factory=dict)
    image_mime: str = "image/jpeg"
    image_base64: str = ""
    detail_image_mime: str = "image/jpeg"
    detail_image_base64: str = ""
    detail_image_batches: list[dict[str, Any]] = Field(default_factory=list)
    detections: list[dict[str, Any]] = Field(default_factory=list)


class TaskAnalysisRequest(BaseModel):
    teaching_context: str = Field(default="", max_length=2000)


@router.get("/status")
def get_agent_status():
    return get_llm_status()


@router.post("/generate")
def generate_agent_report(req: AgentReportRequest):
    warnings = detect_warnings(req.overall, req.segments)
    return _generate_sections(req.overall, req.segments, req.course, warnings)


@router.post("/occlusion")
def review_frame_occlusion(req: OcclusionReviewRequest):
    return assess_frame_occlusion(req.model_dump())


@router.get("/task/{task_id}")
def get_agent_report(task_id: int):
    return _build_task_report(task_id, "")


@router.post("/task/{task_id}")
def generate_contextual_task_report(task_id: int, req: TaskAnalysisRequest):
    return _build_task_report(task_id, req.teaching_context)


def _build_task_report(task_id: int, teaching_context: str) -> dict[str, Any]:
    task = crud.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    result = crud.load_result_json(task.get("result_path"))
    if not result:
        raise HTTPException(status_code=404, detail="任务尚未生成结果")
    overall = result.get("overall", {})
    segments = result.get("segments", [])
    warnings = detect_warnings(overall, segments)
    course = {**task, "teaching_context": teaching_context.strip()}
    return _generate_sections(overall, segments, course, warnings)


def _generate_sections(
    overall: dict[str, Any],
    segments: list[dict[str, Any]],
    course: dict[str, Any],
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "agent_report": generate_rule_based_report(overall, segments, course=course, warnings=warnings),
        "quality_report": generate_quality_report(overall, segments, course=course, warnings=warnings),
        "multi_agent": generate_multi_agent_analysis(overall, segments, course=course, warnings=warnings),
        "llm_diagnosis": generate_llm_diagnosis(overall, segments, course=course, warnings=warnings),
    }
