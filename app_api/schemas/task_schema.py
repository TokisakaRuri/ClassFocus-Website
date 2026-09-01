from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    video_id: int = Field(gt=0)
    confidence_threshold: float = Field(default=0.5, ge=0.1, le=1.0)
    frame_sample_seconds: float = Field(default=1.0, ge=0.2, le=10.0)
    segment_seconds: int = Field(default=60, ge=10, le=600)


class AgentReportRequest(BaseModel):
    overall: dict
    segments: list[dict] = Field(default_factory=list)
    course: dict = Field(default_factory=dict)


class TeacherReviewRequest(BaseModel):
    owner: str = Field(min_length=1, max_length=80)
    due: str = Field(min_length=1, max_length=80)
    actions: str = Field(default="", max_length=4000)
    status: Literal["待提交", "已提交", "复评中", "已完成"] = "待提交"
    review_conclusion: Literal["尚未复核", "与课堂任务一致", "需要持续关注", "证据不足，无法判断"] = "尚未复核"
    context_notes: str = Field(default="", max_length=4000)
