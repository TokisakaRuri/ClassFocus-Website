from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app_api.core.config import settings
from app_api.db import crud


router = APIRouter()
MAX_VIDEO_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.replace(" ", "_")
    return "".join(char for char in name if char.isalnum() or char in "._-") or "video.mp4"


@router.post("/upload")
async def upload_video(
    course_name: str = Form(...),
    teacher_name: str = Form(""),
    class_name: str = Form(""),
    classroom: str = Form(""),
    lesson_date: str = Form(""),
    lesson_section: str = Form(""),
    video_file: UploadFile = File(...),
):
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="缺少视频文件名")
    course_name = course_name.strip()
    if not course_name:
        raise HTTPException(status_code=400, detail="课程名称不能为空")

    suffix = Path(video_file.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        allowed = "、".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"不支持的视频格式，仅支持 {allowed}")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uuid.uuid4().hex[:10]}_{_safe_filename(video_file.filename)}"
    save_path = settings.upload_dir / filename

    total_bytes = 0
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=settings.upload_dir,
            prefix=f".{filename}.",
            suffix=".upload",
            delete=False,
        ) as buffer:
            temporary_path = Path(buffer.name)
            while chunk := await video_file.read(UPLOAD_CHUNK_SIZE):
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="视频文件不能超过 2GB")
                buffer.write(chunk)
            buffer.flush()
            os.fsync(buffer.fileno())
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="视频文件为空")
        os.replace(temporary_path, save_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)

    try:
        course_id, video_id = crud.create_course_with_video(
            {
                "course_name": course_name,
                "teacher_name": teacher_name,
                "class_name": class_name,
                "classroom": classroom,
                "lesson_date": lesson_date,
                "lesson_section": lesson_section,
            },
            {
                "video_name": video_file.filename,
                "video_path": str(save_path),
                "analysis_status": "uploaded",
            },
        )
    except Exception:
        save_path.unlink(missing_ok=True)
        raise

    return {
        "message": "视频上传成功",
        "course_id": course_id,
        "video_id": video_id,
        "video_name": video_file.filename,
        "video_path": str(save_path),
        "size_mb": round(total_bytes / 1024 / 1024, 2),
    }


@router.get("")
async def list_videos(limit: int = Query(50, ge=1, le=200)):
    return {"items": crud.list_videos(limit=limit)}


@router.get("/{video_id}")
async def get_video(video_id: int):
    video = crud.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    return video
