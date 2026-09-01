from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app_api.db import crud


router = APIRouter()


@router.get("/{task_id}")
async def get_report(task_id: int):
    task = crud.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    result = crud.load_result_json(task.get("result_path"))
    report = crud.get_report_by_task(task_id)
    if not result and not report:
        raise HTTPException(status_code=404, detail="报告尚未生成")
    return {"task": task, "report": report, "result": result}


@router.get("/{task_id}/download")
async def download_report(task_id: int):
    report = crud.get_report_by_task(task_id)
    if not report or not report.get("report_path"):
        raise HTTPException(status_code=404, detail="报告文件不存在")

    path = Path(report["report_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="报告文件不存在")

    media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if path.suffix.lower() == ".txt":
        media_type = "text/plain"
    return FileResponse(path, filename=path.name, media_type=media_type)

