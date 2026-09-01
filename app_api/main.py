from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app_api.core.config import ensure_directories, settings
from app_api.core.env import load_local_env
from app_api.core.security import API_TOKEN_HEADER, get_local_api_token
from app_api.db import crud
from app_api.db.database import init_db
from app_api.routers import agent, model, report, task, video


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_directories()
    init_db()
    crud.compact_legacy_result_files()
    yield


app = FastAPI(
    title="ClassFocus API",
    description="基于 YOLO 的学生课堂行为识别与智能分析系统",
    version=settings.app_version,
    lifespan=lifespan,
)

load_local_env()
allowed_origins = [
    item.strip()
    for item in os.getenv(
        "CLASSFOCUS_ALLOWED_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    if item.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", API_TOKEN_HEADER],
    expose_headers=["Content-Disposition"],
)


PUBLIC_PATHS = {"/", "/api/health", "/api/session", "/docs", "/openapi.json", "/redoc"}
SESSION_COOKIE = "classfocus_session"


@app.middleware("http")
async def require_local_api_token(request: Request, call_next):
    if request.method != "OPTIONS" and request.url.path.startswith("/api/") and request.url.path not in PUBLIC_PATHS:
        supplied = request.headers.get(API_TOKEN_HEADER, "") or request.cookies.get(SESSION_COOKIE, "")
        expected = get_local_api_token()
        if not supplied or not hmac.compare_digest(supplied, expected):
            return JSONResponse(status_code=401, content={"detail": "缺少或无效的本地 API 访问令牌"})
    return await call_next(request)

app.include_router(video.router, prefix="/api/videos", tags=["视频管理"])
app.include_router(task.router, prefix="/api/tasks", tags=["分析任务"])
app.include_router(report.router, prefix="/api/reports", tags=["行为报告"])
app.include_router(agent.router, prefix="/api/agent", tags=["总结建议"])
app.include_router(model.router, prefix="/api/models", tags=["模型管理"])


@app.post("/api/session")
async def create_local_session(request: Request):
    origin = request.headers.get("origin", "")
    if origin and origin not in allowed_origins:
        return JSONResponse(status_code=403, content={"detail": "不允许的本地界面来源"})
    response = JSONResponse({"status": "ok"})
    response.set_cookie(
        SESSION_COOKIE,
        get_local_api_token(),
        max_age=12 * 60 * 60,
        httponly=True,
        samesite="strict",
        secure=False,
        path="/",
    )
    return response


@app.get("/api/health")
async def health_check():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "ok",
        "worker": crud.get_worker_health(),
    }


FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")


@app.get("/{full_path:path}", include_in_schema=False)
async def frontend_app(full_path: str):
    index_path = FRONTEND_DIST / "index.html"
    requested = (FRONTEND_DIST / full_path).resolve()
    if full_path and requested.is_relative_to(FRONTEND_DIST.resolve()) and requested.is_file():
        return FileResponse(requested)
    if index_path.is_file():
        return FileResponse(index_path)
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "ok",
        "docs": "/docs",
        "message": "React 工作台尚未构建，请在 frontend 目录运行 npm run build",
    }
