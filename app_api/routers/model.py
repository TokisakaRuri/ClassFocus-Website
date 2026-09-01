from __future__ import annotations

import importlib.util
import os
import tempfile
import uuid
from pathlib import Path

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app_api.core.config import (
    CONFIG_PATH,
    ROOT_DIR,
    current_model_config_path,
    current_model_name,
    current_model_path,
    current_model_repo_path,
    settings,
)


router = APIRouter()
MODEL_SUFFIXES = {".pt", ".pth"}
CONFIG_SUFFIXES = {".yml", ".yaml"}
UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_MODEL_UPLOAD_BYTES = 1024 * 1024 * 1024
MAX_CONFIG_UPLOAD_BYTES = 2 * 1024 * 1024
DEIM_RUNTIME_IMPORTS = {
    "torch": "PyTorch",
    "cv2": "OpenCV",
    "yaml": "PyYAML",
    "tensorboard": "TensorBoard",
    "faster_coco_eval": "faster-coco-eval",
    "calflops": "calflops",
    "pywt": "PyWavelets",
    "pytorch_wavelets": "pytorch_wavelets",
    "timm": "timm",
    "torch_dct": "torch-dct",
    "einops": "einops",
    "transformers": "transformers",
}


def _model_family(path: Path) -> str:
    name = path.name.lower()
    if path.suffix.lower() == ".pt":
        return "YOLO"
    if "dfine" in name or "d-fine" in name:
        return "DFINE"
    if "detr" in name:
        return "DETR"
    if "deim" in name or "stg" in name:
        return "DEIM"
    return "DETR/DFINE/DEIM"


def _model_files() -> list[Path]:
    return sorted(
        path
        for path in settings.model_path.parent.iterdir()
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    )


def _config_files() -> list[Path]:
    config_dir = settings.model_config_path.parent
    if not config_dir.exists():
        return []
    return sorted(
        path
        for path in config_dir.iterdir()
        if path.is_file() and path.suffix.lower() in CONFIG_SUFFIXES
    )


def _relative_model_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def _relative_project_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def _available_destination(directory: Path, source_name: str) -> Path:
    destination = directory / source_name
    if not destination.exists():
        return destination
    source = Path(source_name)
    return directory / f"{source.stem}_{uuid.uuid4().hex[:8]}{source.suffix.lower()}"


async def _store_upload(upload: UploadFile, destination: Path, maximum_bytes: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    total = 0
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".upload",
            delete=False,
        ) as buffer:
            temporary_path = Path(buffer.name)
            while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
                total += len(chunk)
                if total > maximum_bytes:
                    raise HTTPException(status_code=413, detail="上传文件超过允许大小")
                buffer.write(chunk)
            buffer.flush()
            os.fsync(buffer.fileno())
        if total <= 0:
            raise HTTPException(status_code=400, detail="上传文件为空")
        os.replace(temporary_path, destination)
        return total
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def _load_project_config() -> dict:
    data = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        data = {}
    return data


def _save_project_config(data: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=CONFIG_PATH.parent,
            prefix=f".{CONFIG_PATH.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, CONFIG_PATH)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def _write_default_model(path: Path) -> None:
    data = _load_project_config()
    model = data.setdefault("model", {})
    if not isinstance(model, dict):
        model = {}
        data["model"] = model
    model["name"] = f"classroom_{_model_family(path).lower().replace('/', '_')}"
    model["path"] = _relative_model_path(path)
    _save_project_config(data)


def _write_default_config(path: Path) -> None:
    data = _load_project_config()
    model = data.setdefault("model", {})
    if not isinstance(model, dict):
        model = {}
        data["model"] = model
    model["config_path"] = _relative_project_path(path)
    _save_project_config(data)


def _missing_imports(imports: dict[str, str]) -> list[str]:
    return [label for module, label in imports.items() if importlib.util.find_spec(module) is None]


@router.get("/current")
async def get_current_model():
    model_path = current_model_path()
    config_path = current_model_config_path()
    repo_path = current_model_repo_path()
    model_files = _model_files()
    config_files = _config_files()
    ultralytics_ready = importlib.util.find_spec("ultralytics") is not None
    opencv_ready = importlib.util.find_spec("cv2") is not None
    torch_ready = importlib.util.find_spec("torch") is not None
    model_family = _model_family(model_path)
    model_exists = model_path.exists()
    repo_engine_ready = (repo_path / "engine" / "core").exists()
    deim_missing_dependencies = _missing_imports(DEIM_RUNTIME_IMPORTS)
    deim_runtime_ready = (
        not deim_missing_dependencies and config_path.exists() and repo_path.exists() and repo_engine_ready
    )
    inference_supported = model_exists and (
        model_path.suffix.lower() == ".pt"
        or (model_path.suffix.lower() == ".pth" and config_path.exists() and repo_path.exists() and repo_engine_ready)
    )
    runtime_available = (
        model_exists
        and (
            (model_path.suffix.lower() == ".pt" and ultralytics_ready and opencv_ready)
            or (model_path.suffix.lower() == ".pth" and deim_runtime_ready)
        )
    )
    if runtime_available:
        runtime_message = "当前默认模型可用于课堂行为视频推理"
    elif model_path.suffix.lower() == ".pth":
        if not config_path.exists() or not repo_path.exists():
            runtime_message = "当前默认权重是 DETR/DFINE/DEIM .pth 训练权重；需要配置可用的检测 YAML 与 DEIM 工程路径"
        elif not repo_engine_ready:
            runtime_message = "DEIM 工程路径缺少 engine/core 模块，请检查工程目录是否正确"
        elif deim_missing_dependencies:
            runtime_message = f"DEIM/DFINE 推理依赖未安装：{', '.join(deim_missing_dependencies)}"
        else:
            runtime_message = "当前默认 DEIM/DFINE 权重与检测配置已就绪，可按 YAMLConfig 流程执行视频抽帧推理"
    else:
        runtime_message = "当前默认模型暂不可用于视频推理"
    return {
        "model_name": current_model_name(),
        "model_path": str(model_path),
        "config_path": str(config_path),
        "repo_path": str(repo_path),
        "model_family": model_family,
        "exists": model_exists,
        "config_exists": config_path.exists(),
        "repo_exists": repo_path.exists(),
        "repo_engine_ready": repo_engine_ready,
        "deim_missing_dependencies": deim_missing_dependencies,
        "runtime_available": runtime_available,
        "inference_supported": inference_supported,
        "runtime_message": runtime_message,
        "ultralytics_ready": ultralytics_ready,
        "opencv_ready": opencv_ready,
        "torch_ready": torch_ready,
        "device": settings.device,
        "available_models": [
            {
                "name": path.name,
                "path": str(path),
                "family": _model_family(path),
                "is_default": path.resolve() == model_path.resolve(),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
            }
            for path in model_files
        ],
        "available_configs": [
            {
                "name": path.name,
                "path": str(path),
                "is_default": path.resolve() == config_path.resolve(),
                "size_kb": round(path.stat().st_size / 1024, 2),
            }
            for path in config_files
        ],
    }


@router.post("/upload")
async def upload_model(
    model_file: UploadFile = File(...),
    make_default: bool = Form(True),
):
    if not model_file.filename:
        raise HTTPException(status_code=400, detail="缺少模型文件名")

    source_name = Path(model_file.filename).name
    suffix = Path(source_name).suffix.lower()
    if suffix not in MODEL_SUFFIXES:
        raise HTTPException(status_code=400, detail="请上传 .pt 或 .pth 模型权重文件")

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    save_path = _available_destination(settings.model_path.parent, source_name)
    total_bytes = await _store_upload(model_file, save_path, MAX_MODEL_UPLOAD_BYTES)

    if make_default:
        _write_default_model(save_path)

    return {
        "message": "模型上传成功",
        "model_path": str(save_path),
        "model_family": _model_family(save_path),
        "is_default": make_default,
        "size_mb": round(total_bytes / 1024 / 1024, 2),
    }


@router.post("/config")
async def upload_model_config(
    config_file: UploadFile = File(...),
    make_default: bool = Form(True),
):
    if not config_file.filename:
        raise HTTPException(status_code=400, detail="缺少配置文件名")

    source_name = Path(config_file.filename).name
    suffix = Path(source_name).suffix.lower()
    if suffix not in CONFIG_SUFFIXES:
        raise HTTPException(status_code=400, detail="请上传 .yml 或 .yaml 检测配置文件")

    settings.model_config_path.parent.mkdir(parents=True, exist_ok=True)
    save_path = _available_destination(settings.model_config_path.parent, source_name)
    await _store_upload(config_file, save_path, MAX_CONFIG_UPLOAD_BYTES)
    try:
        parsed = yaml.safe_load(save_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("配置根节点必须是对象")
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"检测配置无效：{exc}") from exc

    if make_default:
        _write_default_config(save_path)

    return {
        "message": "检测配置上传成功",
        "config_path": str(save_path),
        "is_default": make_default,
    }
