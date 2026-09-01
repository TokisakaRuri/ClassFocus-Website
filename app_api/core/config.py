from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from app_api.core.env import load_local_env


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"
load_local_env()


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}

    try:
        import yaml
    except ImportError:
        return {}

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


_CONFIG = _load_config()


def _get(path: str, default: Any) -> Any:
    node: Any = _CONFIG
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


@dataclass(frozen=True)
class Settings:
    app_name: str = str(_get("app.name", "ClassFocus"))
    app_version: str = str(_get("app.version", "1.0.0"))
    host: str = str(_get("server.host", "127.0.0.1"))
    port: int = int(_get("server.port", 8000))
    database_path: Path = ROOT_DIR / str(_get("database.path", "classroom_behavior.db"))
    upload_dir: Path = ROOT_DIR / str(_get("storage.upload_dir", "uploads/videos"))
    frame_dir: Path = ROOT_DIR / str(_get("storage.frame_dir", "uploads/frames"))
    result_dir: Path = ROOT_DIR / str(_get("storage.result_dir", "uploads/results"))
    report_dir: Path = ROOT_DIR / str(_get("storage.report_dir", "uploads/reports"))
    model_name: str = str(_get("model.name", "classroom_yolo"))
    model_path: Path = ROOT_DIR / str(_get("model.path", "models/best.pt"))
    model_config_path: Path = ROOT_DIR / str(_get("model.config_path", "configs/detection/scb/deim_MSCF_s_oc3500.yml"))
    model_repo_path: Path = Path(
        os.getenv("DEIM_REPO_PATH") or str(_get("model.repo_path", "vendor/deim"))
    )
    confidence_threshold: float = float(_get("model.confidence_threshold", 0.5))
    iou_threshold: float = float(_get("model.iou_threshold", 0.45))
    device: str = str(_get("model.device", "auto"))
    frame_sample_seconds: float = float(_get("analysis.frame_sample_seconds", 1))
    segment_seconds: int = int(_get("analysis.segment_seconds", 60))
    save_key_frames: bool = bool(_get("analysis.save_key_frames", True))
    max_key_frames: int = max(0, int(_get("analysis.max_key_frames", 24)))


settings = Settings()


def current_model_path() -> Path:
    config = _load_config()
    model = config.get("model", {}) if isinstance(config, dict) else {}
    model_path = model.get("path", str(settings.model_path)) if isinstance(model, dict) else str(settings.model_path)
    candidate = Path(str(model_path))
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def current_model_name() -> str:
    config = _load_config()
    model = config.get("model", {}) if isinstance(config, dict) else {}
    if isinstance(model, dict):
        return str(model.get("name") or settings.model_name)
    return settings.model_name


def current_model_config_path() -> Path:
    config = _load_config()
    model = config.get("model", {}) if isinstance(config, dict) else {}
    config_path = model.get("config_path", str(settings.model_config_path)) if isinstance(model, dict) else str(settings.model_config_path)
    candidate = Path(str(config_path))
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def current_model_repo_path() -> Path:
    configured_env = os.getenv("DEIM_REPO_PATH", "").strip()
    if configured_env:
        return Path(configured_env)
    config = _load_config()
    model = config.get("model", {}) if isinstance(config, dict) else {}
    repo_path = model.get("repo_path", str(settings.model_repo_path)) if isinstance(model, dict) else str(settings.model_repo_path)
    candidate = Path(str(repo_path))
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def ensure_directories() -> None:
    for directory in (
        settings.upload_dir,
        settings.frame_dir,
        settings.result_dir,
        settings.report_dir,
        settings.model_path.parent,
        settings.model_config_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
