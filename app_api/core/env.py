from __future__ import annotations

import os
import threading
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
_ENV_LOCK = threading.Lock()
_ENV_LOADED = False


def load_local_env() -> None:
    """Load the project .env once without overriding process-level settings."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    with _ENV_LOCK:
        if _ENV_LOADED:
            return
        env_path = ROOT_DIR / ".env"
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        _ENV_LOADED = True
