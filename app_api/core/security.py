from __future__ import annotations

import os
import secrets
import threading
from pathlib import Path

from app_api.core.env import ROOT_DIR, load_local_env


API_TOKEN_HEADER = "X-ClassFocus-Token"
TOKEN_PATH = ROOT_DIR / ".classfocus-token"
_TOKEN_LOCK = threading.Lock()
_TOKEN: str | None = None


def get_local_api_token() -> str:
    """Return the shared token used by local ClassFocus clients."""
    global _TOKEN
    if _TOKEN:
        return _TOKEN

    with _TOKEN_LOCK:
        if _TOKEN:
            return _TOKEN
        load_local_env()
        configured = os.getenv("CLASSFOCUS_API_TOKEN", "").strip()
        if configured:
            _TOKEN = configured
            return _TOKEN

        try:
            stored = TOKEN_PATH.read_text(encoding="utf-8").strip()
        except OSError:
            stored = ""
        if stored:
            _TOKEN = stored
            return _TOKEN

        token = secrets.token_urlsafe(32)
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with TOKEN_PATH.open("x", encoding="utf-8") as file:
                file.write(token)
            try:
                TOKEN_PATH.chmod(0o600)
            except OSError:
                pass
            _TOKEN = token
        except FileExistsError:
            _TOKEN = TOKEN_PATH.read_text(encoding="utf-8").strip()
        if not _TOKEN:
            raise RuntimeError("无法创建本地 API 访问令牌")
        return _TOKEN


def api_auth_headers() -> dict[str, str]:
    return {API_TOKEN_HEADER: get_local_api_token()}
