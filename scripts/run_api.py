from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app_api.core.config import settings


def main() -> None:
    uvicorn.run(
        "app_api.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )


if __name__ == "__main__":
    main()
