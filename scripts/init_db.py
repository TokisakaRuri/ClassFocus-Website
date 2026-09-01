from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app_api.core.config import ensure_directories, settings
from app_api.db.database import init_db


def main() -> None:
    ensure_directories()
    init_db()
    print(f"Database initialized: {settings.database_path}")


if __name__ == "__main__":
    main()
