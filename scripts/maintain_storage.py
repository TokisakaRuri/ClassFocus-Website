from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app_api.core.config import ensure_directories, settings
from app_api.db import crud
from app_api.db.database import init_db


def _megabytes(size: int) -> float:
    return round(size / (1024 * 1024), 2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="压缩历史结果文件，并仅保留每个已结束任务最有代表性的关键帧。"
    )
    parser.add_argument(
        "--keep-frames",
        type=int,
        default=settings.max_key_frames,
        help=f"每个任务保留的关键帧数（默认：{settings.max_key_frames}）",
    )
    args = parser.parse_args()
    if args.keep_frames < 0:
        parser.error("--keep-frames 不能小于 0")

    ensure_directories()
    init_db()

    result_summary = crud.compact_legacy_result_files()
    frame_files = 0
    frame_bytes = 0
    processed_tasks = 0
    for task in crud.list_tasks(limit=10_000):
        if task.get("status") in {"waiting", "running"}:
            continue
        summary = crud.prune_task_frames(int(task["id"]), keep=args.keep_frames)
        frame_files += summary["removed"]
        frame_bytes += summary["reclaimed_bytes"]
        processed_tasks += 1

    print(
        json.dumps(
            {
                "processed_tasks": processed_tasks,
                "compacted_result_files": result_summary["compacted"],
                "removed_frame_files": frame_files,
                "reclaimed_result_mb": _megabytes(result_summary["reclaimed_bytes"]),
                "reclaimed_frame_mb": _megabytes(frame_bytes),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
