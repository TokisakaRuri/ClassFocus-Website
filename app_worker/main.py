from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import threading
import time
import uuid

from app_api.core.config import ensure_directories
from app_api.db import crud
from app_api.db.database import init_db, now_iso
from app_api.services.analysis_service import ModelServiceCache, run_analysis


LOGGER = logging.getLogger("classfocus.worker")


class AnalysisWorker:
    def __init__(self, *, poll_seconds: float = 1.0, lease_seconds: int = 45) -> None:
        suffix = uuid.uuid4().hex[:8]
        self.worker_id = f"{socket.gethostname()}-{os.getpid()}-{suffix}"
        self.poll_seconds = max(0.2, poll_seconds)
        self.lease_seconds = max(15, lease_seconds)
        self.started_at = now_iso()
        self.current_task_id: int | None = None
        self.stop_event = threading.Event()
        self.model_cache = ModelServiceCache()

    def stop(self, *_: object) -> None:
        self.stop_event.set()

    def _heartbeat_loop(self) -> None:
        while not self.stop_event.is_set():
            status = "busy" if self.current_task_id is not None else "idle"
            try:
                crud.heartbeat_worker(
                    self.worker_id,
                    started_at=self.started_at,
                    status=status,
                    current_task_id=self.current_task_id,
                )
                if self.current_task_id is not None:
                    crud.heartbeat_task(self.current_task_id, self.worker_id)
            except Exception:
                LOGGER.exception("Worker heartbeat failed")
            self.stop_event.wait(5)

    def run(self) -> None:
        ensure_directories()
        init_db()
        recovered = crud.recover_expired_tasks(self.lease_seconds)
        if recovered:
            LOGGER.info("Recovered %s expired task(s)", recovered)
        heartbeat = threading.Thread(target=self._heartbeat_loop, name="classfocus-worker-heartbeat", daemon=True)
        heartbeat.start()
        LOGGER.info("Worker %s started", self.worker_id)

        while not self.stop_event.is_set():
            try:
                task = crud.claim_next_task(self.worker_id, self.lease_seconds)
            except Exception:
                LOGGER.exception("Unable to claim the next analysis task")
                self.stop_event.wait(self.poll_seconds)
                continue
            if task is None:
                self.stop_event.wait(self.poll_seconds)
                continue
            self.current_task_id = int(task["id"])
            try:
                run_analysis(self.current_task_id, worker_id=self.worker_id, model_cache=self.model_cache)
            finally:
                self.current_task_id = None

        crud.heartbeat_worker(
            self.worker_id,
            started_at=self.started_at,
            status="stopped",
            current_task_id=None,
        )
        LOGGER.info("Worker %s stopped", self.worker_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="ClassFocus 独立推理 Worker")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--lease-seconds", type=int, default=45)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    worker = AnalysisWorker(poll_seconds=args.poll_seconds, lease_seconds=args.lease_seconds)
    signal.signal(signal.SIGINT, worker.stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, worker.stop)
    worker.run()


if __name__ == "__main__":
    main()
