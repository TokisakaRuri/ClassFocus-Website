from __future__ import annotations

import contextlib
import io
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from app_api.core.config import current_model_config_path, current_model_path, current_model_repo_path, settings
from app_api.core.exceptions import AnalysisCanceled


CLASS_NAMES = {
    0: "listening",
    1: "writing",
    2: "reading",
    3: "using phone",
    4: "bowing the head",
    5: "sleeping",
}

BEHAVIOR_LABELS = {
    "listening": "听课",
    "writing": "书写",
    "reading": "阅读",
    "using phone": "使用手机",
    "bowing the head": "低头",
    "sleeping": "睡觉",
}

BOX_COLORS_RGB = {
    "listening": (0, 113, 227),
    "writing": (52, 199, 89),
    "reading": (94, 92, 230),
    "using phone": (255, 159, 10),
    "bowing the head": (255, 59, 48),
    "sleeping": (175, 82, 222),
}


ProgressCallback = Callable[[float], None]


class ClassroomYOLOService:
    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        repo_path: str | Path | None = None,
        device: str | None = None,
    ):
        self.model_path = Path(model_path or current_model_path())
        self.device = device or settings.device
        self.model = None
        self.postprocessor = None
        self.class_names = CLASS_NAMES
        self.config_path = Path(config_path or current_model_config_path())
        self.repo_path = Path(repo_path or current_model_repo_path())
        self.mode = "unavailable"
        self.message = "未加载可用推理模型"
        self._load_model_if_available()

    def _load_model_if_available(self) -> None:
        if not self.model_path.exists():
            self.message = f"模型文件不存在：{self.model_path}"
            return

        suffix = self.model_path.suffix.lower()
        if suffix == ".pth":
            self._load_deim_if_available()
            return
        if suffix != ".pt":
            self.message = f"不支持的模型权重格式：{self.model_path.suffix}"
            return

        yolo_config_base = settings.database_path.parent
        os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_base))

        try:
            from ultralytics import YOLO
        except ImportError:
            self.message = "未安装 ultralytics，无法执行 YOLO 视频推理"
            return

        try:
            self.model = YOLO(str(self.model_path))
        except Exception as exc:
            self.message = f"YOLO 模型加载失败：{exc}"
            return
        self.mode = "yolo"
        self.message = f"已加载 YOLO 模型：{self.model_path.name}"

    def _resolve_torch_device(self):
        import torch

        if self.device and self.device not in {"auto", "cuda"}:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() and self.device != "cpu" else "cpu")

    def _load_deim_if_available(self) -> None:
        if not self.config_path.exists():
            self.message = f"DEIM 检测配置不存在：{self.config_path}"
            return
        if not self.repo_path.exists():
            self.message = f"DEIM 工程路径不存在：{self.repo_path}"
            return

        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        try:
            import torch
            from engine.core import YAMLConfig
        except ImportError as exc:
            self.message = f"DEIM 推理依赖不可用：{exc}"
            return

        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                cfg = YAMLConfig(str(self.config_path), resume=str(self.model_path))
            if "HGNetv2" in cfg.yaml_cfg:
                cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, dict) and checkpoint.get("name"):
                names = checkpoint["name"]
                if isinstance(names, dict):
                    self.class_names = {int(key): str(value) for key, value in names.items()}
                elif isinstance(names, (list, tuple)):
                    self.class_names = {index: str(value) for index, value in enumerate(names)}
            if isinstance(checkpoint, dict) and "ema" in checkpoint:
                state = checkpoint["ema"]["module"]
            elif isinstance(checkpoint, dict) and "model" in checkpoint:
                state = checkpoint["model"]
            else:
                state = checkpoint

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                model = cfg.model
                model.load_state_dict(state)
                postprocessor = cfg.postprocessor.deploy()
            device = self._resolve_torch_device()
            self.model = model.deploy().to(device).eval()
            self.postprocessor = postprocessor
            self.device = str(device)
            self.mode = "deim"
            self.message = f"已加载 DEIM/DFINE 权重：{self.model_path.name}，配置：{self.config_path.name}"
        except Exception as exc:
            self.model = None
            self.postprocessor = None
            self.mode = "unavailable"
            self.message = f"DEIM/DFINE 模型加载失败：{exc}"

    def analyze_video(
        self,
        video_path: str | Path,
        confidence_threshold: float = 0.5,
        frame_sample_seconds: float = 1.0,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        video_path = Path(video_path)
        if self.model is None:
            raise RuntimeError(self.message)

        try:
            if self.mode == "deim":
                return self._analyze_with_deim(
                    video_path=video_path,
                    confidence_threshold=confidence_threshold,
                    frame_sample_seconds=frame_sample_seconds,
                    progress_callback=progress_callback,
                )
            return self._analyze_with_yolo(
                    video_path=video_path,
                    confidence_threshold=confidence_threshold,
                    frame_sample_seconds=frame_sample_seconds,
                    progress_callback=progress_callback,
                )
        except AnalysisCanceled:
            raise
        except Exception as exc:
            raise RuntimeError(f"模型推理失败：{exc}") from exc

    def _analyze_with_yolo(
        self,
        video_path: Path,
        confidence_threshold: float,
        frame_sample_seconds: float,
        progress_callback: ProgressCallback | None,
    ) -> dict[str, Any]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("未安装 opencv-python") from exc

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{video_path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = total_frames / fps if fps > 0 else 0
            sample_interval = max(1, int((fps or 25) * frame_sample_seconds))
            detections: list[dict[str, Any]] = []
            frame_id = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % sample_interval == 0:
                    timestamp = frame_id / fps if fps > 0 else frame_id / 25
                    kwargs = {
                        "source": frame,
                        "conf": confidence_threshold,
                        "verbose": False,
                    }
                    if self.device and self.device != "auto":
                        kwargs["device"] = self.device
                    yolo_results = self.model.predict(**kwargs)
                    parsed = self._parse_result(yolo_results, frame_id, timestamp)

                    detections.extend(parsed)

                frame_id += 1
                if progress_callback and total_frames:
                    progress_callback(min(95, frame_id / total_frames * 100))
        finally:
            cap.release()

        return {
            "mode": "yolo",
            "message": self.message,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration": round(duration, 2),
            "resolution": f"{width}x{height}" if width and height else "",
            "detections": detections,
        }

    def _parse_result(self, yolo_results: Any, frame_id: int, timestamp: float) -> list[dict[str, Any]]:
        result = yolo_results[0]
        if result.boxes is None:
            return []

        detections: list[dict[str, Any]] = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "frame_id": frame_id,
                    "timestamp": round(timestamp, 3),
                    "label": self.class_names.get(class_id, str(class_id)),
                    "confidence": round(confidence, 4),
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2),
                    "image_path": None,
                }
            )
        return detections

    def _analyze_with_deim(
        self,
        video_path: Path,
        confidence_threshold: float,
        frame_sample_seconds: float,
        progress_callback: ProgressCallback | None,
    ) -> dict[str, Any]:
        try:
            import cv2
            import torch
            import torchvision.transforms as T
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(f"DEIM 推理依赖不可用：{exc}") from exc

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{video_path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = total_frames / fps if fps > 0 else 0
            sample_interval = max(1, int((fps or 25) * frame_sample_seconds))
            device = self._resolve_torch_device()
            transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
            detections: list[dict[str, Any]] = []
            frame_id = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % sample_interval == 0:
                    timestamp = frame_id / fps if fps > 0 else frame_id / 25
                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    w, h = frame_pil.size
                    orig_size = torch.tensor([[w, h]], device=device)
                    im_data = transforms(frame_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred = self.model(im_data)
                        labels, boxes, scores = self.postprocessor(pred, orig_size)

                    parsed = self._parse_deim_result(labels, boxes, scores, frame_id, timestamp, confidence_threshold)
                    detections.extend(parsed)

                frame_id += 1
                if progress_callback and total_frames:
                    progress_callback(min(95, frame_id / total_frames * 100))
        finally:
            cap.release()
        return {
            "mode": "deim",
            "message": self.message,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration": round(duration, 2),
            "resolution": f"{width}x{height}" if width and height else "",
            "detections": detections,
        }

    def _parse_deim_result(
        self,
        labels: Any,
        boxes: Any,
        scores: Any,
        frame_id: int,
        timestamp: float,
        confidence_threshold: float,
    ) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        batch_labels = labels[0]
        batch_boxes = boxes[0]
        batch_scores = scores[0]
        valid = batch_scores > confidence_threshold

        for label, box, score in zip(batch_labels[valid], batch_boxes[valid], batch_scores[valid]):
            class_id = int(label.item())
            x1, y1, x2, y2 = [float(value) for value in box.tolist()]
            detections.append(
                {
                    "frame_id": frame_id,
                    "timestamp": round(timestamp, 3),
                    "label": self.class_names.get(class_id, str(class_id)),
                    "confidence": round(float(score.item()), 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "image_path": None,
                }
            )
        return detections

    def save_key_frames(
        self,
        video_path: str | Path,
        detections: list[dict[str, Any]],
        frame_dir: str | Path,
        limit: int = 24,
    ) -> int:
        """Persist only the most informative detected frames for later review."""
        if not detections or limit <= 0:
            return 0
        try:
            import cv2
        except ImportError:
            return 0

        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for detection in detections:
            by_frame[int(detection.get("frame_id") or 0)].append(detection)
        ranked = sorted(
            by_frame.items(),
            key=lambda item: (
                len({str(row.get("label") or "") for row in item[1]}),
                len(item[1]),
                sum(float(row.get("confidence") or 0) for row in item[1]) / max(1, len(item[1])),
            ),
            reverse=True,
        )[:limit]

        destination = Path(frame_dir)
        destination.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return 0
        saved = 0
        try:
            for frame_id, rows in sorted(ranked, key=lambda item: item[0]):
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id))
                ok, frame = capture.read()
                if not ok:
                    continue
                save_path = destination / f"frame_{frame_id}.jpg"
                annotated = self._draw_detection_boxes(frame, rows)
                if not cv2.imwrite(
                    str(save_path),
                    annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82],
                ):
                    continue
                for row in rows:
                    row["image_path"] = str(save_path)
                saved += 1
        finally:
            capture.release()
        return saved

    def _draw_detection_boxes(self, frame: Any, detections: list[dict[str, Any]]) -> Any:
        import cv2
        import numpy as np

        try:
            from PIL import Image, ImageDraw, ImageFont

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            font = None
            for font_path in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyhbd.ttc"]:
                if Path(font_path).exists():
                    font = ImageFont.truetype(font_path, size=max(15, int(min(image.size) / 55)))
                    break
            font = font or ImageFont.load_default()
            line_width = max(2, int(min(image.size) / 220))

            for det in detections:
                x1, y1, x2, y2 = [float(det[key]) for key in ["x1", "y1", "x2", "y2"]]
                label_key = str(det["label"])
                color = BOX_COLORS_RGB.get(label_key, (0, 113, 227))
                label = f"{BEHAVIOR_LABELS.get(label_key, label_key)} {float(det['confidence']) * 100:.0f}%"
                draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
                text_box = draw.textbbox((x1, y1), label, font=font)
                text_w = text_box[2] - text_box[0]
                text_h = text_box[3] - text_box[1]
                text_y = max(0, y1 - text_h - 8)
                draw.rounded_rectangle((x1, text_y, x1 + text_w + 10, text_y + text_h + 7), radius=4, fill=color)
                draw.text((x1 + 5, text_y + 2), label, fill=(255, 255, 255), font=font)
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

        for det in detections:
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            label_key = str(det["label"])
            label = f"{BEHAVIOR_LABELS.get(label_key, label_key)} {float(det['confidence']) * 100:.0f}%"
            rgb = BOX_COLORS_RGB.get(label_key, (0, 113, 227))
            color = (rgb[2], rgb[1], rgb[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text_origin = (x1, max(18, y1 - 6))
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(
                frame,
                (text_origin[0], text_origin[1] - text_height - 8),
                (text_origin[0] + text_width + 8, text_origin[1] + 4),
                color,
                -1,
            )
            cv2.putText(frame, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return frame
