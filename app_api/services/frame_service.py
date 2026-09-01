from __future__ import annotations

import base64
import io
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app_api.core.config import settings
from app_api.db import crud


BEHAVIOR_LABELS = {
    "listening": "听课",
    "writing": "书写",
    "reading": "阅读",
    "using phone": "使用手机",
    "bowing the head": "低头",
    "sleeping": "睡觉",
}
REVIEW_CUE_LABELS = {"using phone", "sleeping"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_target_contact_sheet(
    image: Any,
    rows: list[dict[str, Any]],
    *,
    id_offset: int = 0,
) -> bytes | None:
    """Build a readable target sheet so the vision model does not inspect 40+ tiny people at once."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        if not rows:
            return None
        if len(rows) <= 6:
            columns, cell_width, cell_height = 3, 320, 250
        elif len(rows) <= 12:
            columns, cell_width, cell_height = 4, 260, 210
        else:
            columns, cell_width, cell_height = (6 if len(rows) > 30 else 5), 206, 174
        header_height, padding = 24, 7
        sheet_rows = (len(rows) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * cell_width, sheet_rows * cell_height), (238, 242, 247))
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except OSError:
            font = ImageFont.load_default()

        for local_id, row in enumerate(rows):
            detection_id = id_offset + local_id
            column = local_id % columns
            line = local_id // columns
            cell_x, cell_y = column * cell_width, line * cell_height
            draw.rounded_rectangle(
                (cell_x + 3, cell_y + 3, cell_x + cell_width - 3, cell_y + cell_height - 3),
                radius=10,
                fill=(255, 255, 255),
                outline=(207, 214, 224),
                width=1,
            )
            draw.text((cell_x + 10, cell_y + 6), f"ID {detection_id}", fill=(29, 29, 31), font=font)

            x1 = _safe_number(row.get("x1"))
            y1 = _safe_number(row.get("y1"))
            x2 = _safe_number(row.get("x2"))
            y2 = _safe_number(row.get("y2"))
            box_width = max(12.0, x2 - x1)
            box_height = max(12.0, y2 - y1)
            crop_box = (
                max(0, int(x1 - box_width * 0.3)),
                max(0, int(y1 - box_height * 0.22)),
                min(image.width, int(x2 + box_width * 0.3)),
                # Preserve nearby blockers while keeping the upper body large enough
                # for a 9B vision model to inspect shoulder/chest occlusion.
                min(image.height, int(y2 + box_height * 0.18)),
            )
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                continue
            crop = image.crop(crop_box)
            target_width = cell_width - padding * 2
            target_height = cell_height - header_height - padding * 2
            scale = min(target_width / crop.width, target_height / crop.height)
            resized = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
            paste_x = cell_x + (cell_width - resized.width) // 2
            paste_y = cell_y + header_height + (target_height - resized.height) // 2 + padding
            sheet.paste(resized, (paste_x, paste_y))

            target_box = (
                paste_x + int((x1 - crop_box[0]) * scale),
                paste_y + int((y1 - crop_box[1]) * scale),
                paste_x + int((x2 - crop_box[0]) * scale),
                paste_y + int((y2 - crop_box[1]) * scale),
            )
            draw.rectangle(target_box, outline=(10, 132, 255), width=2)

        buffer = io.BytesIO()
        sheet.save(buffer, format="JPEG", quality=86, optimize=True)
        return buffer.getvalue()
    except Exception:
        return None


def _format_timestamp(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _safe_child(path_value: Any, root: Path) -> Path | None:
    if not path_value:
        return None
    candidate = Path(str(path_value))
    if not candidate.is_absolute():
        candidate = settings.database_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
        root_resolved = root.resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_file() or resolved.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    return resolved if resolved.is_relative_to(root_resolved) else None


def _task_frame_file(task_id: int, path_value: Any) -> Path | None:
    return _safe_child(path_value, settings.frame_dir / str(task_id))


def _parse_resolution(value: Any) -> tuple[int, int]:
    try:
        width, height = str(value).lower().split("x", 1)
        width_value, height_value = int(width), int(height)
        if width_value > 0 and height_value > 0:
            return width_value, height_value
    except (TypeError, ValueError):
        pass
    return 1920, 1080


def build_frame_analysis(task: dict[str, Any], limit: int = 8) -> dict[str, Any]:
    task_id = int(task["id"])
    summaries = []
    for row in crud.list_frame_summaries(task_id):
        total = int(row.get("target_count") or 0)
        review_cue_count = int(row.get("review_cue_count") or 0)
        summaries.append(
            {
                **row,
                "frame_id": int(row.get("frame_id") or 0),
                "timestamp": _safe_number(row.get("timestamp")),
                "target_count": total,
                "class_count": int(row.get("class_count") or 0),
                "average_confidence": round(_safe_number(row.get("average_confidence")) * 100, 2),
                "review_cue_count": review_cue_count,
                "review_cue_rate": round(review_cue_count / total * 100, 2) if total else 0,
            }
        )

    ranked = sorted(
        summaries,
        key=lambda row: (row["class_count"], row["target_count"], row["average_confidence"]),
        reverse=True,
    )
    chosen = ranked[:limit]
    highlight_frame_id = chosen[0]["frame_id"] if chosen else None
    chosen.sort(key=lambda row: (row["timestamp"], row["frame_id"]))

    detections_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in crud.list_detections_for_frames(task_id, [item["frame_id"] for item in chosen]):
        frame_id = int(row.get("frame_id") or 0)
        detection_id = len(detections_by_frame[frame_id])
        label = str(row.get("label") or "")
        detections_by_frame[frame_id].append(
            {
                "id": detection_id,
                "label": label,
                "labelText": BEHAVIOR_LABELS.get(label, label),
                "confidence": round(_safe_number(row.get("confidence")) * 100, 2),
                "isReviewCue": label in REVIEW_CUE_LABELS,
                "box": {
                    "x1": round(_safe_number(row.get("x1")), 2),
                    "y1": round(_safe_number(row.get("y1")), 2),
                    "x2": round(_safe_number(row.get("x2")), 2),
                    "y2": round(_safe_number(row.get("y2")), 2),
                },
            }
        )

    result_payload = crud.load_result_json(task.get("result_path")) or {}
    width, height = _parse_resolution(
        task.get("resolution") or (result_payload.get("video_info") or {}).get("resolution")
    )
    frames = []
    for row in chosen:
        frame_id = row["frame_id"]
        detections = detections_by_frame.get(frame_id, [])
        behavior_counts = Counter(item["label"] for item in detections)
        dominant = behavior_counts.most_common(1)[0][0] if behavior_counts else ""
        frames.append(
            {
                "frameId": frame_id,
                "timestamp": row["timestamp"],
                "timeLabel": _format_timestamp(row["timestamp"]),
                "targetCount": row["target_count"],
                "classCount": row["class_count"],
                "reviewCueCount": row["review_cue_count"],
                "reviewCueRate": row["review_cue_rate"],
                "averageConfidence": row["average_confidence"],
                "dominantBehavior": BEHAVIOR_LABELS.get(dominant, dominant),
                "behaviorCounts": {key: int(behavior_counts.get(key, 0)) for key in BEHAVIOR_LABELS},
                "imageUrl": f"/api/tasks/{task_id}/frames/{frame_id}/image",
                "cleanImageUrl": f"/api/tasks/{task_id}/frames/{frame_id}/image?variant=clean_strict",
                "detections": detections,
            }
        )

    return {
        "taskId": task_id,
        "availableFrameCount": len(summaries),
        "highlightFrameId": highlight_frame_id,
        "selectionRule": "按行为类别覆盖、目标数量与检测置信度综合选取，优先展示信息丰富的关键帧",
        "resolution": {"width": width, "height": height},
        "frames": frames,
    }


def get_frame_source(task: dict[str, Any], frame_id: int) -> tuple[Path | None, list[dict[str, Any]]] | None:
    task_id = int(task["id"])
    rows = crud.list_detections_for_frames(task_id, [frame_id])
    if not rows:
        return None
    image_path = _task_frame_file(task_id, rows[0].get("image_path"))
    return image_path, rows


def extract_clean_frame(task: dict[str, Any], frame_id: int) -> bytes | None:
    video_path = Path(str(task.get("video_path") or ""))
    if not video_path.is_absolute():
        video_path = settings.database_path.parent / video_path
    try:
        video_resolved = video_path.resolve(strict=True)
        upload_root = settings.upload_dir.resolve(strict=True)
    except OSError:
        return None
    if not video_resolved.is_file() or not video_resolved.is_relative_to(upload_root):
        return None
    try:
        import cv2

        capture = cv2.VideoCapture(str(video_resolved))
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_id)))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            return None
        encoded, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return bytes(buffer) if encoded else None
    except Exception:
        return None


def render_annotated_frame(task: dict[str, Any], frame_id: int) -> bytes | None:
    source = get_frame_source(task, frame_id)
    if source is None:
        return None
    stored_path, rows = source
    clean = extract_clean_frame(task, frame_id)
    if not clean:
        if stored_path is None:
            return None
        try:
            return stored_path.read_bytes()
        except OSError:
            return None
    try:
        from PIL import Image, ImageDraw, ImageFont

        image = Image.open(io.BytesIO(clean)).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        colors = {
            "listening": (10, 132, 255),
            "writing": (48, 209, 88),
            "reading": (94, 92, 230),
            "using phone": (255, 159, 10),
            "bowing the head": (255, 69, 58),
            "sleeping": (191, 90, 242),
        }
        for row in rows:
            label = str(row.get("label") or "")
            box = tuple(_safe_number(row.get(key)) for key in ("x1", "y1", "x2", "y2"))
            color = colors.get(label, (10, 132, 255))
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0] + 3, max(0, box[1] - 14)), BEHAVIOR_LABELS.get(label, label), fill=color, font=font)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=84, optimize=True)
        return buffer.getvalue()
    except Exception:
        return None


def build_occlusion_payload(task: dict[str, Any], frame_id: int) -> dict[str, Any] | None:
    source = get_frame_source(task, frame_id)
    if source is None:
        return None
    image_path, rows = source
    try:
        from PIL import Image, ImageDraw, ImageFont

        clean = extract_clean_frame(task, frame_id)
        if clean:
            clean_image = Image.open(io.BytesIO(clean)).convert("RGB")
        elif image_path is not None:
            clean_image = Image.open(image_path).convert("RGB")
        else:
            return None
        original_size = clean_image.size
        detail_sheet = _build_target_contact_sheet(clean_image, rows)
        detail_image_batches = []
        try:
            review_batch_size = int(os.getenv("VISION_BATCH_SIZE", "6"))
        except ValueError:
            review_batch_size = 6
        review_batch_size = max(4, min(16, review_batch_size))
        for batch_start in range(0, len(rows), review_batch_size):
            batch_rows = rows[batch_start : batch_start + review_batch_size]
            batch_sheet = _build_target_contact_sheet(clean_image, batch_rows, id_offset=batch_start)
            if batch_sheet:
                detail_image_batches.append(
                    {
                        "ids": list(range(batch_start, batch_start + len(batch_rows))),
                        "image_mime": "image/jpeg",
                        "image_base64": base64.b64encode(batch_sheet).decode("ascii"),
                    }
                )
        image = clean_image.copy()
        max_width = 1280
        scale = min(1.0, max_width / image.width)
        if scale < 1:
            image = image.resize((max_width, int(image.height * scale)))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        detections = []
        for detection_id, row in enumerate(rows):
            x1 = _safe_number(row.get("x1")) * scale
            y1 = _safe_number(row.get("y1")) * scale
            x2 = _safe_number(row.get("x2")) * scale
            y2 = _safe_number(row.get("y2")) * scale
            draw.rectangle((x1, y1, x2, y2), outline=(255, 214, 10), width=max(2, int(3 * scale)))
            draw.rectangle((x1, y1, x1 + 42, y1 + 18), fill=(29, 29, 31))
            draw.text((x1 + 4, y1 + 3), f"ID {detection_id}", fill=(255, 255, 255), font=font)
            detections.append(
                {
                    "id": detection_id,
                    "label": str(row.get("label") or ""),
                    "confidence": round(_safe_number(row.get("confidence")), 4),
                    "box": {
                        "x1": round(_safe_number(row.get("x1")), 2),
                        "y1": round(_safe_number(row.get("y1")), 2),
                        "x2": round(_safe_number(row.get("x2")), 2),
                        "y2": round(_safe_number(row.get("y2")), 2),
                    },
                }
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82, optimize=True)
        timestamp = _safe_number(rows[0].get("timestamp"))
        result = {
            "task_id": int(task["id"]),
            "frame_id": int(frame_id),
            "timestamp": timestamp,
            "image_size": {"width": original_size[0], "height": original_size[1]},
            "image_mime": "image/jpeg",
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "detections": detections,
        }
        if detail_sheet:
            result["detail_image_mime"] = "image/jpeg"
            result["detail_image_base64"] = base64.b64encode(detail_sheet).decode("ascii")
        if detail_image_batches:
            result["detail_image_batches"] = detail_image_batches
        return result
    except Exception:
        return None
