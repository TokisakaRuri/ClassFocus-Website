from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from app_api.core.env import load_local_env
from app_api.services.statistics_service import (
    BEHAVIOR_LABELS,
    derive_overall_evidence,
    derive_segment_evidence,
)

import requests


LOGGER = logging.getLogger(__name__)


def _compact_payload(
    overall: dict[str, Any],
    segments: list[dict[str, Any]],
    course: dict[str, Any] | None,
    warnings: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    course = course or {}
    warnings = warnings or []
    overall_evidence = derive_overall_evidence(overall)
    segment_evidence = []
    for segment in segments[:12]:
        evidence = derive_segment_evidence(segment)
        segment_evidence.append(
            {
                "time_range": segment.get("time_range"),
                "total_count": evidence["total_count"],
                "dominant_behavior": BEHAVIOR_LABELS.get(evidence["dominant_behavior"], "暂无"),
                "dominant_behavior_rate": evidence["dominant_behavior_rate"],
                "behavior_distribution": evidence["behavior_distribution"],
                "review_priority": evidence["review_priority"],
                "review_reason": evidence["review_reason"],
            }
        )
    return {
        "course": {
            "course_name": course.get("course_name"),
            "teacher_name": course.get("teacher_name"),
            "class_name": course.get("class_name"),
            "lesson_date": course.get("lesson_date"),
            "lesson_section": course.get("lesson_section"),
            "teaching_context": course.get("teaching_context"),
        },
        "overall": {
            "total_count": overall_evidence["total_count"],
            "class_count": overall_evidence["class_count"],
            "dominant_behavior": BEHAVIOR_LABELS.get(overall_evidence["dominant_behavior"], "暂无"),
            "dominant_behavior_rate": overall_evidence["dominant_behavior_rate"],
            "behavior_distribution": overall_evidence["behavior_distribution"],
        },
        "segments": segment_evidence,
        "warnings": warnings[:8],
        "interpretation_boundary": (
            "数据仅代表可观察行为。不得将任何行为固定映射为学生真实认知专注状态；"
            "信息不足时必须写明需教师复核。"
        ),
    }


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_llm_status() -> dict[str, Any]:
    """Return safe capability metadata without exposing credentials or endpoints."""
    load_local_env()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
    fallback_model = os.getenv("LLM_FALLBACK_MODEL", "").strip()
    vision_key = os.getenv("VISION_API_KEY") or api_key
    vision_model = (os.getenv("VISION_MODEL") or os.getenv("LLM_OCCLUSION_MODEL") or "").strip()
    vision_fallback = (os.getenv("VISION_FALLBACK_MODEL") or os.getenv("LLM_OCCLUSION_FALLBACK_MODEL") or "").strip()
    return {
        "enabled": bool(api_key),
        "model": model,
        "fallback_configured": bool(fallback_model and fallback_model != model),
        "text": {
            "configured": bool(api_key and model),
            "model": model,
            "fallback_configured": bool(fallback_model and fallback_model != model),
        },
        "vision": {
            "configured": bool(vision_key and vision_model),
            "model": vision_model,
            "fallback_configured": bool(vision_fallback),
        },
    }


def _request_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout_seconds: int,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1200,
    retry_attempts: int | None = None,
    response_format: dict[str, str] | None = None,
) -> str:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if not _bool_env("LLM_ENABLE_THINKING", False):
        body["enable_thinking"] = False
    if response_format:
        body["response_format"] = response_format
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if retry_attempts is None:
        try:
            retry_attempts = int(os.getenv("LLM_RETRY_ATTEMPTS", "2"))
        except ValueError:
            retry_attempts = 2
    retry_attempts = max(1, min(3, retry_attempts))
    retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
    last_error: Exception | None = None

    for attempt in range(retry_attempts):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=(10, timeout_seconds),
            )
            if response.status_code == 400:
                error_text = response.text.lower()
                changed = False
                if "enable_thinking" in error_text and "enable_thinking" in body:
                    body.pop("enable_thinking", None)
                    changed = True
                if "response_format" in error_text and "response_format" in body:
                    body.pop("response_format", None)
                    changed = True
            else:
                changed = False
            if changed:
                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=(10, timeout_seconds),
                )
            if response.status_code in retryable_statuses and attempt + 1 < retry_attempts:
                retry_after = response.headers.get("Retry-After", "")
                try:
                    delay = max(0.5, min(8.0, float(retry_after)))
                except (TypeError, ValueError):
                    delay = 1.2 * (2**attempt)
                time.sleep(delay)
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"模型服务返回 HTTP {response.status_code}")
            data = response.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise RuntimeError("模型返回结果不完整：缺少 choices")
            choice = choices[0]
            content = choice.get("message", {}).get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("模型未返回有效文本")
            if choice.get("finish_reason") == "length":
                raise RuntimeError("模型输出达到长度上限，结果被截断")
            return content.strip()
        except (requests.Timeout, requests.ConnectionError, requests.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 >= retry_attempts:
                break
            time.sleep(1.2 * (2**attempt))

    raise RuntimeError(str(last_error or "模型请求未完成"))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded_float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    value = _as_float(os.getenv(name), default)
    return max(minimum, min(maximum, value))


def _empty_occlusion_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for position, det in enumerate(payload.get("detections") or []):
        det_id = int(_as_float(det.get("id"), position))
        items.append(
            {
                "id": det_id,
                "occlusion_type": "",
                "occlusion_label": "无遮挡",
                "hg_assist": "",
                "confidence": 0.0,
                "reason": "Qwen未完成视觉遮挡判定",
            }
        )
    return items


def _extract_json_object(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _sanitize_occlusion_items(
    payload: dict[str, Any],
    raw_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    min_confidence = _bounded_float_env("VISION_OCCLUSION_MIN_CONFIDENCE", 0.35, 0.0, 1.0)
    expected_ids = [
        int(_as_float(det.get("id"), position))
        for position, det in enumerate(payload.get("detections") or [])
    ]
    if not raw_items:
        raise ValueError("视觉模型没有返回逐目标复核结果")
    if any(not isinstance(item, dict) or "id" not in item for item in raw_items):
        raise ValueError("视觉模型返回的目标结果格式不完整")
    by_id = {int(_as_float(item.get("id"), -1)): item for item in raw_items}
    if len(by_id) != len(raw_items):
        raise ValueError("视觉模型返回了重复的目标编号")
    missing_ids = [item_id for item_id in expected_ids if item_id not in by_id]
    extra_ids = [item_id for item_id in by_id if item_id not in expected_ids]
    if missing_ids or extra_ids:
        raise ValueError(f"视觉模型目标编号不完整（缺少 {len(missing_ids)}，多出 {len(extra_ids)}）")

    clean_items: list[dict[str, Any]] = []
    for position, det in enumerate(payload.get("detections") or []):
        det_id = int(_as_float(det.get("id"), position))
        item = by_id[det_id]
        if "occlusion_type" in item:
            raw_type = item.get("occlusion_type")
        elif "遮挡类型" in item:
            raw_type = item.get("遮挡类型")
        else:
            raise ValueError(f"目标 {det_id} 缺少遮挡类型字段")
        occlusion_type = str(raw_type or "").strip().upper()
        if occlusion_type in {"NONE", "NO", "NULL", "无遮挡", "无"}:
            occlusion_type = ""
        if occlusion_type not in {"S-S", "S-O"}:
            if occlusion_type:
                raise ValueError(f"目标 {det_id} 返回了未知遮挡类型")
            occlusion_type = ""
        reason = str(item.get("reason") or item.get("依据") or "").strip()
        if len(reason) < 4:
            raise ValueError(f"目标 {det_id} 缺少可核对的视觉依据")
        visibility_value = str(item.get("upper_body_visibility") or item.get("上半身可见性") or "").strip().lower()
        visibility_map = {
            "complete": "complete",
            "partial": "partial",
            "unclear": "unclear",
            "完整": "complete",
            "完整可见": "complete",
            "局部缺失": "partial",
            "部分缺失": "partial",
            "看不清": "unclear",
            "不确定": "unclear",
        }
        visibility = visibility_map.get(visibility_value, "")
        blocker_value = str(item.get("blocker") or item.get("遮挡物") or "").strip().lower()
        blocker_map = {
            "person": "person",
            "object": "object",
            "none": "none",
            "人体": "person",
            "学生": "person",
            "物体": "object",
            "无": "none",
            "无遮挡物": "none",
        }
        blocker = blocker_map.get(blocker_value, "")
        if not visibility or not blocker:
            raise ValueError(f"目标 {det_id} 缺少上半身可见性或遮挡物核对字段")
        confidence = _as_float(item.get("confidence"), 0.0)
        if confidence > 1:
            confidence /= 100.0
        confidence = max(0.0, min(1.0, confidence))
        if occlusion_type:
            expected_blocker = "person" if occlusion_type == "S-S" else "object"
            if visibility != "partial" or blocker != expected_blocker:
                raise ValueError(f"目标 {det_id} 的遮挡类型与视觉核对字段矛盾")
            if confidence < min_confidence:
                occlusion_type = ""
                reason = f"视觉证据置信度低于 {min_confidence:.2f}：{reason}"
        elif visibility == "partial" and blocker in {"person", "object"}:
            # Qwen occasionally identifies both the missing region and blocker but
            # leaves the redundant type field empty. Preserve that visual finding.
            occlusion_type = "S-S" if blocker == "person" else "S-O"
            if confidence < min_confidence:
                occlusion_type = ""
                reason = f"视觉证据置信度低于 {min_confidence:.2f}：{reason}"
        occlusion_label = "无遮挡"
        if occlusion_type == "S-S":
            occlusion_label = "学生-学生遮挡"
        elif occlusion_type == "S-O":
            occlusion_label = "学生-物体遮挡"
        clean_items.append(
            {
                "id": det_id,
                "occlusion_type": occlusion_type,
                "occlusion_label": occlusion_label,
                "hg_assist": "HG" if occlusion_type else "",
                "confidence": round(confidence, 3) if occlusion_type else 0.0,
                "reason": reason[:160],
                "upper_body_visibility": visibility,
                "blocker": blocker,
            }
        )
    return clean_items


def _occlusion_summary(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "occluded_count": sum(1 for item in items if item.get("occlusion_type")),
        "ss_count": sum(1 for item in items if item.get("occlusion_type") == "S-S"),
        "so_count": sum(1 for item in items if item.get("occlusion_type") == "S-O"),
        "hg_count": sum(1 for item in items if item.get("hg_assist")),
    }


def _occlusion_model_candidates(primary_model: str, fallback_models: str) -> list[str]:
    configured = (os.getenv("VISION_MODEL") or os.getenv("LLM_OCCLUSION_MODEL") or "").strip()
    raw_models = [
        configured or primary_model.strip(),
        *[item.strip() for item in fallback_models.split(",") if item.strip()],
    ]
    models: list[str] = []
    for item in raw_models:
        if item and item not in models:
            models.append(item)
    return models


def _bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _vision_review_batches(payload: dict[str, Any]) -> list[dict[str, Any]]:
    detections = payload.get("detections") or []
    by_id = {
        int(_as_float(detection.get("id"), position)): detection
        for position, detection in enumerate(detections)
    }
    batches: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    for raw_batch in payload.get("detail_image_batches") or []:
        if not isinstance(raw_batch, dict):
            continue
        ids = [int(_as_float(item_id, -1)) for item_id in raw_batch.get("ids") or []]
        batch_detections = [by_id[item_id] for item_id in ids if item_id in by_id and item_id not in used_ids]
        if not batch_detections:
            continue
        used_ids.update(int(_as_float(item.get("id"), -1)) for item in batch_detections)
        batches.append(
            {
                "detections": batch_detections,
                "detail_image_base64": str(raw_batch.get("image_base64") or "").strip(),
                "detail_image_mime": str(raw_batch.get("image_mime") or "image/jpeg").strip() or "image/jpeg",
            }
        )

    remaining = [
        detection
        for position, detection in enumerate(detections)
        if int(_as_float(detection.get("id"), position)) not in used_ids
    ]
    batch_size = _bounded_int_env("VISION_BATCH_SIZE", 6, 4, 16)
    legacy_detail = str(payload.get("detail_image_base64") or "").strip()
    legacy_mime = str(payload.get("detail_image_mime") or "image/jpeg").strip() or "image/jpeg"
    for start in range(0, len(remaining), batch_size):
        batches.append(
            {
                "detections": remaining[start : start + batch_size],
                "detail_image_base64": legacy_detail,
                "detail_image_mime": legacy_mime,
            }
        )
    return batches


def _build_occlusion_messages(
    payload: dict[str, Any],
    batch: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    batch_payload = {
        "frame": {
            "task_id": payload.get("task_id"),
            "frame_id": payload.get("frame_id"),
            "timestamp": payload.get("timestamp"),
            "image_size": payload.get("image_size"),
        },
        "detections": batch["detections"],
    }
    expected_ids = [
        int(_as_float(item.get("id"), position))
        for position, item in enumerate(batch["detections"])
    ]
    text_prompt = (
        "请联合查看课堂全景定位图和目标局部图，只复核本批目标。蓝框和编号仅用于定位，"
        "不得根据检测框重叠、目标距离、框大小、置信度或画面边缘截断推断遮挡。\n"
        "目标是尽可能找出真实遮挡，漏检的代价高于少量误报，请采用高召回率判定。上半身关键区域包括头发/头部、颈部、任一侧肩部、上臂近肩处和胸腹部；"
        "只要能看出前景人体或物体覆盖其中任一区域，令该处轮廓、衣服或身体部分在交界处中断或不可见，即使遮挡面积很小、目标仍大部分可见、"
        "只遮一侧肩部/上臂/胸腹部边缘，也应判为 partial 遮挡，不要求大面积缺失或多个部位同时被遮住。\n"
        "S-S 表示另一人体遮挡目标上半身，例如前排学生覆盖后排学生的肩部或胸部；"
        "S-O 表示桌面、显示器、书本、电脑、椅背、讲台、栏杆等物体遮挡目标上半身。只要存在合理的遮挡边界和前后层次，"
        "包括轻微或疑似但有视觉依据的真实覆盖，都应优先给出对应遮挡类型，并用较低 confidence 表达不确定程度，而不是直接判为无遮挡。\n"
        "低头、弯腰、侧身、背对镜头、只遮住手臂/腰部/下半身仍不算；多人靠近但没有真实覆盖也不算。"
        "仅当头、肩、躯干轮廓连续可见且没有任何外部覆盖证据时才判 complete/none；只有图像完全无法辨认人物边界时才使用 unclear。"
        "不要因为遮挡较轻、目标大部分仍可见或信心一般就判为无遮挡。\n"
        f"必须且只能返回这些 ID：{expected_ids}。每个 ID 恰好一项，依据应能从图像核对。"
        "只输出 JSON 对象，不要 Markdown："
        '{"items":[{"id":0,"upper_body_visibility":"complete|partial|unclear",'
        '"blocker":"person|object|none","occlusion_type":"S-S|S-O|",'
        '"confidence":0.0,"reason":"具体可见依据"}]}\n'
        f"本批结构化目标信息：{json.dumps(batch_payload, ensure_ascii=False)}"
    )
    image_base64 = str(payload.get("image_base64") or "").strip()
    image_mime = str(payload.get("image_mime") or "image/jpeg").strip() or "image/jpeg"
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": text_prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{image_mime};base64,{image_base64}", "detail": "high"},
        },
    ]
    detail_base64 = batch.get("detail_image_base64")
    if detail_base64:
        user_content.extend(
            [
                {"type": "text", "text": "以下局部图只包含本批 ID，请结合全景图逐项核对。"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{batch['detail_image_mime']};base64,{detail_base64}",
                        "detail": "high",
                    },
                },
            ]
        )
    return (
        [
            {
                "role": "system",
                "content": "你是课堂行为遮挡视觉复核助手。严格按给定 JSON 架构返回，不得遗漏、增加或重复目标。",
            },
            {"role": "user", "content": user_content},
        ],
        batch_payload,
    )


def _review_occlusion_batch(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    batch_payload: dict[str, Any],
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    format_attempts = _bounded_int_env("VISION_FORMAT_RETRY_ATTEMPTS", 2, 1, 2)
    current_messages = messages
    last_error: Exception | None = None
    for format_attempt in range(format_attempts):
        content = _request_chat_completion(
            base_url,
            api_key,
            model,
            current_messages,
            timeout_seconds,
            temperature=0.0,
            max_tokens=max(1000, min(2800, len(batch_payload["detections"]) * 180)),
            response_format={"type": "json_object"},
        )
        try:
            data = _extract_json_object(content)
            raw_items = data.get("items") if isinstance(data, dict) else []
            if not isinstance(raw_items, list):
                raw_items = []
            return _sanitize_occlusion_items(batch_payload, raw_items)
        except ValueError as exc:
            last_error = exc
            if format_attempt + 1 >= format_attempts:
                break
            expected_ids = [item.get("id") for item in batch_payload["detections"]]
            current_messages = [
                *messages,
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        f"上次 JSON 校验失败：{exc}。请重新检查图像，只返回 ID {expected_ids}，"
                        "补全 upper_body_visibility、blocker、occlusion_type、confidence、reason，且只输出 JSON。"
                    ),
                },
            ]
    raise ValueError(str(last_error or "视觉模型返回格式不完整"))


def assess_frame_occlusion(payload: dict[str, Any]) -> dict[str, Any]:
    load_local_env()
    api_key = os.getenv("VISION_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    image_base64 = str(payload.get("image_base64") or "").strip()
    detections = payload.get("detections") or []
    if not image_base64:
        items = _empty_occlusion_items(payload)
        return {
            "enabled": False,
            "used_llm": False,
            "model": "",
            "items": items,
            "summary": _occlusion_summary(items),
            "reason": "缺少单帧图像，未进行Qwen视觉遮挡判定。",
        }
    if not api_key:
        items = _empty_occlusion_items(payload)
        return {
            "enabled": False,
            "used_llm": False,
            "model": "",
            "items": items,
            "summary": _occlusion_summary(items),
            "reason": "未配置大模型密钥，未进行Qwen视觉遮挡判定。",
        }

    base_url = (
        os.getenv("VISION_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    text_model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "Qwen/Qwen3.5-9B"
    model = (os.getenv("VISION_MODEL") or os.getenv("LLM_OCCLUSION_MODEL") or "").strip()
    fallback_model = (
        os.getenv("VISION_FALLBACK_MODEL")
        or os.getenv("LLM_OCCLUSION_FALLBACK_MODEL")
        or ""
    ).strip()
    try:
        timeout_seconds = max(
            30,
            min(
                180,
                int(os.getenv("VISION_TIMEOUT_SECONDS") or os.getenv("LLM_OCCLUSION_TIMEOUT_SECONDS", "100")),
            ),
        )
    except ValueError:
        timeout_seconds = 100

    candidates_models = _occlusion_model_candidates(model or text_model, fallback_model)
    batches = _vision_review_batches(payload)
    if not batches:
        return {
            "enabled": True,
            "used_llm": False,
            "model": candidates_models[0] if candidates_models else model,
            "items": [],
            "summary": _occlusion_summary([]),
            "reviewed_count": 0,
            "batch_count": 0,
            "reason": "当前关键帧没有可复核的检测目标。",
        }

    errors: list[str] = []
    for current_model in candidates_models:
        try:
            items: list[dict[str, Any]] = []
            for batch in batches:
                batch_messages, batch_payload = _build_occlusion_messages(payload, batch)
                items.extend(
                    _review_occlusion_batch(
                        base_url,
                        api_key,
                        current_model,
                        batch_messages,
                        batch_payload,
                        timeout_seconds,
                    )
                )
            expected_ids = [
                int(_as_float(detection.get("id"), position))
                for position, detection in enumerate(detections)
            ]
            items_by_id = {int(item["id"]): item for item in items}
            if set(items_by_id) != set(expected_ids):
                raise ValueError("分批复核结果未覆盖全部目标")
            items = [items_by_id[item_id] for item_id in expected_ids]
            return {
                "enabled": True,
                "used_llm": True,
                "mode": "vision-review",
                "model": current_model,
                "primary_model": candidates_models[0],
                "used_fallback": current_model != candidates_models[0],
                "items": items,
                "summary": _occlusion_summary(items),
                "reviewed_count": len(items),
                "batch_count": len(batches),
            }
        except Exception as exc:
            errors.append(f"{current_model}: {exc}")
            LOGGER.warning("Vision review failed for %s: %s", current_model, exc)

    items = _empty_occlusion_items(payload)
    return {
        "enabled": False,
        "used_llm": False,
        "model": candidates_models[0] if candidates_models else model,
        "fallback_model": fallback_model,
        "items": items,
        "summary": _occlusion_summary(items),
        "error_code": "vision_review_failed",
        "reason": "视觉复核未完成：Qwen3.5-9B 主模型连接超时、限流或返回结果不完整，可稍后再次复核。",
        "attempted_models": candidates_models,
    }


def _normalize_diagnosis_content(content: str) -> str:
    required_sections = ("行为证据汇总", "情境一致性分析", "教学反思建议")
    data = _extract_json_object(content)
    if data:
        aliases = (
            ("behavior_evidence", "行为证据汇总"),
            ("context_consistency", "情境一致性分析"),
            ("teaching_reflection", "教学反思建议"),
        )
        values: list[str] = []
        for english_key, chinese_key in aliases:
            value = str(data.get(english_key) or data.get(chinese_key) or "").strip()
            if not value:
                raise ValueError(f"返回 JSON 缺少 {english_key}")
            values.append(value)
        return "\n\n".join(
            f"{title}：{value}"
            for title, value in zip(required_sections, values, strict=True)
        )

    positions = [content.find(section) for section in required_sections]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise ValueError("返回内容缺少必要分析分段")
    return content.strip()


def generate_llm_diagnosis(
    overall: dict[str, Any],
    segments: list[dict[str, Any]],
    course: dict[str, Any] | None = None,
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    load_local_env()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        return {"enabled": False, "reason": "未配置大模型密钥"}

    base_url = (os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
    fallback_model = os.getenv("LLM_FALLBACK_MODEL", "").strip()
    try:
        timeout_seconds = max(30, int(os.getenv("LLM_TIMEOUT_SECONDS", "150")))
    except ValueError:
        timeout_seconds = 150
    payload = _compact_payload(overall, segments, course, warnings)
    messages = [
        {
            "role": "system",
            "content": (
                "你是课堂行为证据分析助手。请严格依据用户提供的数据，用中文输出简洁、可追溯的辅助分析，"
                "不得编造检测结果，也不得推断学生内在认知状态。只输出 JSON 对象，字段必须为 "
                "behavior_evidence、context_consistency、teaching_reflection；每个字段为一段中文，总字数控制在450字内。"
                "每项判断必须引用行为数量、占比或时间段。"
                "行为标签必须翻译为中文：listening=听课，writing=书写，reading=阅读，"
                "using phone=使用手机，bowing the head=低头，sleeping=睡觉。"
                "听课、书写、阅读、低头和使用手机都具有情境依赖性，不能固定划分为专注或不专注；"
                "睡觉类别也需核对持续性、遮挡和误检。信息不足时明确写“需教师复核”。"
                "不得进行个体排名、贴标签、纪律定性或惩罚性决策。建议需说明行动、适用时段与可观察复查依据。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]

    candidates = [model]
    if fallback_model and fallback_model != model:
        candidates.append(fallback_model)

    errors: list[str] = []
    for current_model in candidates:
        current_messages = messages
        format_attempts = _bounded_int_env("LLM_FORMAT_RETRY_ATTEMPTS", 2, 1, 2)
        for format_attempt in range(format_attempts):
            try:
                raw_content = _request_chat_completion(
                    base_url,
                    api_key,
                    current_model,
                    current_messages,
                    timeout_seconds,
                    max_tokens=1600,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                errors.append(f"{current_model}：{exc}")
                LOGGER.warning("Text analysis failed for %s: %s", current_model, exc)
                break
            try:
                content = _normalize_diagnosis_content(raw_content)
            except ValueError as exc:
                errors.append(f"{current_model}：{exc}")
                LOGGER.warning("Text analysis format validation failed for %s: %s", current_model, exc)
                if format_attempt + 1 >= format_attempts:
                    break
                current_messages = [
                    *messages,
                    {"role": "assistant", "content": raw_content},
                    {
                        "role": "user",
                        "content": (
                            "上次输出未通过格式校验。请保留原有分析含义并重新输出完整 JSON，且只包含 "
                            "behavior_evidence、context_consistency、teaching_reflection 三个非空字符串字段。"
                        ),
                    },
                ]
                continue
            return {
                "enabled": True,
                "model": current_model,
                "primary_model": model,
                "used_fallback": current_model != model,
                "content": content,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
    return {
        "enabled": False,
        "model": model,
        "fallback_model": fallback_model,
        "error_code": "text_analysis_failed",
        "reason": "辅助分析未完成：Qwen3.5-9B 主模型连接超时、限流或返回内容不完整，可稍后重新生成。",
        "attempted_models": candidates,
    }
