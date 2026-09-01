from __future__ import annotations

from collections import defaultdict
from typing import Any


ALL_CLASSES = [
    "listening",
    "writing",
    "reading",
    "using phone",
    "bowing the head",
    "sleeping",
]

CLASS_COUNT_KEYS = {
    "listening": "listening_count",
    "writing": "writing_count",
    "reading": "reading_count",
    "using phone": "using_phone_count",
    "bowing the head": "bowing_head_count",
    "sleeping": "sleeping_count",
}

BEHAVIOR_LABELS = {
    "listening": "听课",
    "writing": "书写",
    "reading": "阅读",
    "using phone": "使用手机",
    "bowing the head": "低头",
    "sleeping": "睡觉",
}

# These classes are review cues only. They are never treated as measurements of
# a student's internal cognitive state.
REVIEW_CUE_CLASSES = {"using phone", "sleeping"}
PHONE_REVIEW_RATE = 10.0
SLEEPING_REVIEW_RATE = 3.0


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    return f"{minutes:02d}:{sec:02d}"


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _counts_from_payload(payload: dict[str, Any]) -> dict[str, int]:
    raw_counts = payload.get("raw_counts") if isinstance(payload.get("raw_counts"), dict) else {}
    return {
        class_name: _safe_int(raw_counts.get(class_name, payload.get(CLASS_COUNT_KEYS[class_name], 0)))
        for class_name in ALL_CLASSES
    }


def _distribution_from_counts(counts: dict[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {class_name: 0 for class_name in ALL_CLASSES}
    return {
        class_name: round(counts.get(class_name, 0) / total * 100, 2)
        for class_name in ALL_CLASSES
    }


def derive_overall_evidence(overall: dict[str, Any]) -> dict[str, Any]:
    """Normalize current and historical results into validity-safe evidence metrics."""
    counts = _counts_from_payload(overall)
    counted_total = sum(counts.values())
    total = counted_total or _safe_int(overall.get("total_count"))
    distribution = _distribution_from_counts(counts, total)
    if not counted_total and isinstance(overall.get("behavior_distribution"), dict):
        distribution = {
            class_name: round(float(overall["behavior_distribution"].get(class_name, 0) or 0), 2)
            for class_name in ALL_CLASSES
        }

    present = [class_name for class_name in ALL_CLASSES if counts.get(class_name, 0) > 0 or distribution[class_name] > 0]
    dominant = max(ALL_CLASSES, key=lambda name: distribution.get(name, 0)) if present else ""
    review_cue_count = sum(counts.get(class_name, 0) for class_name in REVIEW_CUE_CLASSES)
    return {
        "total_count": total,
        "class_count": len(present),
        "dominant_behavior": dominant,
        "dominant_behavior_rate": round(distribution.get(dominant, 0), 2) if dominant else 0,
        "review_cue_count": review_cue_count,
        "review_cue_rate": round(review_cue_count / total * 100, 2) if total else 0,
        "evidence_status": "行为证据已生成" if total else "暂无行为证据",
        "behavior_distribution": distribution,
        "raw_counts": counts,
    }


def derive_segment_evidence(segment: dict[str, Any]) -> dict[str, Any]:
    counts = _counts_from_payload(segment)
    counted_total = sum(counts.values())
    total = counted_total or _safe_int(segment.get("total_count"))
    distribution = _distribution_from_counts(counts, total)
    present = [class_name for class_name in ALL_CLASSES if counts.get(class_name, 0) > 0]
    dominant = max(ALL_CLASSES, key=lambda name: counts.get(name, 0)) if present else ""
    phone_count = counts["using phone"]
    sleeping_count = counts["sleeping"]
    review_cue_count = phone_count + sleeping_count
    phone_rate = round(phone_count / total * 100, 2) if total else 0
    sleeping_rate = round(sleeping_count / total * 100, 2) if total else 0

    if total <= 0:
        priority = "无数据"
        reason = "当前时间段没有有效检测结果"
    elif sleeping_rate >= SLEEPING_REVIEW_RATE:
        priority = "高"
        reason = f"睡觉类别占比 {sleeping_rate:.2f}%，需结合持续性、遮挡情况与关键帧由教师复核"
    elif phone_rate >= PHONE_REVIEW_RATE:
        priority = "待复核"
        reason = f"使用手机类别占比 {phone_rate:.2f}%，需结合课堂任务与关键帧由教师复核"
    else:
        priority = "常规"
        reason = "未发现需要优先复核的行为线索"

    return {
        "total_count": total,
        "class_count": len(present),
        "dominant_behavior": dominant,
        "dominant_behavior_rate": round(counts.get(dominant, 0) / total * 100, 2) if total and dominant else 0,
        "review_cue_count": review_cue_count,
        "review_cue_rate": round(review_cue_count / total * 100, 2) if total else 0,
        "review_priority": priority,
        "review_reason": reason,
        "requires_review": priority in {"高", "待复核"},
        "behavior_distribution": distribution,
        "raw_counts": counts,
    }


def aggregate_by_segment(
    detections: list[dict[str, Any]],
    segment_seconds: int = 60,
    duration: float | None = None,
) -> list[dict[str, Any]]:
    segment_map: dict[int, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for detection in detections:
        timestamp = float(detection.get("timestamp", 0) or 0)
        label = str(detection.get("label", ""))
        if label in ALL_CLASSES:
            segment_map[int(timestamp // segment_seconds)][label] += 1

    if duration and duration > 0:
        last_segment = int(max(0, duration - 0.001) // segment_seconds)
        for segment_id in range(last_segment + 1):
            segment_map[segment_id]

    results: list[dict[str, Any]] = []
    for segment_id in sorted(segment_map):
        start_time = segment_id * segment_seconds
        end_time = (segment_id + 1) * segment_seconds
        base: dict[str, Any] = {
            "start_time": start_time,
            "end_time": end_time,
            "time_range": f"{format_seconds(start_time)}-{format_seconds(end_time)}",
        }
        for class_name in ALL_CLASSES:
            base[CLASS_COUNT_KEYS[class_name]] = segment_map[segment_id][class_name]
        evidence = derive_segment_evidence(base)
        results.append({**base, **evidence})
    return results


def calculate_overall_statistics(detections: list[dict[str, Any]]) -> dict[str, Any]:
    counts: defaultdict[str, int] = defaultdict(int)
    for detection in detections:
        label = str(detection.get("label", ""))
        if label in ALL_CLASSES:
            counts[label] += 1
    return derive_overall_evidence({"raw_counts": dict(counts)})


def detect_warnings(overall: dict[str, Any], segments: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return review prompts, never cognitive-state or discipline conclusions."""
    normalized_segments = [derive_segment_evidence(segment) | {
        "time_range": str(segment.get("time_range") or "")
    } for segment in segments]
    sleeping_ranges = [
        item["time_range"] for item in normalized_segments
        if item["behavior_distribution"]["sleeping"] >= SLEEPING_REVIEW_RATE
    ]
    phone_ranges = [
        item["time_range"] for item in normalized_segments
        if item["behavior_distribution"]["using phone"] >= PHONE_REVIEW_RATE
    ]
    prompts: list[dict[str, str]] = []

    if sleeping_ranges:
        prompts.append({
            "type": "高优先级片段复核",
            "priority": "高",
            "detail": f"{_join_ranges(sleeping_ranges)} 检测到睡觉类别；需核对持续性、遮挡和误检，不直接推断认知状态。",
        })
    if phone_ranges:
        prompts.append({
            "type": "课堂任务一致性复核",
            "priority": "待复核",
            "detail": f"{_join_ranges(phone_ranges)} 检测到使用手机类别；需结合课堂任务（如扫码答题、资料查询）解释。",
        })

    evidence = derive_overall_evidence(overall)
    if evidence["total_count"] and not prompts:
        prompts.append({
            "type": "常规抽样复核",
            "priority": "常规",
            "detail": "当前未发现优先复核线索，建议抽样核对关键帧与课堂任务记录。",
        })
    return prompts


def _join_ranges(ranges: list[str], limit: int = 4) -> str:
    selected = [item for item in ranges if item][:limit]
    text = "、".join(selected) or "相关时间段"
    return text + ("等时间段" if len(ranges) > limit else "")
