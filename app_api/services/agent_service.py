from __future__ import annotations

from typing import Any

from app_api.services.statistics_service import (
    ALL_CLASSES,
    BEHAVIOR_LABELS,
    derive_overall_evidence,
    derive_segment_evidence,
)


AGENT_ROLES = [
    {"name": "行为证据汇总", "focus": "六类外显行为的数量、占比与时间分布", "color": "#0a84ff"},
    {"name": "情境一致性分析", "focus": "行为线索与课堂任务、教学环节的一致性", "color": "#30d158"},
    {"name": "教学反思建议", "focus": "关键片段复核与可执行的教学改进", "color": "#5e5ce6"},
]


def _review_segments(segments: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in segments:
        evidence = derive_segment_evidence(source)
        if not evidence["requires_review"]:
            continue
        rows.append({
            "time_range": source.get("time_range", ""),
            "review_priority": evidence["review_priority"],
            "review_reason": evidence["review_reason"],
            "review_cue_rate": evidence["review_cue_rate"],
            "dominant_behavior": BEHAVIOR_LABELS.get(evidence["dominant_behavior"], "暂无"),
        })
    rows.sort(key=lambda row: (row["review_priority"] == "高", row["review_cue_rate"]), reverse=True)
    return rows[:limit]


def _distribution_text(evidence: dict[str, Any]) -> str:
    distribution = evidence["behavior_distribution"]
    return "、".join(f"{BEHAVIOR_LABELS[name]} {distribution[name]:.2f}%" for name in ALL_CLASSES)


def _context_statement(course: dict[str, Any]) -> str:
    context = str(course.get("teaching_context") or "").strip()
    if context:
        return f"教师提供的课堂任务说明为“{context}”；行为含义仍需结合关键帧逐段确认。"
    return "尚未提供课堂任务说明；低头、书写、阅读和使用手机等行为均存在多种解释，需教师复核。"


def build_report_prompt(statistics: dict[str, Any]) -> str:
    return f"""
你是一名课堂行为证据分析助手。请只解释可观察行为，不测量或推断学生的内在认知专注状态。

数据：
{statistics}

请按“行为证据汇总、情境一致性分析、教学反思建议”三部分输出。每项判断必须引用行为数量、占比或时间段。
低头、书写、阅读、听课和使用手机均具有情境依赖性，不能固定归类为专注或不专注；信息不足时写明“需教师复核”。
不得进行个体排名、贴标签、纪律定性或惩罚性决策，不得编造未提供的数据。
""".strip()


def generate_rule_based_report(
    overall: dict[str, Any],
    segments: list[dict[str, Any]],
    course: dict[str, Any] | None = None,
    warnings: list[dict[str, str]] | None = None,
) -> dict[str, str]:
    course = course or {}
    warnings = warnings or []
    evidence = derive_overall_evidence(overall)
    review_segments = _review_segments(segments)
    dominant_label = BEHAVIOR_LABELS.get(evidence["dominant_behavior"], "暂无")
    course_name = str(course.get("course_name") or "本节课")
    review_text = "；".join(
        f"{item['time_range']}（{item['review_reason']}）" for item in review_segments
    ) or "当前没有由规则筛出的优先复核片段，仍建议进行关键帧抽样核对"
    summary = (
        f"{course_name} 共形成 {evidence['total_count']} 条可观察行为证据，覆盖 {evidence['class_count']} 类行为；"
        f"主要行为为{dominant_label}（{evidence['dominant_behavior_rate']:.2f}%）。"
        "这些结果反映外显行为分布，不代表学生真实认知专注状态。"
    )
    suggestion = (
        "先结合教学环节复核候选片段，再判断行为是否与课堂任务一致；"
        "对确认需要改进的片段，可调整讲授密度、任务说明或互动反馈，并在后续课堂使用同一行为口径复查。"
    )
    full_report = f"""
一、行为证据汇总
{summary}
六类行为分布：{_distribution_text(evidence)}。

二、情境一致性分析
{_context_statement(course)}
待复核片段：{review_text}。

三、教学反思建议
{suggestion}

解释边界
系统输出仅用于课堂过程回看与教师辅助判断，不用于个体排名、标签化评价或惩罚性决策。
""".strip()
    return {
        "summary": summary,
        "main_problem": f"待复核片段 {len(review_segments)} 个",
        "suggestion": suggestion,
        "full_report": full_report,
        "prompt": build_report_prompt({"overall": evidence, "segments": review_segments, "course": course, "warnings": warnings}),
    }


def generate_quality_report(
    overall: dict[str, Any],
    segments: list[dict[str, Any]],
    course: dict[str, Any] | None = None,
    warnings: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Describe evidence readiness, not teaching quality or student cognition."""
    course = course or {}
    evidence = derive_overall_evidence(overall)
    valid_segments = [derive_segment_evidence(item) for item in segments if derive_segment_evidence(item)["total_count"] > 0]
    category_coverage = round(evidence["class_count"] / len(ALL_CLASSES) * 100, 2)
    temporal_coverage = 100.0 if valid_segments else 0.0
    traceability = 100.0 if evidence["total_count"] and valid_segments else (50.0 if evidence["total_count"] else 0.0)
    context_ready = 100.0 if str(course.get("teaching_context") or "").strip() else 50.0
    completeness = round((category_coverage + temporal_coverage + traceability + context_ready) / 4, 2)
    dimensions = [
        {"name": "行为类别覆盖", "score": category_coverage, "evidence": f"检测到 {evidence['class_count']}/{len(ALL_CLASSES)} 类行为。"},
        {"name": "时序证据覆盖", "score": temporal_coverage, "evidence": f"形成 {len(valid_segments)} 个有效时间段。"},
        {"name": "片段可追溯性", "score": traceability, "evidence": "检测结果可关联时间段与关键帧。" if valid_segments else "尚无可追溯时间段。"},
        {"name": "情境信息完整性", "score": context_ready, "evidence": _context_statement(course)},
    ]
    return {
        "evidence_completeness": completeness,
        "quality_score": completeness,
        "summary": f"行为证据完整度 {completeness:.2f}%，该数值只描述证据覆盖，不评价教学质量或学生认知状态。",
        "dimension_scores": dimensions,
        "recommendations": [
            {"title": "补充课堂情境", "content": "记录本节课的讲授、练习、讨论和数字化任务时段，供行为线索核对。"},
            {"title": "复核关键片段", "content": "优先查看使用手机或睡觉类别所在时间段，并核对持续性、遮挡与误检。"},
            {"title": "形成改进记录", "content": "只对教师确认后的课堂现象制定改进措施，并在后续课堂按同一口径复查。"},
        ],
    }


def generate_multi_agent_analysis(
    overall: dict[str, Any],
    segments: list[dict[str, Any]],
    course: dict[str, Any] | None = None,
    warnings: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    course = course or {}
    evidence = derive_overall_evidence(overall)
    review_segments = _review_segments(segments)
    dominant_label = BEHAVIOR_LABELS.get(evidence["dominant_behavior"], "暂无")
    review_ranges = "、".join(str(item["time_range"]) for item in review_segments) or "无优先片段"
    agents = [
        {
            **AGENT_ROLES[0],
            "status": "已完成",
            "finding": f"共汇总 {evidence['total_count']} 条行为证据，覆盖 {evidence['class_count']} 类，主要行为为{dominant_label}。",
            "evidence": _distribution_text(evidence),
        },
        {
            **AGENT_ROLES[1],
            "status": "需教师复核" if not course.get("teaching_context") or review_segments else "已结合情境初步分析",
            "finding": _context_statement(course),
            "evidence": f"候选复核时间段：{review_ranges}。",
        },
        {
            **AGENT_ROLES[2],
            "status": "已生成",
            "finding": "建议先复核片段与课堂任务的一致性，再针对教师确认的问题调整活动设计并持续观察。",
            "evidence": "改进依据仅来自可追溯行为证据，不对学生内在状态作确定性判断。",
        },
    ]
    return {
        "agents": agents,
        "review_segments": review_segments,
        "consensus": (
            f"{course.get('course_name') or '本节课'} 已形成可回溯的行为分布、时间段和关键帧证据。"
            "系统不直接判断学生真实认知状态；所有情境解释与教学行动均需教师确认。"
        ),
    }


def behavior_distribution_rows(overall: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = derive_overall_evidence(overall)
    return [
        {
            "behavior": class_name,
            "label": BEHAVIOR_LABELS[class_name],
            "count": evidence["raw_counts"][class_name],
            "percentage": evidence["behavior_distribution"][class_name],
        }
        for class_name in ALL_CLASSES
    ]
