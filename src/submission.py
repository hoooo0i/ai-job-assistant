from __future__ import annotations

import re

from src.career_tools import contains_unresolved_placeholder
from src.schemas import (
    CandidateFact,
    JobProfile,
    MatchAnalysis,
    SubmissionCheckItem,
    SubmissionChecklist,
)


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?%?", text))


def build_submission_checklist(
    analysis: MatchAnalysis,
    decisions: dict[str, dict[str, str]],
    job_profile: JobProfile,
    candidate_facts: list[CandidateFact],
) -> SubmissionChecklist:
    items: list[SubmissionCheckItem] = []

    def add(code: str, passed: bool, blocking: bool, label: str, detail: str) -> None:
        items.append(
            SubmissionCheckItem(
                code=code,
                passed=passed,
                blocking=blocking,
                label=label,
                detail=detail,
            )
        )

    accepted_records = [
        (analysis.resume_suggestions[index], record)
        for index, suggestion in enumerate(analysis.resume_suggestions)
        if (record := decisions.get(str(index), {})).get("decision") == "accepted"
    ]
    add(
        "accepted",
        bool(accepted_records),
        False,
        "已选择简历修改",
        f"当前采纳 {len(accepted_records)} 条建议。" if accepted_records else "尚未采纳任何建议，可继续使用原简历。",
    )

    placeholder_count = sum(
        contains_unresolved_placeholder(record.get("text", ""))
        for _, record in accepted_records
    )
    add(
        "placeholders",
        placeholder_count == 0,
        True,
        "无待填写占位符",
        "所有已采纳内容均已填写完整。" if not placeholder_count else f"有 {placeholder_count} 条内容仍包含占位符。",
    )

    match_by_id = {item.requirement_id: item for item in analysis.matches}
    requirement_text_by_id = {
        item.id: item.original_text for item in job_profile.requirements
    }
    unsupported = 0
    for suggestion, record in accepted_records:
        related_matches = [
            match_by_id[identifier]
            for identifier in suggestion.requirement_ids
            if identifier in match_by_id
        ]
        related_requirement_texts = {
            requirement_text_by_id[identifier]
            for identifier in suggestion.requirement_ids
            if identifier in requirement_text_by_id
        }
        source_text = " ".join(
            [
                suggestion.original_text,
                *(item.text for match in related_matches for item in match.evidence),
                *(item for match in related_matches for item in match.resume_evidence),
                *(
                    " ".join(filter(None, [fact.statement, fact.metrics]))
                    for fact in candidate_facts
                    if fact.source_requirement_text in related_requirement_texts
                ),
            ]
        )
        unsupported += int(
            not _numbers(record.get("text", "")).issubset(_numbers(source_text))
        )
    add(
        "numbers",
        unsupported == 0,
        True,
        "数字均有来源",
        "未发现新增的无来源数字。" if not unsupported else f"有 {unsupported} 条已采纳内容包含证据中未出现的数字。",
    )

    pending = sum(
        decisions.get(str(index), {}).get("decision", "pending") == "pending"
        for index, _ in enumerate(analysis.resume_suggestions)
    )
    add(
        "reviewed",
        pending == 0,
        False,
        "建议已逐条处理",
        "所有建议均已处理。" if not pending else f"仍有 {pending} 条建议待决定。",
    )

    metadata_ok = bool(job_profile.company.strip() and job_profile.title.strip())
    add(
        "target",
        metadata_ok,
        True,
        "目标岗位正确",
        f"{job_profile.company} · {job_profile.title}" if metadata_ok else "公司或岗位名称缺失。",
    )
    ready = not any(not item.passed and item.blocking for item in items)
    return SubmissionChecklist(ready=ready, items=items)


def safe_resume_filename(job_profile: JobProfile) -> str:
    raw = f"tailored-resume-{job_profile.company}-{job_profile.title}"
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", raw).strip("-")
    return f"{cleaned[:90] or 'tailored-resume'}.docx"
