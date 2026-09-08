from __future__ import annotations

import re
from dataclasses import dataclass

from src.privacy import redact_sensitive_info
from src.schemas import (
    InterviewCopilotGuidance,
    InterviewPreparation,
    InterviewQuestion,
    MatchAnalysis,
    ResumeProfile,
)


MAX_EVIDENCE_ITEMS = 12


@dataclass(frozen=True)
class InterviewEvidence:
    id: str
    source: str
    text: str


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _number_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?%?", text))


def collect_interview_evidence(
    question: InterviewQuestion,
    resume_profile: ResumeProfile,
    analysis: MatchAnalysis,
) -> list[InterviewEvidence]:
    """Collect a small, deterministic evidence set without storing raw documents."""
    candidates: list[tuple[str, str]] = []
    related_ids = set(question.requirement_ids)
    for match in analysis.matches:
        if match.requirement_id in related_ids:
            if match.evidence:
                candidates.extend(
                    (
                        "用户确认"
                        if item.source.value == "user_confirmed"
                        else f"简历·岗位要求 {match.requirement_id}",
                        item.text,
                    )
                    for item in match.evidence
                )
            else:
                candidates.extend(
                    (f"简历·岗位要求 {match.requirement_id}", item)
                    for item in match.resume_evidence
                )
    candidates.extend(
        (item.source_section or "简历", item.text)
        for item in resume_profile.evidence_chunks
    )

    seen: set[str] = set()
    evidence: list[InterviewEvidence] = []
    for source, raw_text in candidates:
        text = redact_sensitive_info(raw_text).strip()
        key = _normalise(text)
        if not key or key in seen:
            continue
        seen.add(key)
        evidence.append(
            InterviewEvidence(
                id=f"ev_{len(evidence) + 1:03d}",
                source=source,
                text=text,
            )
        )
        if len(evidence) >= MAX_EVIDENCE_ITEMS:
            break
    return evidence


def collect_copilot_evidence(
    resume_profile: ResumeProfile,
    analysis: MatchAnalysis,
) -> list[InterviewEvidence]:
    """Collect a small evidence set for live prompts without retaining audio."""
    synthetic_question = InterviewQuestion(
        category="job_knowledge",
        question="实时面试问题",
        why_asked="为即时提示收集可核对证据。",
        answer_outline=[],
        requirement_ids=[item.requirement_id for item in analysis.matches],
    )
    return collect_interview_evidence(synthetic_question, resume_profile, analysis)


def sanitise_copilot_guidance(
    guidance: InterviewCopilotGuidance,
    evidence: list[InterviewEvidence],
) -> InterviewCopilotGuidance:
    valid_ids = {item.id for item in evidence}
    evidence_ids = list(
        dict.fromkeys(
            identifier for identifier in guidance.evidence_ids if identifier in valid_ids
        )
    )
    selected_text = " ".join(
        item.text for item in evidence if item.id in set(evidence_ids)
    )
    allowed_numbers = _number_tokens(selected_text)
    talking_points: list[str] = []
    removed_number = False
    for point in guidance.talking_points:
        if not _number_tokens(point).issubset(allowed_numbers):
            removed_number = True
            continue
        talking_points.append(redact_sensitive_info(point).strip())
    cautions = [redact_sensitive_info(item).strip() for item in guidance.caution_notes]
    if removed_number:
        cautions.append("已移除证据中不存在的数字，请只使用本人可核对数据。")
    return guidance.model_copy(
        update={
            "detected_question": redact_sensitive_info(
                guidance.detected_question
            ).strip(),
            "answer_framework": [
                redact_sensitive_info(item).strip()
                for item in guidance.answer_framework
                if item.strip()
            ],
            "talking_points": [item for item in talking_points if item],
            "evidence_ids": evidence_ids,
            "missing_information": [
                redact_sensitive_info(item).strip()
                for item in guidance.missing_information
                if item.strip()
            ],
            "caution_notes": list(dict.fromkeys(item for item in cautions if item)),
        }
    )


def sanitise_interview_preparation(
    preparation: InterviewPreparation,
    evidence: list[InterviewEvidence],
) -> InterviewPreparation:
    valid_ids = {item.id for item in evidence}
    selected_ids = list(dict.fromkeys(
        identifier for identifier in preparation.evidence_ids if identifier in valid_ids
    ))
    selected_text = " ".join(
        item.text for item in evidence if item.id in set(selected_ids)
    )
    answer = preparation.personalized_answer
    missing = list(preparation.missing_information)

    if not selected_ids:
        answer = None
        if "个人资料中没有足够证据，无法生成个性化回答。" not in missing:
            missing.append("个人资料中没有足够证据，无法生成个性化回答。")
    elif answer and not _number_tokens(answer).issubset(_number_tokens(selected_text)):
        answer = None
        missing.append("回答草稿包含证据中不存在的数字，已停止展示，请补充真实信息。")

    return preparation.model_copy(
        update={
            "personalized_answer": answer,
            "evidence_ids": selected_ids,
            "missing_information": list(dict.fromkeys(missing)),
        }
    )
