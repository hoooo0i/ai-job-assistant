from __future__ import annotations

import hashlib
import re

from src.privacy import redact_sensitive_info
from src.schemas import (
    CandidateFact,
    ClarificationAnswer,
    ClarificationQuestion,
    JobProfile,
    JobRequirement,
    MatchAnalysis,
    MatchEvidence,
    MatchStatus,
    PreliminaryAnalysis,
    RequirementCategory,
    RequirementImportance,
    SupplementDetail,
)


MAX_CLARIFICATION_QUESTIONS = 5
MIN_SUPPLEMENT_CHARACTERS = 20
DIRECT_CONFIRMATION_CATEGORIES = {
    RequirementCategory.availability,
    RequirementCategory.location,
}


class EvidenceFlowError(ValueError):
    """A safe validation error for candidate-supplied evidence."""


def _non_whitespace_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _allows_direct_confirmation(requirement: JobRequirement) -> bool:
    return (
        requirement.category in DIRECT_CONFIRMATION_CATEGORIES
        or requirement.category is RequirementCategory.other
        and requirement.is_hard_condition
    )


def _question_priority(question: ClarificationQuestion, job: JobProfile) -> tuple[int, int, str]:
    requirement_by_id = {item.id: item for item in job.requirements}
    requirement = requirement_by_id[question.requirement_id]
    if requirement.is_hard_condition:
        group = 0
    elif requirement.importance is RequirementImportance.must_have:
        group = 1
    elif requirement.importance is RequirementImportance.preferred:
        group = 2
    else:
        group = 3
    return group, job.requirements.index(requirement), question.id


def prepare_clarification_questions(
    preliminary: PreliminaryAnalysis,
    job: JobProfile,
) -> list[ClarificationQuestion]:
    """Keep valid unresolved questions, fill omissions, then apply a stable priority limit."""
    requirement_by_id = {item.id: item for item in job.requirements}
    match_by_id = {item.requirement_id: item for item in preliminary.matches}
    unresolved_ids = {
        item.requirement_id
        for item in preliminary.matches
        if item.status in {MatchStatus.partial, MatchStatus.missing, MatchStatus.unknown}
    }
    questions: list[ClarificationQuestion] = []
    seen_requirements: set[str] = set()
    seen_question_ids: set[str] = set()
    for question in preliminary.clarification_questions:
        if (
            question.requirement_id not in unresolved_ids
            or question.requirement_id not in requirement_by_id
            or question.requirement_id in seen_requirements
            or question.id in seen_question_ids
            or not question.prompt.strip()
        ):
            continue
        seen_requirements.add(question.requirement_id)
        seen_question_ids.add(question.id)
        questions.append(question)

    for requirement in job.requirements:
        if requirement.id not in unresolved_ids or requirement.id in seen_requirements:
            continue
        match = match_by_id[requirement.id]
        question_id = f"cq_{requirement.id}"
        suffix = 2
        while question_id in seen_question_ids:
            question_id = f"cq_{requirement.id}_{suffix}"
            suffix += 1
        questions.append(
            ClarificationQuestion(
                id=question_id,
                requirement_id=requirement.id,
                prompt=(
                    f"你是否实际满足“{requirement.original_text}”？"
                    f"如果具备，请补充简历中未写清的真实经历。"
                    f"当前判断：{match.explanation}"
                ),
            )
        )
        seen_question_ids.add(question_id)
    return sorted(questions, key=lambda item: _question_priority(item, job))[
        :MAX_CLARIFICATION_QUESTIONS
    ]


def validate_clarification_answers(
    answers: list[ClarificationAnswer],
    questions: list[ClarificationQuestion],
    job: JobProfile,
) -> list[ClarificationAnswer]:
    question_by_id = {item.id: item for item in questions}
    requirement_by_id = {item.id: item for item in job.requirements}
    validated: list[ClarificationAnswer] = []
    seen: set[str] = set()
    for answer in answers:
        question = question_by_id.get(answer.question_id)
        if question is None or question.requirement_id != answer.requirement_id:
            raise EvidenceFlowError("补充回答与岗位要求不匹配，请刷新后重试。")
        if answer.question_id in seen:
            raise EvidenceFlowError("同一个补充问题不能重复提交。")
        seen.add(answer.question_id)
        evidence_text = redact_sensitive_info(answer.evidence_text).strip()
        metrics = redact_sensitive_info(answer.metrics or "").strip() or None
        validated.append(
            answer.model_copy(update={"evidence_text": evidence_text, "metrics": metrics})
        )
    return validated


def facts_from_answers(
    answers: list[ClarificationAnswer],
    job: JobProfile,
    job_id: str,
) -> list[CandidateFact]:
    requirement_by_id = {item.id: item for item in job.requirements}
    facts: list[CandidateFact] = []
    for answer in answers:
        if answer.status != "have":
            continue
        requirement = requirement_by_id[answer.requirement_id]
        statement = answer.evidence_text.strip() or f"用户确认满足：{requirement.original_text}"
        identity = "\x00".join(
            [requirement.category.value, statement, answer.metrics or ""]
        )
        fact_id = "fact_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        facts.append(
            CandidateFact(
                id=fact_id,
                category=requirement.category,
                statement=statement,
                metrics=answer.metrics,
                source_job_id=job_id,
                source_requirement_text=requirement.original_text,
            )
        )
    return facts


def merge_candidate_facts(
    existing: list[CandidateFact],
    replacements: list[CandidateFact],
    job_id: str,
) -> list[CandidateFact]:
    """Replace facts originating from the active job while preserving reusable facts."""
    kept = [item for item in existing if item.source_job_id != job_id]
    merged = kept + replacements
    return list({item.id: item for item in merged}.values())


def select_important_supplements(
    job: JobProfile,
    analysis: MatchAnalysis,
    answers: list[ClarificationAnswer],
    limit: int = MAX_CLARIFICATION_QUESTIONS,
) -> list[JobRequirement]:
    """Select high-impact, truthful gaps that could benefit from richer evidence."""
    matches = {item.requirement_id: item for item in analysis.matches}
    answer_by_requirement = {item.requirement_id: item for item in answers}
    candidates: list[JobRequirement] = []
    for requirement in job.requirements:
        if _allows_direct_confirmation(requirement):
            continue
        answer = answer_by_requirement.get(requirement.id)
        if answer and answer.status in {"not_have", "unsure"}:
            continue
        match = matches.get(requirement.id)
        if match is None or match.status is not MatchStatus.partial:
            continue
        candidates.append(requirement)

    def priority(requirement: JobRequirement) -> tuple[int, int]:
        if requirement.is_hard_condition:
            group = 0
        elif requirement.importance is RequirementImportance.must_have:
            group = 1
        elif requirement.importance is RequirementImportance.preferred:
            group = 2
        else:
            group = 3
        return group, job.requirements.index(requirement)

    return sorted(candidates, key=priority)[:limit]


def sanitise_supplement_drafts(
    details: list[SupplementDetail],
    job: JobProfile,
    allowed_requirement_ids: set[str],
    *,
    require_complete: bool = False,
) -> list[SupplementDetail]:
    """Redact saved drafts and validate completed drafts before an AI rewrite."""
    requirement_by_id = {item.id: item for item in job.requirements}
    cleaned: list[SupplementDetail] = []
    seen: set[str] = set()
    for detail in details:
        if (
            detail.requirement_id not in requirement_by_id
            or detail.requirement_id not in allowed_requirement_ids
        ):
            raise EvidenceFlowError("重点补充内容与岗位要求不匹配，请刷新后重试。")
        if detail.requirement_id in seen:
            raise EvidenceFlowError("同一个岗位要求不能重复提交补充内容。")
        seen.add(detail.requirement_id)
        updated = detail.model_copy(
            update={
                "situation": redact_sensitive_info(detail.situation).strip(),
                "action": redact_sensitive_info(detail.action).strip(),
                "result": redact_sensitive_info(detail.result).strip(),
                "metrics": redact_sensitive_info(detail.metrics or "").strip() or None,
            }
        )
        combined = " ".join(
            filter(None, [updated.situation, updated.action, updated.result, updated.metrics])
        )
        if not combined:
            continue
        if require_complete and _non_whitespace_length(combined) < MIN_SUPPLEMENT_CHARACTERS:
            requirement = requirement_by_id[detail.requirement_id]
            raise EvidenceFlowError(
                f"“{requirement.normalized_name}”的补充内容至少需要 "
                f"{MIN_SUPPLEMENT_CHARACTERS} 个非空白字符。"
            )
        cleaned.append(updated)
    if require_complete and not cleaned:
        raise EvidenceFlowError("请至少填写一项真实经历后再进行 AI 批量优化。")
    return cleaned


def facts_from_supplement_details(
    details: list[SupplementDetail],
    job: JobProfile,
    job_id: str,
) -> list[CandidateFact]:
    requirement_by_id = {item.id: item for item in job.requirements}
    facts: list[CandidateFact] = []
    for detail in details:
        requirement = requirement_by_id[detail.requirement_id]
        parts = [
            f"情境：{detail.situation}" if detail.situation else "",
            f"行动：{detail.action}" if detail.action else "",
            f"结果：{detail.result}" if detail.result else "",
        ]
        statement = "；".join(filter(None, parts))
        identity = "\x00".join(
            [requirement.category.value, statement, detail.metrics or ""]
        )
        facts.append(
            CandidateFact(
                id="fact_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12],
                category=requirement.category,
                statement=statement,
                metrics=detail.metrics,
                source_job_id=job_id,
                source_requirement_text=requirement.original_text,
            )
        )
    return facts


def apply_answers_to_final_analysis(
    analysis: MatchAnalysis,
    answers: list[ClarificationAnswer],
    facts: list[CandidateFact],
    job: JobProfile,
    preliminary: PreliminaryAnalysis | None = None,
) -> MatchAnalysis:
    answer_by_requirement = {item.requirement_id: item for item in answers}
    fact_by_requirement = {
        item.source_requirement_text: item for item in facts if item.user_confirmed
    }
    requirement_by_id = {item.id: item for item in job.requirements}
    preliminary_by_id = (
        {item.requirement_id: item for item in preliminary.matches}
        if preliminary is not None
        else {}
    )
    updated = []
    for match in analysis.matches:
        answer = answer_by_requirement.get(match.requirement_id)
        if answer is None or answer.status == "unanswered":
            updated.append(preliminary_by_id.get(match.requirement_id, match))
            continue
        requirement = requirement_by_id[match.requirement_id]
        if answer.status == "not_have":
            updated.append(
                match.model_copy(
                    update={
                        "status": MatchStatus.missing,
                        "evidence": [],
                        "resume_evidence": [],
                        "explanation": "用户已确认目前不具备该条件。",
                        "confidence": 1.0,
                    }
                )
            )
            continue
        if answer.status == "unsure":
            updated.append(
                match.model_copy(
                    update={
                        "status": MatchStatus.unknown,
                        "evidence": [],
                        "resume_evidence": [],
                        "explanation": "用户尚不确定是否满足该条件。",
                        "confidence": 1.0,
                    }
                )
            )
            continue
        fact = fact_by_requirement.get(requirement.original_text)
        if fact is None:
            updated.append(match)
            continue
        status = match.status
        if _allows_direct_confirmation(requirement):
            status = MatchStatus.matched
        elif status in {MatchStatus.missing, MatchStatus.unknown}:
            status = MatchStatus.partial
        evidence = list(match.evidence)
        if not any(item.fact_id == fact.id for item in evidence):
            evidence.append(
                MatchEvidence(
                    source="user_confirmed",
                    text=" ".join(filter(None, [fact.statement, fact.metrics])),
                    fact_id=fact.id,
                )
            )
        updated.append(
            match.model_copy(
                update={
                    "status": status,
                    "evidence": evidence,
                    "explanation": "已结合用户明确确认的补充信息重新判断。",
                    "confidence": max(match.confidence, 0.9),
                }
            )
        )
    return analysis.model_copy(update={"matches": updated})


def invalidate_generated_materials(job_analysis: dict) -> None:
    for key in [
        "final_analysis",
        "resume_edits",
        "tailored_resume_file",
        "cover_letters",
        "interview_preparations",
        "interview_feedback",
        "report_files",
        "application_package",
    ]:
        job_analysis.pop(key, None)
    job_analysis["stage"] = "clarification"
