from __future__ import annotations

import re

from src.privacy import redact_sensitive_info
from src.schemas import (
    CandidateFact,
    EvidenceSource,
    JobProfile,
    MatchAnalysis,
    MatchEvidence,
    MatchStatus,
    RequirementImportance,
    RequirementMatch,
    ScoreResult,
)


CALCULATION_VERSION = "v1.0"

IMPORTANCE_WEIGHTS = {
    RequirementImportance.must_have: 3,
    RequirementImportance.preferred: 2,
    RequirementImportance.other: 1,
}

STATUS_COEFFICIENTS = {
    MatchStatus.matched: 1.0,
    MatchStatus.partial: 0.5,
    MatchStatus.missing: 0.0,
}


class MatchValidationError(ValueError):
    """Raised when model match output cannot be reconciled with the JD."""


def _normalise_for_evidence_check(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _number_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?%?", text))


def validate_and_sanitise_matches(
    analysis: MatchAnalysis,
    job_profile: JobProfile,
    resume_text: str,
    candidate_facts: list[CandidateFact] | None = None,
) -> MatchAnalysis:
    """Require one result per JD requirement and remove unsupported evidence."""
    expected_ids = [requirement.id for requirement in job_profile.requirements]
    returned_ids = [match.requirement_id for match in analysis.matches]

    if len(returned_ids) != len(set(returned_ids)):
        raise MatchValidationError("模型返回了重复的岗位要求匹配结果，请重试。")
    if set(returned_ids) != set(expected_ids):
        raise MatchValidationError("模型返回的匹配结果未完整覆盖岗位要求，请重试。")

    safe_resume = _normalise_for_evidence_check(redact_sensitive_info(resume_text))
    facts = candidate_facts or []
    fact_text_by_id = {
        item.id: _normalise_for_evidence_check(
            " ".join(filter(None, [item.statement, item.metrics]))
        )
        for item in facts
    }
    combined_source = " ".join([safe_resume, *fact_text_by_id.values()])
    matches_by_id = {match.requirement_id: match for match in analysis.matches}
    sanitised_matches: list[RequirementMatch] = []

    for requirement_id in expected_ids:
        match = matches_by_id[requirement_id]
        valid_evidence = [
            evidence.strip()
            for evidence in match.resume_evidence
            if evidence.strip()
            and _normalise_for_evidence_check(evidence) in safe_resume
        ]
        structured_evidence: list[MatchEvidence] = []
        for evidence in match.evidence:
            text = redact_sensitive_info(evidence.text).strip()
            normalised_text = _normalise_for_evidence_check(text)
            if not normalised_text:
                continue
            if evidence.source is EvidenceSource.resume and normalised_text in safe_resume:
                structured_evidence.append(
                    evidence.model_copy(update={"text": text, "fact_id": None})
                )
            elif (
                evidence.source is EvidenceSource.user_confirmed
                and evidence.fact_id in fact_text_by_id
                and normalised_text in fact_text_by_id[evidence.fact_id]
            ):
                structured_evidence.append(evidence.model_copy(update={"text": text}))
        known_structured_text = {
            _normalise_for_evidence_check(item.text) for item in structured_evidence
        }
        for text in valid_evidence:
            if _normalise_for_evidence_check(text) not in known_structured_text:
                structured_evidence.append(
                    MatchEvidence(source=EvidenceSource.resume, text=text, fact_id=None)
                )

        status = match.status
        explanation = match.explanation
        if status in {MatchStatus.matched, MatchStatus.partial} and not structured_evidence:
            status = MatchStatus.unknown
            explanation = (
                "模型未提供可在简历原文中验证的证据，系统已保守调整为待确认。"
            )

        sanitised_matches.append(
            match.model_copy(
                update={
                    "status": status,
                    "resume_evidence": valid_evidence,
                    "evidence": structured_evidence,
                    "explanation": explanation,
                }
            )
        )

    valid_requirement_ids = set(expected_ids)
    source_numbers = _number_tokens(combined_source)
    valid_suggestions = []
    for suggestion in analysis.resume_suggestions:
        original_is_present = (
            _normalise_for_evidence_check(suggestion.original_text) in combined_source
        )
        ids_are_valid = bool(suggestion.requirement_ids) and set(
            suggestion.requirement_ids
        ).issubset(valid_requirement_ids)
        adds_numbers = not _number_tokens(suggestion.suggested_text).issubset(
            source_numbers
        )
        if original_is_present and ids_are_valid and not adds_numbers:
            valid_suggestions.append(suggestion)

    valid_interview_questions = [
        question
        for question in analysis.interview_questions
        if question.requirement_ids
        and set(question.requirement_ids).issubset(valid_requirement_ids)
    ]

    return MatchAnalysis(
        matches=sanitised_matches,
        resume_suggestions=valid_suggestions,
        interview_questions=valid_interview_questions,
    )


def calculate_scores(job_profile: JobProfile, analysis: MatchAnalysis) -> ScoreResult:
    """Calculate deterministic evidence match and information completeness scores."""
    matches_by_id = {match.requirement_id: match for match in analysis.matches}
    expected_ids = {requirement.id for requirement in job_profile.requirements}
    if set(matches_by_id) != expected_ids:
        raise MatchValidationError("评分前的匹配结果与岗位要求不一致。")

    total_weight = 0
    known_weight = 0
    weighted_score = 0.0

    for requirement in job_profile.requirements:
        weight = IMPORTANCE_WEIGHTS[requirement.importance]
        match = matches_by_id[requirement.id]
        total_weight += weight
        if match.status is MatchStatus.unknown:
            continue
        known_weight += weight
        weighted_score += weight * STATUS_COEFFICIENTS[match.status]

    match_score = round(weighted_score / known_weight * 100, 1) if known_weight else None
    completeness = round(known_weight / total_weight * 100, 1) if total_weight else 0.0

    return ScoreResult(
        match_score=match_score,
        information_completeness=completeness,
        known_weight=known_weight,
        total_weight=total_weight,
        calculation_version=CALCULATION_VERSION,
    )


def apply_user_confirmations(
    base_analysis: MatchAnalysis,
    answers: dict[str, str],
) -> MatchAnalysis:
    """Apply explicit user answers to unknown matches without calling a model."""
    allowed_answers = {"unknown", "matched", "missing"}
    updated_matches: list[RequirementMatch] = []

    for match in base_analysis.matches:
        answer = answers.get(match.requirement_id, "unknown")
        if answer not in allowed_answers:
            raise MatchValidationError("待确认问题包含无效回答。")

        if match.status is not MatchStatus.unknown or answer == "unknown":
            updated_matches.append(match)
            continue

        new_status = MatchStatus(answer)
        explanation = (
            "用户已确认满足该条件。"
            if new_status is MatchStatus.matched
            else "用户已确认不满足该条件。"
        )
        updated_matches.append(
            match.model_copy(
                update={
                    "status": new_status,
                    "resume_evidence": [],
                    "explanation": explanation,
                    "confidence": 1.0,
                }
            )
        )

    return base_analysis.model_copy(update={"matches": updated_matches})
