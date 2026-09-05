from __future__ import annotations

from src.ats_checker import build_ats_report
from src.application_tracker import get_application_record
from src.matching import calculate_scores
from src.schemas import (
    JobComparisonItem,
    JobProfile,
    MatchAnalysis,
    MatchStatus,
    PdfLayoutSignals,
    PreliminaryAnalysis,
)


def _analysis_from_bundle(bundle: dict) -> MatchAnalysis:
    if bundle.get("stage") == "final" and bundle.get("final_analysis"):
        return MatchAnalysis.model_validate(bundle["final_analysis"])
    preliminary = PreliminaryAnalysis.model_validate(bundle["preliminary_analysis"])
    return MatchAnalysis(
        matches=preliminary.matches,
        resume_suggestions=preliminary.resume_suggestions,
        interview_questions=preliminary.interview_questions,
    )


def build_job_comparison(
    candidate_profile: dict,
    job_analyses: dict[str, dict],
) -> list[JobComparisonItem]:
    """Build a locally calculated, ranked comparison for the current resume."""
    layout = PdfLayoutSignals.model_validate(
        candidate_profile.get(
            "pdf_layout",
            {
                "page_count": candidate_profile.get("page_count", 0) or 0,
                "readable": candidate_profile.get("resume_source") == "pdf",
            },
        )
    )
    rows: list[JobComparisonItem] = []
    for job_id, bundle in job_analyses.items():
        job_profile = JobProfile.model_validate(bundle["job_profile"])
        analysis = _analysis_from_bundle(bundle)
        score = calculate_scores(job_profile, analysis)
        ats = build_ats_report(
            candidate_profile.get("resume_text", ""),
            job_profile,
            analysis,
            layout,
            candidate_profile.get("resume_source", "pdf"),
        )
        matches = {item.requirement_id: item for item in analysis.matches}
        hard_risks = sum(
            requirement.is_hard_condition
            and matches[requirement.id].status
            in {MatchStatus.missing, MatchStatus.unknown}
            for requirement in job_profile.requirements
            if requirement.id in matches
        )
        must_have_gaps = sum(
            requirement.importance.value == "must_have"
            and matches[requirement.id].status is MatchStatus.missing
            for requirement in job_profile.requirements
            if requirement.id in matches
        )
        match_score = score.match_score or 0.0
        recommendation_score = max(
            0.0,
            round(
                match_score * 0.65
                + score.information_completeness * 0.15
                + ats.score * 0.20
                - hard_risks * 15,
                1,
            ),
        )
        rows.append(
            JobComparisonItem(
                job_id=job_id,
                company=job_profile.company,
                title=job_profile.title,
                stage="final" if bundle.get("stage") == "final" else "clarification",
                match_score=score.match_score,
                information_completeness=score.information_completeness,
                ats_score=ats.score,
                hard_risks=hard_risks,
                must_have_gaps=must_have_gaps,
                recommendation_score=recommendation_score,
                application_status=get_application_record(bundle).status.value,
            )
        )
    return sorted(rows, key=lambda item: item.recommendation_score, reverse=True)
