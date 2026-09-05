from src.schemas import (
    CandidateFact,
    JobProfile,
    JobRequirement,
    MatchAnalysis,
    MatchStatus,
    RequirementCategory,
    RequirementImportance,
    RequirementMatch,
    ResumeSuggestion,
)
from src.submission import build_submission_checklist, safe_resume_filename


def _fixtures() -> tuple[JobProfile, MatchAnalysis, list[CandidateFact]]:
    requirement = JobRequirement(
        id="req_001",
        original_text="Python reporting",
        normalized_name="Python",
        category=RequirementCategory.skill,
        importance=RequirementImportance.must_have,
        is_hard_condition=False,
    )
    job = JobProfile(
        company="Example / Co",
        title="Data Analyst",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[requirement],
        domain_background=[],
    )
    analysis = MatchAnalysis(
        matches=[
            RequirementMatch(
                requirement_id=requirement.id,
                status=MatchStatus.partial,
                resume_evidence=["Processed 500 records with Python."],
                explanation="Relevant evidence.",
                confidence=0.8,
            )
        ],
        resume_suggestions=[
            ResumeSuggestion(
                original_text="Processed 500 records with Python.",
                suggested_text="Used Python to process 500 records.",
                requirement_ids=[requirement.id],
                reason="Clarify the action.",
                follow_up_question=None,
            )
        ],
    )
    fact = CandidateFact(
        id="fact_1",
        category=RequirementCategory.skill,
        statement="Processed 500 records with Python.",
        metrics="500 records",
        source_job_id="job_1",
        source_requirement_text=requirement.original_text,
    )
    return job, analysis, [fact]


def test_submission_checklist_allows_supported_edits() -> None:
    job, analysis, facts = _fixtures()
    checklist = build_submission_checklist(
        analysis,
        {"0": {"decision": "accepted", "text": "Used Python to process 500 records."}},
        job,
        facts,
    )

    assert checklist.ready is True


def test_submission_checklist_blocks_placeholders_and_new_numbers() -> None:
    job, analysis, facts = _fixtures()
    checklist = build_submission_checklist(
        analysis,
        {"0": {"decision": "accepted", "text": "Improved results by 99% in [项目]."}},
        job,
        facts,
    )

    assert checklist.ready is False
    failed_codes = {item.code for item in checklist.items if not item.passed}
    assert {"placeholders", "numbers"}.issubset(failed_codes)


def test_resume_filename_is_targeted_and_safe() -> None:
    job, _, _ = _fixtures()

    assert safe_resume_filename(job) == "tailored-resume-Example-Co-Data-Analyst.docx"
