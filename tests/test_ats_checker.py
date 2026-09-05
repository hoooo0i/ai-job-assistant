import fitz

from src.ats_checker import build_ats_report, inspect_pdf_layout
from src.schemas import (
    JobProfile,
    JobRequirement,
    MatchAnalysis,
    MatchStatus,
    RequirementCategory,
    RequirementImportance,
    RequirementMatch,
)


def _pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "EXPERIENCE\nPython reporting project\nEDUCATION\nSKILLS")
    data = document.tobytes()
    document.close()
    return data


def _job_and_analysis() -> tuple[JobProfile, MatchAnalysis]:
    requirement = JobRequirement(
        id="req_001",
        original_text="Python",
        normalized_name="Python",
        category=RequirementCategory.skill,
        importance=RequirementImportance.must_have,
        is_hard_condition=False,
    )
    job = JobProfile(
        company="Example",
        title="Analyst",
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
                status=MatchStatus.matched,
                resume_evidence=["Python reporting project"],
                explanation="Direct evidence.",
                confidence=0.9,
            )
        ]
    )
    return job, analysis


def test_pdf_layout_inspection_returns_only_layout_signals() -> None:
    result = inspect_pdf_layout(_pdf_bytes())

    assert result.readable is True
    assert result.page_count == 1
    assert result.text_block_count >= 1
    assert result.minimum_font_size is not None


def test_ats_report_checks_keywords_without_a_model() -> None:
    job, analysis = _job_and_analysis()
    text = "EXPERIENCE\nPython reporting project\nEDUCATION\nSKILLS\nname@example.test"

    report = build_ats_report(text, job, analysis, inspect_pdf_layout(_pdf_bytes()), "pdf")

    assert report.keyword_coverage == 100.0
    assert next(item for item in report.checks if item.code == "contact").severity == "passed"
    assert next(item for item in report.checks if item.code == "text_pdf").severity == "passed"


def test_fallback_resume_is_flagged_as_not_ats_readable() -> None:
    job, analysis = _job_and_analysis()
    layout = inspect_pdf_layout(b"not a pdf")

    report = build_ats_report("Python experience", job, analysis, layout, "pasted")

    check = next(item for item in report.checks if item.code == "text_pdf")
    assert check.severity == "critical"
    assert report.score < 100
