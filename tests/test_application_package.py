from io import BytesIO
from zipfile import ZipFile

from src.application_package import build_application_package
from src.schemas import (
    CoverLetterDraft,
    CoverLetterParagraph,
    JobProfile,
    JobRequirement,
    MatchAnalysis,
    MatchStatus,
    RequirementCategory,
    RequirementImportance,
    RequirementMatch,
    ResumeProfile,
    ResumeVersion,
    ScoreResult,
)


def _package_inputs():
    resume = ResumeProfile(
        summary="Synthetic candidate",
        education=[],
        experience=[],
        projects=[],
        skills=[],
        languages=[],
        evidence_chunks=[],
    )
    requirement = JobRequirement(
        id="req_1",
        original_text="需要数据分析经验",
        normalized_name="数据分析经验",
        category=RequirementCategory.experience,
        importance=RequirementImportance.must_have,
        is_hard_condition=False,
    )
    job = JobProfile(
        company="示例公司",
        title="数据分析师",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[requirement],
        domain_background=[],
    )
    analysis = MatchAnalysis(
        matches=[
            RequirementMatch(
                requirement_id="req_1",
                status=MatchStatus.matched,
                resume_evidence=["Completed a verified analytics project."],
                explanation="Found direct evidence.",
                confidence=0.9,
            )
        ]
    )
    score = ScoreResult(
        match_score=100.0,
        information_completeness=100.0,
        known_weight=3,
        total_weight=3,
        calculation_version="v1.0",
    )
    return resume, job, analysis, score


def test_package_always_contains_reports_and_checklist() -> None:
    resume, job, analysis, score = _package_inputs()

    result = build_application_package(resume, job, analysis, score, {}, [])

    with ZipFile(BytesIO(result.data)) as archive:
        names = set(archive.namelist())
        assert {
            "README.txt",
            "job-analysis-report.docx",
            "job-analysis-report.pdf",
            "submission-checklist.txt",
        }.issubset(names)
        assert "tailored-resume.docx" not in names
        assert not any(name.startswith("cover-letter") for name in names)
        assert "尚未保存简历版本" in "\n".join(result.warnings)


def test_package_includes_saved_resume_and_generated_cover_letter() -> None:
    resume, job, analysis, score = _package_inputs()
    version = ResumeVersion(
        id="version_1",
        label="投递版",
        created_at="2026-09-06T00:00:00+00:00",
        decisions={},
        accepted_suggestions=[("Original", "Verified tailored statement.")],
    )
    draft = CoverLetterDraft(
        language="zh",
        salutation="尊敬的招聘团队：",
        paragraphs=[
            CoverLetterParagraph(
                text="我的数据分析经历与该岗位相关。",
                evidence_ids=["req_1"],
            )
        ],
        closing="感谢您的考虑。",
        caution_notes=[],
    )
    bundle = {
        "resume_versions": [version.model_dump(mode="json")],
        "application_tracking": {"resume_version_id": "version_1"},
        "cover_letters": {"zh": {"draft": draft.model_dump(mode="json")}},
    }

    result = build_application_package(resume, job, analysis, score, bundle, [])

    with ZipFile(BytesIO(result.data)) as archive:
        names = set(archive.namelist())
        assert "tailored-resume.docx" in names
        assert "cover-letter-zh.docx" in names
        assert "cover-letter-zh.pdf" in names
        assert archive.read("job-analysis-report.pdf").startswith(b"%PDF")
        assert archive.read("tailored-resume.docx").startswith(b"PK")
    assert not result.warnings
