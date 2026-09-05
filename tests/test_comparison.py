from src.comparison import build_job_comparison
from src.schemas import (
    JobProfile,
    JobRequirement,
    MatchStatus,
    PreliminaryAnalysis,
    RequirementCategory,
    RequirementImportance,
    RequirementMatch,
)


def _job_bundle(job_id: str, company: str, status: MatchStatus, hard: bool) -> dict:
    requirement = JobRequirement(
        id="req_001",
        original_text="需要 Python 经验",
        normalized_name="Python",
        category=RequirementCategory.skill,
        importance=RequirementImportance.must_have,
        is_hard_condition=hard,
    )
    profile = JobProfile(
        company=company,
        title="数据岗位",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[requirement],
        domain_background=[],
    )
    analysis = PreliminaryAnalysis(
        matches=[
            RequirementMatch(
                requirement_id="req_001",
                status=status,
                resume_evidence=["Python"] if status is MatchStatus.matched else [],
                explanation="Synthetic evaluation",
                confidence=0.9,
            )
        ],
        clarification_questions=[],
    )
    return {
        "job_id": job_id,
        "fingerprint": job_id,
        "job_profile": profile.model_dump(mode="json"),
        "preliminary_analysis": analysis.model_dump(mode="json"),
        "clarification_questions": [],
        "clarification_answers": [],
        "stage": "clarification",
        "application_tracking": {
            "status": "applied" if job_id == "strong" else "not_started"
        },
    }


def test_job_comparison_ranks_stronger_job_first_and_counts_risks() -> None:
    candidate = {
        "resume_text": "Python project experience",
        "resume_source": "pasted",
        "page_count": 0,
    }
    jobs = {
        "strong": _job_bundle("strong", "甲公司", MatchStatus.matched, False),
        "risk": _job_bundle("risk", "乙公司", MatchStatus.missing, True),
    }

    rows = build_job_comparison(candidate, jobs)

    assert [item.job_id for item in rows] == ["strong", "risk"]
    assert rows[0].recommendation_score > rows[1].recommendation_score
    assert rows[1].hard_risks == 1
    assert rows[1].must_have_gaps == 1
    assert rows[0].application_status == "applied"
