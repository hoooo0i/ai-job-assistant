import pytest

from src.matching import (
    MatchValidationError,
    apply_user_confirmations,
    calculate_scores,
    validate_and_sanitise_matches,
)
from src.schemas import (
    CandidateFact,
    JobProfile,
    JobRequirement,
    InterviewCategory,
    InterviewQuestion,
    MatchAnalysis,
    MatchEvidence,
    MatchStatus,
    RequirementCategory,
    RequirementImportance,
    RequirementMatch,
    ResumeSuggestion,
)


def requirement(
    identifier: str,
    importance: RequirementImportance,
    category: RequirementCategory = RequirementCategory.skill,
) -> JobRequirement:
    return JobRequirement(
        id=identifier,
        original_text=f"Requirement {identifier}",
        normalized_name=f"Requirement {identifier}",
        category=category,
        importance=importance,
        is_hard_condition=category is RequirementCategory.availability,
    )


def match(identifier: str, status: MatchStatus, evidence: list[str] | None = None) -> RequirementMatch:
    return RequirementMatch(
        requirement_id=identifier,
        status=status,
        resume_evidence=evidence or [],
        explanation="Synthetic explanation",
        confidence=0.8,
    )


def job_with_three_requirements() -> JobProfile:
    return JobProfile(
        company="示例公司",
        title="示例岗位",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[
            requirement("req_001", RequirementImportance.must_have),
            requirement("req_002", RequirementImportance.preferred),
            requirement(
                "req_003",
                RequirementImportance.other,
                RequirementCategory.availability,
            ),
        ],
        domain_background=[],
    )


def test_calculates_match_score_and_completeness_with_unknown_excluded() -> None:
    job = job_with_three_requirements()
    analysis = MatchAnalysis(
        matches=[
            match("req_001", MatchStatus.matched, ["Used Python in a project"]),
            match("req_002", MatchStatus.partial, ["Studied analytics"]),
            match("req_003", MatchStatus.unknown),
        ]
    )

    score = calculate_scores(job, analysis)

    assert score.match_score == 80.0
    assert score.information_completeness == 83.3
    assert score.known_weight == 5
    assert score.total_weight == 6
    assert score.calculation_version == "v1.0"


def test_returns_no_match_score_when_everything_is_unknown() -> None:
    job = JobProfile(
        company="示例公司",
        title="示例岗位",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[requirement("req_001", RequirementImportance.must_have)],
        domain_background=[],
    )
    score = calculate_scores(
        job,
        MatchAnalysis(matches=[match("req_001", MatchStatus.unknown)]),
    )

    assert score.match_score is None
    assert score.information_completeness == 0.0


def test_user_confirmation_updates_unknown_and_recalculates_scores() -> None:
    job = job_with_three_requirements()
    base = MatchAnalysis(
        matches=[
            match("req_001", MatchStatus.matched, ["Used Python in a project"]),
            match("req_002", MatchStatus.partial, ["Studied analytics"]),
            match("req_003", MatchStatus.unknown),
        ]
    )

    updated = apply_user_confirmations(base, {"req_003": "matched"})
    score = calculate_scores(job, updated)

    assert base.matches[2].status is MatchStatus.unknown
    assert updated.matches[2].status is MatchStatus.matched
    assert updated.matches[2].confidence == 1.0
    assert updated.matches[2].explanation == "用户已确认满足该条件。"
    assert score.match_score == 83.3
    assert score.information_completeness == 100.0


def test_user_can_confirm_unknown_as_missing() -> None:
    base = MatchAnalysis(matches=[match("req_001", MatchStatus.unknown)])

    updated = apply_user_confirmations(base, {"req_001": "missing"})

    assert updated.matches[0].status is MatchStatus.missing


def test_rejects_invalid_confirmation_value() -> None:
    base = MatchAnalysis(matches=[match("req_001", MatchStatus.unknown)])

    with pytest.raises(MatchValidationError):
        apply_user_confirmations(base, {"req_001": "partial"})


def test_invalid_evidence_is_removed_and_positive_status_becomes_unknown() -> None:
    job = JobProfile(
        company="示例公司",
        title="示例岗位",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[requirement("req_001", RequirementImportance.must_have)],
        domain_background=[],
    )
    analysis = MatchAnalysis(
        matches=[match("req_001", MatchStatus.matched, ["Invented evidence"])]
    )

    sanitised = validate_and_sanitise_matches(
        analysis,
        job,
        "Used Python in a real synthetic project.",
    )

    assert sanitised.matches[0].status is MatchStatus.unknown
    assert sanitised.matches[0].resume_evidence == []
    assert "系统已保守调整" in sanitised.matches[0].explanation


def test_valid_evidence_allows_whitespace_differences() -> None:
    job = JobProfile(
        company="示例公司",
        title="示例岗位",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[requirement("req_001", RequirementImportance.must_have)],
        domain_background=[],
    )
    analysis = MatchAnalysis(
        matches=[match("req_001", MatchStatus.matched, ["Used Python in a project"])]
    )

    sanitised = validate_and_sanitise_matches(
        analysis,
        job,
        "Used   Python\nin a project to analyse synthetic data.",
    )

    assert sanitised.matches[0].status is MatchStatus.matched


def test_user_confirmed_evidence_is_validated_against_its_fact() -> None:
    job = JobProfile(
        company="示例公司",
        title="示例岗位",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[requirement("req_001", RequirementImportance.must_have)],
        domain_background=[],
    )
    fact = CandidateFact(
        id="fact_001",
        category=RequirementCategory.skill,
        statement="Built a Python workflow for synthetic reporting.",
        metrics="Processed 500 records.",
        source_job_id="job_1",
        source_requirement_text="Requirement req_001",
    )
    analysis = MatchAnalysis(
        matches=[
            RequirementMatch(
                requirement_id="req_001",
                status=MatchStatus.matched,
                resume_evidence=[],
                evidence=[
                    MatchEvidence(
                        source="user_confirmed",
                        text="Built a Python workflow",
                        fact_id=fact.id,
                    )
                ],
                explanation="User supplied evidence.",
                confidence=0.9,
            )
        ],
        resume_suggestions=[
            ResumeSuggestion(
                original_text="Built a Python workflow",
                suggested_text="Built a Python workflow processing 500 records.",
                requirement_ids=["req_001"],
                reason="Add confirmed scope.",
                follow_up_question=None,
            )
        ],
    )

    sanitised = validate_and_sanitise_matches(
        analysis,
        job,
        "Resume without this project.",
        [fact],
    )

    assert sanitised.matches[0].status is MatchStatus.matched
    assert sanitised.matches[0].evidence[0].source.value == "user_confirmed"
    assert len(sanitised.resume_suggestions) == 1


def test_user_evidence_with_wrong_fact_id_is_rejected() -> None:
    job = JobProfile(
        company="示例公司",
        title="示例岗位",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[requirement("req_001", RequirementImportance.must_have)],
        domain_background=[],
    )
    analysis = MatchAnalysis(
        matches=[
            RequirementMatch(
                requirement_id="req_001",
                status=MatchStatus.matched,
                resume_evidence=[],
                evidence=[
                    MatchEvidence(
                        source="user_confirmed",
                        text="Invented fact",
                        fact_id="fact_missing",
                    )
                ],
                explanation="Unsupported.",
                confidence=0.9,
            )
        ]
    )

    sanitised = validate_and_sanitise_matches(analysis, job, "Resume text.", [])

    assert sanitised.matches[0].status is MatchStatus.unknown
    assert sanitised.matches[0].evidence == []


def test_filters_unsupported_resume_suggestions_and_invalid_interview_links() -> None:
    job = JobProfile(
        company="示例公司",
        title="示例岗位",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[requirement("req_001", RequirementImportance.must_have)],
        domain_background=[],
    )
    resume_text = "Used Python to analyse synthetic data and present findings."
    analysis = MatchAnalysis(
        matches=[
            match(
                "req_001",
                MatchStatus.matched,
                ["Used Python to analyse synthetic data"],
            )
        ],
        resume_suggestions=[
            ResumeSuggestion(
                original_text="Used Python to analyse synthetic data",
                suggested_text="Analysed synthetic data using Python and presented findings.",
                requirement_ids=["req_001"],
                reason="突出工具和行动。",
                follow_up_question=None,
            ),
            ResumeSuggestion(
                original_text="Used Python to analyse synthetic data",
                suggested_text="Improved performance by 50% using Python.",
                requirement_ids=["req_001"],
                reason="加入了没有来源的数字。",
                follow_up_question=None,
            ),
            ResumeSuggestion(
                original_text="This sentence does not exist",
                suggested_text="Unsupported rewrite.",
                requirement_ids=["req_001"],
                reason="原文不存在。",
                follow_up_question=None,
            ),
        ],
        interview_questions=[
            InterviewQuestion(
                category=InterviewCategory.project_deep_dive,
                question="你如何使用 Python 分析数据？",
                why_asked="验证项目深度。",
                answer_outline=["说明任务", "说明方法", "说明真实结果"],
                requirement_ids=["req_001"],
            ),
            InterviewQuestion(
                category=InterviewCategory.job_knowledge,
                question="无效关联问题",
                why_asked="测试过滤。",
                answer_outline=[],
                requirement_ids=["req_missing"],
            ),
        ],
    )

    sanitised = validate_and_sanitise_matches(analysis, job, resume_text)

    assert len(sanitised.resume_suggestions) == 1
    assert len(sanitised.interview_questions) == 1
    assert sanitised.interview_questions[0].requirement_ids == ["req_001"]


@pytest.mark.parametrize(
    "matches",
    [
        [match("req_001", MatchStatus.missing)],
        [
            match("req_001", MatchStatus.missing),
            match("req_001", MatchStatus.unknown),
            match("req_003", MatchStatus.unknown),
        ],
    ],
)
def test_rejects_incomplete_or_duplicate_requirement_coverage(matches) -> None:
    with pytest.raises(MatchValidationError):
        validate_and_sanitise_matches(
            MatchAnalysis(matches=matches),
            job_with_three_requirements(),
            "Synthetic resume content",
        )
