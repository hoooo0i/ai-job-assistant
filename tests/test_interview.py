from src.interview import collect_interview_evidence, sanitise_interview_preparation
from src.schemas import (
    EvidenceChunk,
    InterviewCategory,
    InterviewPreparation,
    InterviewQuestion,
    MatchAnalysis,
    MatchStatus,
    RequirementMatch,
    ResumeProfile,
    StarOutline,
)


def resume_with_evidence() -> ResumeProfile:
    return ResumeProfile(
        summary=None,
        education=[],
        experience=[],
        projects=[],
        skills=[],
        languages=[],
        evidence_chunks=[
            EvidenceChunk(
                source_section="项目",
                text="Built a Python dashboard for synthetic sales data.",
            ),
            EvidenceChunk(
                source_section="联系方式",
                text="Email candidate@example.test for more information.",
            ),
        ],
    )


def question() -> InterviewQuestion:
    return InterviewQuestion(
        category=InterviewCategory.project_deep_dive,
        question="请介绍一个 Python 项目。",
        why_asked="验证项目经验。",
        answer_outline=["说明背景", "说明行动"],
        requirement_ids=["req_001"],
    )


def analysis() -> MatchAnalysis:
    return MatchAnalysis(
        matches=[
            RequirementMatch(
                requirement_id="req_001",
                status=MatchStatus.matched,
                resume_evidence=["Built a Python dashboard for synthetic sales data."],
                explanation="存在证据。",
                confidence=0.9,
            )
        ]
    )


def preparation(answer: str | None, evidence_ids: list[str]) -> InterviewPreparation:
    return InterviewPreparation(
        personalized_answer=answer,
        key_points=[],
        star_outline=StarOutline(situation=None, task=None, action=None, result=None),
        evidence_ids=evidence_ids,
        missing_information=[],
        caution_notes=[],
    )


def test_collects_related_evidence_first_and_deduplicates() -> None:
    evidence = collect_interview_evidence(question(), resume_with_evidence(), analysis())

    assert evidence[0].id == "ev_001"
    assert evidence[0].source == "简历·岗位要求 req_001"
    assert len([item for item in evidence if "Python dashboard" in item.text]) == 1
    assert "candidate@example.test" not in evidence[1].text
    assert "[已隐藏邮箱]" in evidence[1].text


def test_preparation_removes_unknown_evidence_ids() -> None:
    evidence = collect_interview_evidence(question(), resume_with_evidence(), analysis())
    result = sanitise_interview_preparation(
        preparation("I built a Python dashboard.", ["ev_missing"]),
        evidence,
    )

    assert result.personalized_answer is None
    assert result.evidence_ids == []
    assert result.missing_information


def test_preparation_hides_answer_that_invents_a_number() -> None:
    evidence = collect_interview_evidence(question(), resume_with_evidence(), analysis())
    result = sanitise_interview_preparation(
        preparation("I improved performance by 50%.", ["ev_001"]),
        evidence,
    )

    assert result.personalized_answer is None
    assert "不存在的数字" in result.missing_information[0]
