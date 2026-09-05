from types import SimpleNamespace

import pytest

from src.ai_parser import (
    AiParserError,
    MAX_AI_INPUT_CHARACTERS,
    MissingApiKeyError,
    create_openai_client,
    finalize_match_requirements,
    generate_cover_letter,
    generate_supplement_resume_suggestions,
    get_model_name,
    match_requirements,
    parse_job,
    parse_resume,
    preliminary_match_requirements,
    prepare_interview_answer,
    review_interview_answer,
)
from src.schemas import (
    CandidateFact,
    ClarificationAnswer,
    ClarificationQuestion,
    CoverLetterDraft,
    CoverLetterParagraph,
    EvidenceChunk,
    InterviewCategory,
    InterviewFeedback,
    InterviewQuestion,
    JobProfile,
    JobRequirement,
    MatchAnalysis,
    MatchStatus,
    PreliminaryAnalysis,
    RequirementCategory,
    RequirementImportance,
    RequirementMatch,
    ResumeProfile,
    ResumeSuggestion,
    SupplementRewriteResult,
)
from src.validators import JobInput


class FakeResponses:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.parsed)


class FakeClient:
    def __init__(self, parsed):
        self.responses = FakeResponses(parsed)


def empty_resume_profile() -> ResumeProfile:
    return ResumeProfile(
        summary=None,
        education=[],
        experience=[],
        projects=[],
        skills=[],
        languages=[],
        evidence_chunks=[],
    )


def empty_job_profile() -> JobProfile:
    return JobProfile(
        company="示例公司",
        title="示例岗位",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[],
        domain_background=[],
    )


def test_default_model_is_gpt_5_5(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert get_model_name() == "gpt-5.5"


def test_model_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.5-2026-04-23")

    assert get_model_name() == "gpt-5.5-2026-04-23"


def test_missing_api_key_has_safe_error(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingApiKeyError):
        create_openai_client()


def test_resume_parse_uses_structured_output_and_redacts_contact_details(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    client = FakeClient(empty_resume_profile())
    result = parse_resume(
        "Project experience and technical skills. Email sample.person@example.test. " * 3,
        client=client,
    )

    call = client.responses.calls[0]
    assert result == empty_resume_profile()
    assert call["model"] == "gpt-5.5"
    assert call["text_format"] is ResumeProfile
    assert call["reasoning"] == {"effort": "low"}
    assert call["text"] == {"verbosity": "low"}
    assert "verbosity" not in call
    assert call["store"] is False
    assert "sample.person@example.test" not in call["input"][0]["content"]
    assert "[已隐藏邮箱]" in call["input"][0]["content"]


def test_job_parse_uses_metadata_and_job_schema() -> None:
    client = FakeClient(empty_job_profile())
    job = JobInput(
        company="示例公司",
        job_title="示例岗位",
        location="示例城市",
        job_type="实习",
        jd_text="岗位职责、技能要求、工作安排和协作能力说明。" * 6,
    )

    result = parse_job(job, client=client)
    call = client.responses.calls[0]

    assert result == empty_job_profile()
    assert call["text_format"] is JobProfile
    assert "Company: 示例公司" in call["input"][0]["content"]
    assert "Title: 示例岗位" in call["input"][0]["content"]


def test_rejects_oversized_ai_input() -> None:
    client = FakeClient(empty_resume_profile())

    with pytest.raises(AiParserError, match="超过"):
        parse_resume("A" * (MAX_AI_INPUT_CHARACTERS + 1), client=client)


def test_match_requirements_uses_match_schema_and_preserves_valid_evidence() -> None:
    resume = empty_resume_profile()
    job = JobProfile(
        company="示例公司",
        title="示例岗位",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[
            JobRequirement(
                id="req_001",
                original_text="需要 Python 项目经验",
                normalized_name="Python 项目经验",
                category=RequirementCategory.skill,
                importance=RequirementImportance.must_have,
                is_hard_condition=False,
            )
        ],
        domain_background=[],
    )
    parsed = MatchAnalysis(
        matches=[
            RequirementMatch(
                requirement_id="req_001",
                status=MatchStatus.matched,
                resume_evidence=["Used Python in a project"],
                explanation="存在直接项目证据。",
                confidence=0.9,
            )
        ]
    )
    client = FakeClient(parsed)

    result = match_requirements(
        resume,
        job,
        "Used Python in a project to analyse synthetic data.",
        client=client,
    )

    call = client.responses.calls[0]
    assert call["text_format"] is MatchAnalysis
    assert result.matches[0].status is MatchStatus.matched
    assert result.matches[0].resume_evidence == ["Used Python in a project"]


def test_preliminary_match_returns_matches_questions_and_hidden_final_materials() -> None:
    resume = empty_resume_profile()
    job = JobProfile(
        company="示例公司",
        title="示例岗位",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[
            JobRequirement(
                id="req_001",
                original_text="需要 Python 项目经验",
                normalized_name="Python 项目经验",
                category=RequirementCategory.skill,
                importance=RequirementImportance.must_have,
                is_hard_condition=False,
            )
        ],
        domain_background=[],
    )
    parsed = PreliminaryAnalysis(
        matches=[
            RequirementMatch(
                requirement_id="req_001",
                status=MatchStatus.missing,
                resume_evidence=[],
                explanation="没有证据。",
                confidence=0.8,
            )
        ],
        clarification_questions=[
            ClarificationQuestion(
                id="cq_001",
                requirement_id="req_001",
                prompt="你是否有未写入简历的 Python 项目？",
            )
        ],
    )
    client = FakeClient(parsed)

    result = preliminary_match_requirements(
        resume,
        job,
        "Synthetic resume text without matching project experience.",
        client=client,
    )

    assert client.responses.calls[0]["text_format"] is PreliminaryAnalysis
    assert len(result.clarification_questions) == 1
    assert result.resume_suggestions == []
    assert result.interview_questions == []


def test_supplement_rewrite_batches_requirements_into_one_model_call() -> None:
    resume = empty_resume_profile()
    requirement = JobRequirement(
        id="req_001",
        original_text="需要 Python 项目经验",
        normalized_name="Python 项目经验",
        category=RequirementCategory.skill,
        importance=RequirementImportance.must_have,
        is_hard_condition=False,
    )
    job = JobProfile(
        company="示例公司",
        title="示例岗位",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[requirement],
        domain_background=[],
    )
    fact = CandidateFact(
        id="fact_001",
        category=RequirementCategory.skill,
        statement="情境：课程项目；行动：使用 Python 清洗并分析数据；结果：完成可视化报告",
        metrics=None,
        source_job_id="job_1",
        source_requirement_text=requirement.original_text,
    )
    analysis = MatchAnalysis(
        matches=[
            RequirementMatch(
                requirement_id=requirement.id,
                status=MatchStatus.partial,
                resume_evidence=[],
                evidence=[
                    {
                        "source": "user_confirmed",
                        "text": fact.statement,
                        "fact_id": fact.id,
                    }
                ],
                explanation="用户确认并补充了具体经历。",
                confidence=0.9,
            )
        ]
    )
    suggestion = ResumeSuggestion(
        original_text=fact.statement,
        suggested_text="在课程项目中使用 Python 清洗和分析数据，并完成可视化报告。",
        requirement_ids=[requirement.id],
        reason="突出岗位相关行动和产出。",
        follow_up_question=None,
    )
    client = FakeClient(SupplementRewriteResult(suggestions=[suggestion]))

    result = generate_supplement_resume_suggestions(
        resume,
        job,
        "Synthetic resume without this project detail.",
        analysis,
        [fact],
        [requirement.id],
        client=client,
    )

    assert len(client.responses.calls) == 1
    assert result == [suggestion]
    assert client.responses.calls[0]["text_format"] is SupplementRewriteResult


def test_final_match_accepts_user_confirmed_evidence_with_provenance() -> None:
    resume = empty_resume_profile()
    requirement = JobRequirement(
        id="req_001",
        original_text="需要 Python 项目经验",
        normalized_name="Python 项目经验",
        category=RequirementCategory.skill,
        importance=RequirementImportance.must_have,
        is_hard_condition=False,
    )
    job = JobProfile(
        company="示例公司",
        title="示例岗位",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[requirement],
        domain_background=[],
    )
    preliminary = PreliminaryAnalysis(
        matches=[
            RequirementMatch(
                requirement_id="req_001",
                status=MatchStatus.missing,
                resume_evidence=[],
                explanation="没有证据。",
                confidence=0.8,
            )
        ],
        clarification_questions=[],
    )
    fact = CandidateFact(
        id="fact_001",
        category=RequirementCategory.skill,
        statement="Built a Python workflow for synthetic reporting.",
        metrics=None,
        source_job_id="job_1",
        source_requirement_text=requirement.original_text,
    )
    answer = ClarificationAnswer(
        question_id="cq_001",
        requirement_id="req_001",
        status="have",
        evidence_text=fact.statement,
    )
    parsed = MatchAnalysis(
        matches=[
            RequirementMatch(
                requirement_id="req_001",
                status=MatchStatus.partial,
                resume_evidence=[],
                evidence=[
                    {
                        "source": "user_confirmed",
                        "text": fact.statement,
                        "fact_id": fact.id,
                    }
                ],
                explanation="用户提供了相关经验。",
                confidence=0.9,
            )
        ]
    )
    client = FakeClient(parsed)

    result = finalize_match_requirements(
        resume,
        job,
        "Synthetic resume text without matching project experience.",
        preliminary,
        [fact],
        [answer],
        client=client,
    )

    assert result.matches[0].evidence[0].source.value == "user_confirmed"
    assert result.matches[0].resume_evidence == []


def interview_question() -> InterviewQuestion:
    return InterviewQuestion(
        category=InterviewCategory.project_deep_dive,
        question="请介绍一个 Python 项目。",
        why_asked="验证项目深度。",
        answer_outline=["说明背景", "说明行动"],
        requirement_ids=[],
    )


def test_prepare_interview_answer_skips_model_when_no_evidence() -> None:
    preparation, evidence = prepare_interview_answer(
        interview_question(),
        empty_resume_profile(),
        MatchAnalysis(matches=[]),
    )

    assert evidence == []
    assert preparation.personalized_answer is None
    assert preparation.key_points == ["说明背景", "说明行动"]
    assert preparation.missing_information


def test_review_interview_answer_uses_feedback_schema_and_redacts_email() -> None:
    feedback = InterviewFeedback(
        completeness_score=3,
        star_score=2,
        relevance_score=4,
        clarity_score=3,
        strengths=["说明了行动。"],
        improvements=["补充结果。"],
        unsupported_claims=[],
        improved_structure=["先交代背景。"],
        follow_up_question=None,
    )
    client = FakeClient(feedback)

    result = review_interview_answer(
        interview_question(),
        "I completed a synthetic project and explained the implementation. candidate@example.test",
        empty_job_profile(),
        [],
        client=client,
    )

    call = client.responses.calls[0]
    assert result == feedback
    assert call["text_format"] is InterviewFeedback
    assert "candidate@example.test" not in call["input"][0]["content"]
    assert "[已隐藏邮箱]" in call["input"][0]["content"]


def test_review_interview_answer_rejects_short_content() -> None:
    with pytest.raises(AiParserError, match="至少需要 20"):
        review_interview_answer(
            interview_question(),
            "too short",
            empty_job_profile(),
            [],
        )


def test_cover_letter_removes_paragraph_with_unsupported_number() -> None:
    resume = empty_resume_profile().model_copy(
        update={
            "evidence_chunks": [
                EvidenceChunk(
                    source_section="项目",
                    text="Built a Python dashboard for synthetic reporting.",
                )
            ]
        }
    )
    draft = CoverLetterDraft(
        language="zh",
        salutation="尊敬的招聘团队：",
        paragraphs=[
            CoverLetterParagraph(
                text="我曾构建 Python 仪表板用于报表。",
                evidence_ids=["ev_001"],
            ),
            CoverLetterParagraph(
                text="我将性能提升了 50%。",
                evidence_ids=["ev_001"],
            ),
        ],
        closing="谢谢考虑。",
        caution_notes=[],
    )
    client = FakeClient(draft)

    result, evidence = generate_cover_letter(
        resume,
        empty_job_profile(),
        MatchAnalysis(matches=[]),
        "zh",
        client=client,
    )

    assert len(evidence) == 1
    assert len(result.paragraphs) == 1
    assert "50%" not in result.paragraphs[0].text
    assert "已移除 1 段" in result.caution_notes[0]


def test_cover_letter_requires_resume_evidence() -> None:
    with pytest.raises(AiParserError, match="可验证证据"):
        generate_cover_letter(
            empty_resume_profile(),
            empty_job_profile(),
            MatchAnalysis(matches=[]),
            "zh",
        )
