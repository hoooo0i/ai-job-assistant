from types import SimpleNamespace

import pytest

from src.ai_parser import (
    AiParserError,
    MAX_AI_INPUT_CHARACTERS,
    MissingApiKeyError,
    create_openai_client,
    get_model_name,
    match_requirements,
    parse_job,
    parse_resume,
)
from src.schemas import (
    JobProfile,
    JobRequirement,
    MatchAnalysis,
    MatchStatus,
    RequirementCategory,
    RequirementImportance,
    RequirementMatch,
    ResumeProfile,
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
