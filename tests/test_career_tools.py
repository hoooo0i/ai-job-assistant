from src.career_tools import (
    accepted_resume_suggestions,
    analyse_keyword_gaps,
    collect_application_evidence,
    render_resume_diff_html,
)
from src.schemas import (
    EvidenceChunk,
    JobProfile,
    JobRequirement,
    MatchAnalysis,
    MatchStatus,
    RequirementCategory,
    RequirementImportance,
    RequirementMatch,
    ResumeProfile,
    ResumeSuggestion,
)


def _resume() -> ResumeProfile:
    return ResumeProfile(
        summary=None,
        education=[],
        experience=[],
        projects=[],
        skills=[],
        languages=[],
        evidence_chunks=[
            EvidenceChunk(source_section="项目", text="Built a Python dashboard."),
            EvidenceChunk(source_section="联系方式", text="candidate@example.test"),
        ],
    )


def _analysis() -> MatchAnalysis:
    return MatchAnalysis(
        matches=[
            RequirementMatch(
                requirement_id="req_001",
                status=MatchStatus.matched,
                resume_evidence=["Built a Python dashboard."],
                explanation="有直接证据。",
                confidence=0.9,
            )
        ],
        resume_suggestions=[
            ResumeSuggestion(
                original_text="Built a Python dashboard.",
                suggested_text="Built a Python dashboard for business reporting.",
                requirement_ids=["req_001"],
                reason="表述用途。",
                follow_up_question=None,
            )
        ],
    )


def test_application_evidence_is_deduplicated_and_redacted() -> None:
    evidence = collect_application_evidence(_resume(), _analysis())

    assert len([item for item in evidence if "Python dashboard" in item.text]) == 1
    assert "candidate@example.test" not in evidence[1].text
    assert "[已隐藏邮箱]" in evidence[1].text


def test_keyword_gaps_follow_match_status() -> None:
    job = JobProfile(
        company="示例公司",
        title="数据岗位",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[
            JobRequirement(
                id="req_001",
                original_text="Python",
                normalized_name="Python",
                category=RequirementCategory.skill,
                importance=RequirementImportance.must_have,
                is_hard_condition=False,
            )
        ],
        domain_background=[],
    )

    result = analyse_keyword_gaps(job, _analysis())

    assert result[0].keyword == "Python"
    assert result[0].status is MatchStatus.matched


def test_only_explicitly_accepted_resume_edits_are_exported() -> None:
    result = accepted_resume_suggestions(
        _analysis(),
        {"0": {"decision": "accepted", "text": "Final tailored statement."}},
    )

    assert result == [("Built a Python dashboard.", "Final tailored statement.")]


def test_unresolved_placeholders_are_never_exported() -> None:
    result = accepted_resume_suggestions(
        _analysis(),
        {"0": {"decision": "accepted", "text": "Used Python in [具体项目]。"}},
    )

    assert result == []


def test_resume_diff_highlights_changes_and_escapes_html() -> None:
    rendered = render_resume_diff_html("Used Python <script>", "Used Python for reporting")

    assert "<mark>" in rendered
    assert "<del>" in rendered
    assert "<script>" not in rendered
    assert "&lt;" in rendered
