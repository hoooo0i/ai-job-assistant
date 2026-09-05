import pytest

from src.evidence_flow import (
    EvidenceFlowError,
    apply_answers_to_final_analysis,
    facts_from_answers,
    facts_from_supplement_details,
    invalidate_generated_materials,
    merge_candidate_facts,
    prepare_clarification_questions,
    sanitise_supplement_drafts,
    select_important_supplements,
    validate_clarification_answers,
)
from src.schemas import (
    CandidateFact,
    ClarificationAnswer,
    ClarificationQuestion,
    JobProfile,
    JobRequirement,
    MatchAnalysis,
    MatchStatus,
    PreliminaryAnalysis,
    RequirementCategory,
    RequirementImportance,
    RequirementMatch,
    SupplementDetail,
)


def _requirement(
    identifier: str,
    category: RequirementCategory = RequirementCategory.skill,
    *,
    hard: bool = False,
    importance: RequirementImportance = RequirementImportance.must_have,
) -> JobRequirement:
    return JobRequirement(
        id=identifier,
        original_text=f"Requirement {identifier}",
        normalized_name=f"Skill {identifier}",
        category=category,
        importance=importance,
        is_hard_condition=hard,
    )


def _match(identifier: str, status: MatchStatus = MatchStatus.missing) -> RequirementMatch:
    return RequirementMatch(
        requirement_id=identifier,
        status=status,
        resume_evidence=[],
        explanation="No evidence.",
        confidence=0.8,
    )


def _job(requirements: list[JobRequirement]) -> JobProfile:
    return JobProfile(
        company="Example",
        title="Role",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=requirements,
        domain_background=[],
    )


def test_questions_only_cover_unresolved_requirements_and_are_limited() -> None:
    requirements = [
        _requirement("req_001", hard=True),
        *[_requirement(f"req_{index:03d}") for index in range(2, 7)],
        _requirement("req_007"),
    ]
    matches = [_match(item.id) for item in requirements]
    matches[-1] = _match("req_007", MatchStatus.matched)
    questions = [
        ClarificationQuestion(id=f"cq_{item.id}", requirement_id=item.id, prompt=item.original_text)
        for item in requirements
    ]
    preliminary = PreliminaryAnalysis(matches=matches, clarification_questions=questions)

    result = prepare_clarification_questions(preliminary, _job(requirements))

    assert len(result) == 5
    assert result[0].requirement_id == "req_001"
    assert all(item.requirement_id != "req_007" for item in result)


def test_choice_only_answers_do_not_require_detailed_evidence() -> None:
    skill = _requirement("req_001")
    availability = _requirement(
        "req_002",
        RequirementCategory.availability,
        hard=True,
    )
    job = _job([skill, availability])
    questions = [
        ClarificationQuestion(id="cq_1", requirement_id=skill.id, prompt="Skill?"),
        ClarificationQuestion(id="cq_2", requirement_id=availability.id, prompt="Available?"),
    ]

    result = validate_clarification_answers(
        [
            ClarificationAnswer(
                question_id="cq_1",
                requirement_id=skill.id,
                status="have",
            ),
            ClarificationAnswer(
                question_id="cq_2",
                requirement_id=availability.id,
                status="have",
            )
        ],
        questions,
        job,
    )
    assert [item.status for item in result] == ["have", "have"]


def test_detailed_supplement_requires_content_only_when_ai_rewrite_is_requested() -> None:
    skill = _requirement("req_001")
    job = _job([skill])
    draft = SupplementDetail(requirement_id=skill.id, situation="课程", action="分析数据")

    saved = sanitise_supplement_drafts([draft], job, {skill.id})
    assert saved[0].situation == "课程"

    with pytest.raises(EvidenceFlowError, match="至少需要 20"):
        sanitise_supplement_drafts(
            [draft],
            job,
            {skill.id},
            require_complete=True,
        )


def test_important_supplements_prioritise_confirmed_partial_requirements() -> None:
    preferred = _requirement(
        "req_001",
        importance=RequirementImportance.preferred,
    )
    must = _requirement("req_002")
    availability = _requirement(
        "req_003",
        RequirementCategory.availability,
        hard=True,
    )
    job = _job([preferred, must, availability])
    analysis = MatchAnalysis(matches=[_match(item.id, MatchStatus.partial) for item in job.requirements])
    answers = [
        ClarificationAnswer(question_id=f"cq_{item.id}", requirement_id=item.id, status="have")
        for item in job.requirements
    ]

    result = select_important_supplements(job, analysis, answers)

    assert [item.id for item in result] == [must.id, preferred.id]


def test_detailed_supplement_becomes_user_confirmed_fact() -> None:
    skill = _requirement("req_001")
    detail = SupplementDetail(
        requirement_id=skill.id,
        situation="课程数据分析项目",
        action="使用 Python 清洗数据并制作可视化报告",
        result="完成课堂展示",
        metrics=None,
    )

    facts = facts_from_supplement_details([detail], _job([skill]), "job_a")

    assert facts[0].source_job_id == "job_a"
    assert "使用 Python" in facts[0].statement


def test_candidate_supplied_evidence_is_redacted_before_storage() -> None:
    skill = _requirement("req_001")
    job = _job([skill])
    question = ClarificationQuestion(id="cq_1", requirement_id=skill.id, prompt="Skill?")
    answer = ClarificationAnswer(
        question_id=question.id,
        requirement_id=skill.id,
        status="have",
        evidence_text=(
            "Built a Python workflow for synthetic reporting and shared the result. "
            "candidate@example.test"
        ),
    )

    result = validate_clarification_answers([answer], [question], job)

    assert "candidate@example.test" not in result[0].evidence_text
    assert "[已隐藏邮箱]" in result[0].evidence_text


def test_confirmed_fact_is_reusable_and_keeps_user_source() -> None:
    requirement = _requirement("req_001")
    job = _job([requirement])
    answer = ClarificationAnswer(
        question_id="cq_1",
        requirement_id=requirement.id,
        status="have",
        evidence_text="Built a Python workflow for a synthetic reporting project.",
        metrics="Processed 500 synthetic records.",
    )
    facts = facts_from_answers([answer], job, "job_a")
    merged = merge_candidate_facts([], facts, "job_a")
    analysis = MatchAnalysis(matches=[_match(requirement.id)])

    result = apply_answers_to_final_analysis(analysis, [answer], merged, job)

    assert merged[0].source_job_id == "job_a"
    assert result.matches[0].status is MatchStatus.partial
    assert result.matches[0].evidence[0].source.value == "user_confirmed"
    assert result.matches[0].evidence[0].fact_id == merged[0].id


def test_direct_condition_and_negative_choices_update_locally() -> None:
    availability = _requirement(
        "req_001",
        RequirementCategory.availability,
        hard=True,
    )
    missing_skill = _requirement("req_002")
    unsure_skill = _requirement("req_003")
    job = _job([availability, missing_skill, unsure_skill])
    answers = [
        ClarificationAnswer(question_id="cq_1", requirement_id=availability.id, status="have"),
        ClarificationAnswer(question_id="cq_2", requirement_id=missing_skill.id, status="not_have"),
        ClarificationAnswer(question_id="cq_3", requirement_id=unsure_skill.id, status="unsure"),
    ]
    facts = facts_from_answers(answers, job, "job_a")
    analysis = MatchAnalysis(matches=[_match(item.id) for item in job.requirements])

    result = apply_answers_to_final_analysis(analysis, answers, facts, job)

    assert [item.status for item in result.matches] == [
        MatchStatus.matched,
        MatchStatus.missing,
        MatchStatus.unknown,
    ]


def test_unanswered_requirement_keeps_preliminary_status() -> None:
    requirement = _requirement("req_001")
    job = _job([requirement])
    answer = ClarificationAnswer(
        question_id="cq_1",
        requirement_id=requirement.id,
        status="unanswered",
    )
    analysis = MatchAnalysis(matches=[_match(requirement.id, MatchStatus.missing)])

    final_model_result = MatchAnalysis(matches=[_match(requirement.id, MatchStatus.partial)])
    preliminary = PreliminaryAnalysis(
        matches=analysis.matches,
        clarification_questions=[],
    )

    result = apply_answers_to_final_analysis(
        final_model_result,
        [answer],
        [],
        job,
        preliminary,
    )

    assert result.matches[0].status is MatchStatus.missing


def test_replacing_one_jobs_facts_preserves_other_jobs_facts() -> None:
    old_a = CandidateFact(
        id="fact_a",
        category=RequirementCategory.skill,
        statement="Old A",
        metrics=None,
        source_job_id="job_a",
        source_requirement_text="A",
    )
    fact_b = old_a.model_copy(
        update={"id": "fact_b", "statement": "B", "source_job_id": "job_b"}
    )
    new_a = old_a.model_copy(update={"id": "fact_a2", "statement": "New A"})

    result = merge_candidate_facts([old_a, fact_b], [new_a], "job_a")

    assert {item.id for item in result} == {"fact_a2", "fact_b"}


def test_invalidating_materials_keeps_preliminary_state() -> None:
    bundle = {
        "preliminary_analysis": {"matches": []},
        "final_analysis": {"matches": []},
        "report_files": {"pdf": b"data"},
        "cover_letters": {"zh": {}},
        "stage": "final",
    }

    invalidate_generated_materials(bundle)

    assert bundle["stage"] == "clarification"
    assert "preliminary_analysis" in bundle
    assert "final_analysis" not in bundle
    assert "report_files" not in bundle
