from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Optional

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from src.ai_parser import (
    AiParserError,
    create_ai_provider,
    generate_supplement_resume_suggestions,
    get_model_name,
    generate_cover_letter,
    has_api_key,
    parse_job,
    parse_resume,
    preliminary_match_requirements,
    prepare_interview_answer,
    review_interview_answer,
)
from src.ai_provider import AiProviderError, get_provider_name
from src.application_tracker import (
    STATUS_LABELS as APPLICATION_STATUS_LABELS,
    ApplicationTrackerError,
    build_application_metrics,
    build_application_record,
    get_application_record,
    upcoming_application_actions,
)
from src.application_package import build_application_package
from src.archive import (
    WorkspaceArchiveError,
    build_workspace_archive,
    load_workspace_archive,
)
from src.ats_checker import build_ats_report, contains_contact_details, inspect_pdf_layout
from src.career_tools import (
    accepted_resume_suggestions,
    analyse_keyword_gaps,
    contains_unresolved_placeholder,
    render_resume_diff_html,
)
from src.comparison import build_job_comparison
from src.evidence_flow import (
    EvidenceFlowError,
    apply_answers_to_final_analysis,
    facts_from_answers,
    facts_from_supplement_details,
    invalidate_generated_materials,
    merge_candidate_facts,
    sanitise_supplement_drafts,
    select_important_supplements,
    validate_clarification_answers,
)
from src.interview import InterviewEvidence, collect_interview_evidence
from src.job_link import JobLinkError, fetch_job_posting
from src.matching import calculate_scores
from src.pdf_parser import (
    EncryptedPdfError,
    InvalidPdfError,
    PdfExtractionResult,
    PdfReadError,
    extract_pdf_text,
)
from src.privacy import redact_sensitive_info
from src.reporting import (
    build_cover_letter_docx,
    build_cover_letter_pdf,
    build_docx_report,
    build_job_comparison_docx,
    build_job_comparison_pdf,
    build_pdf_report,
    build_tailored_resume_docx,
)
from src.resume_versions import (
    ResumeVersionError,
    add_resume_version,
    create_resume_version,
    restore_resume_decisions,
)
from src.schemas import (
    CandidateFact,
    ApplicationStatus,
    ClarificationAnswer,
    ClarificationQuestion,
    CoverLetterDraft,
    InterviewFeedback,
    InterviewPreparation,
    JobProfile,
    MatchAnalysis,
    MatchStatus,
    PreliminaryAnalysis,
    PdfLayoutSignals,
    ResumeProfile,
    ResumeVersion,
    ScoreResult,
    SupplementDetail,
)
from src.submission import build_submission_checklist, safe_resume_filename
from src.validators import (
    JobInput,
    InputValidationError,
    has_valid_resume_text,
    select_resume_text,
    validate_pdf_upload,
)


RESUME_PREVIEW_LIMIT = 2_000
JD_PREVIEW_LIMIT = 1_500

STATUS_DISPLAY = {
    MatchStatus.matched: ("✅", "已匹配"),
    MatchStatus.partial: ("◐", "部分匹配"),
    MatchStatus.missing: ("❌", "缺失"),
    MatchStatus.unknown: ("❓", "待确认"),
}

STAGE_STEPS = ["上传资料", "初步匹配", "补充真实信息", "最终建议与材料"]
ANALYSIS_CACHE_VERSION = "interactive-v2"
MAX_JOBS_PER_SESSION = 5


class _SessionCountingProvider:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.name = delegate.name
        self.model = delegate.model

    def parse(self, **kwargs):
        st.session_state["model_call_count"] = (
            st.session_state.get("model_call_count", 0) + 1
        )
        return self._delegate.parse(**kwargs)


def _create_counted_provider() -> _SessionCountingProvider:
    return _SessionCountingProvider(create_ai_provider())


class _LazySessionProvider:
    def __init__(self) -> None:
        self._provider: Optional[_SessionCountingProvider] = None

    @property
    def name(self) -> str:
        return get_provider_name()

    @property
    def model(self) -> str:
        return get_model_name()

    def parse(self, **kwargs):
        if self._provider is None:
            self._provider = _create_counted_provider()
        return self._provider.parse(**kwargs)


def _save_active_job_analysis(job_analysis: dict) -> None:
    job_id = st.session_state.get("active_job_id")
    if not job_id:
        return
    analyses = st.session_state.setdefault("job_analyses", {})
    analyses[job_id] = job_analysis


def _render_step_progress(active_step: int) -> None:
    st.progress(active_step / len(STAGE_STEPS))
    st.caption(
        "  →  ".join(
            f"**{index}. {label}**" if index == active_step else f"{index}. {label}"
            for index, label in enumerate(STAGE_STEPS, start=1)
        )
    )


def _preview(text: str, limit: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}\n\n……（预览已截断）"


def _analysis_fingerprint(
    uploaded_file,
    company: str,
    job_title: str,
    location: str,
    job_type: str,
    jd_text: str,
    fallback_text: str,
) -> str:
    digest = hashlib.sha256()
    values = [company, job_title, location, job_type, jd_text, fallback_text]
    if uploaded_file is not None:
        digest.update(uploaded_file.name.encode("utf-8", errors="ignore"))
        digest.update(uploaded_file.getvalue())
    for value in values:
        digest.update(b"\x00")
        digest.update(value.encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def _resume_fingerprint(uploaded_file, fallback_text: str) -> str:
    digest = hashlib.sha256()
    if uploaded_file is not None:
        digest.update(uploaded_file.name.encode("utf-8", errors="ignore"))
        digest.update(uploaded_file.getvalue())
    digest.update(fallback_text.encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def _job_fingerprint(job: JobInput) -> str:
    digest = hashlib.sha256()
    for value in [job.company, job.job_title, job.location, job.job_type, job.jd_text]:
        digest.update(b"\x00")
        digest.update((value or "").encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def _model_cache_key(kind: str, *parts: str) -> str:
    digest = hashlib.sha256()
    for value in [
        ANALYSIS_CACHE_VERSION,
        get_provider_name(),
        get_model_name(),
        kind,
        *parts,
    ]:
        digest.update(b"\x00")
        digest.update(value.encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def _analysis_cache() -> dict:
    cache = st.session_state.setdefault("analysis_cache", {})
    for section in ["resumes", "jobs", "initial"]:
        cache.setdefault(section, {})
    return cache


def _cached_resume_profile(
    resume_id: str,
    resume_text: str,
    provider: _LazySessionProvider,
) -> tuple[ResumeProfile, bool]:
    cache = _analysis_cache()["resumes"]
    key = _model_cache_key("resume", resume_id)
    if key in cache:
        return ResumeProfile.model_validate(cache[key]), True
    profile = parse_resume(resume_text, provider=provider)
    cache[key] = profile.model_dump(mode="json")
    return profile, False


def _cached_job_analysis(
    resume_id: str,
    resume_profile: ResumeProfile,
    resume_text: str,
    job: JobInput,
    provider: _LazySessionProvider,
) -> tuple[str, JobProfile, PreliminaryAnalysis, tuple[bool, bool]]:
    cache = _analysis_cache()
    job_id = _job_fingerprint(job)
    job_key = _model_cache_key("job", job_id)
    initial_key = _model_cache_key("initial", resume_id, job_id)
    job_cached = job_key in cache["jobs"]
    if job_cached:
        job_profile = JobProfile.model_validate(cache["jobs"][job_key])
    else:
        job_profile = parse_job(job, provider=provider)
        cache["jobs"][job_key] = job_profile.model_dump(mode="json")
    initial_cached = initial_key in cache["initial"]
    if initial_cached:
        preliminary = PreliminaryAnalysis.model_validate(cache["initial"][initial_key])
    else:
        preliminary = preliminary_match_requirements(
            resume_profile,
            job_profile,
            resume_text,
            provider=provider,
        )
        cache["initial"][initial_key] = preliminary.model_dump(mode="json")
    return job_id, job_profile, preliminary, (job_cached, initial_cached)


def _new_job_bundle(
    job_id: str,
    job_profile: JobProfile,
    preliminary: PreliminaryAnalysis,
    job_url: str | None = None,
) -> dict:
    return {
        "job_id": job_id,
        "fingerprint": job_id,
        "job_profile": job_profile.model_dump(mode="json"),
        "preliminary_analysis": preliminary.model_dump(mode="json"),
        "clarification_questions": [
            item.model_dump(mode="json") for item in preliminary.clarification_questions
        ],
        "clarification_answers": [],
        "application_tracking": {
            "status": "not_started",
            "job_url": job_url,
        },
        "stage": "clarification",
    }


def _format_job_errors(exc: ValidationError) -> list[str]:
    labels = {
        "company": "公司名称",
        "job_title": "岗位名称",
        "jd_text": "岗位 JD",
        "location": "工作地点",
        "job_type": "岗位类型",
        "job_url": "岗位链接",
    }
    messages: list[str] = []
    for error in exc.errors(include_url=False):
        field = str(error["loc"][0]) if error["loc"] else "输入"
        message = str(error["msg"]).removeprefix("Value error, ")
        messages.append(f"{labels.get(field, field)}：{message}")
    return messages


def _parse_uploaded_pdf(uploaded_file) -> tuple[Optional[PdfExtractionResult], Optional[str]]:
    if uploaded_file is None:
        return None, None

    try:
        validate_pdf_upload(uploaded_file.name, uploaded_file.size)
        return extract_pdf_text(uploaded_file.getvalue(), uploaded_file.name), None
    except (InputValidationError, InvalidPdfError, EncryptedPdfError, PdfReadError) as exc:
        return None, str(exc)


def _render_resume_profile(profile: ResumeProfile) -> None:
    if profile.summary:
        st.markdown(f"**简历摘要：** {profile.summary}")

    metric_columns = st.columns(4)
    metric_columns[0].metric("教育经历", len(profile.education))
    metric_columns[1].metric("工作经历", len(profile.experience))
    metric_columns[2].metric("项目经历", len(profile.projects))
    metric_columns[3].metric("证据片段", len(profile.evidence_chunks))

    with st.expander("教育经历", expanded=bool(profile.education)):
        if not profile.education:
            st.caption("未从简历中提取到教育经历。")
        for item in profile.education:
            title = " · ".join(filter(None, [item.institution, item.degree, item.field_of_study]))
            st.markdown(f"**{title}**")
            st.caption(" - ".join(filter(None, [item.start_date, item.end_date])))
            for highlight in item.highlights:
                st.write(f"- {highlight}")

    with st.expander("工作与实习经历", expanded=bool(profile.experience)):
        if not profile.experience:
            st.caption("未从简历中提取到工作或实习经历。")
        for item in profile.experience:
            st.markdown(f"**{item.organization} · {item.title or '未注明岗位'}**")
            st.caption(" ｜ ".join(filter(None, [item.start_date, item.end_date, item.location])))
            for bullet in item.bullets:
                st.write(f"- {bullet}")

    with st.expander("项目与技能"):
        for project in profile.projects:
            st.markdown(f"**{project.name}**{f' · {project.role}' if project.role else ''}")
            for bullet in project.bullets:
                st.write(f"- {bullet}")
            if project.technologies:
                st.caption("技术：" + "、".join(project.technologies))
        for group in profile.skills:
            st.markdown(f"**{group.category}：** " + "、".join(group.skills))
        if profile.languages:
            st.markdown("**语言：** " + "、".join(profile.languages))

    with st.expander("简历证据片段"):
        if not profile.evidence_chunks:
            st.caption("未提取到证据片段。")
        for evidence in profile.evidence_chunks:
            st.markdown(f"**{evidence.source_section}**")
            st.write(f"> {evidence.text}")


def _render_job_profile(profile: JobProfile) -> None:
    st.markdown(f"**{profile.company} · {profile.title}**")
    st.caption(f"工作地点：{profile.location or '未提供'} ｜ 岗位类型：{profile.job_type or '未提供'}")

    metric_columns = st.columns(3)
    metric_columns[0].metric("岗位职责", len(profile.responsibilities))
    metric_columns[1].metric("岗位要求", len(profile.requirements))
    metric_columns[2].metric(
        "硬性条件", sum(requirement.is_hard_condition for requirement in profile.requirements)
    )

    with st.expander("岗位职责", expanded=True):
        if not profile.responsibilities:
            st.caption("未从 JD 中提取到明确职责。")
        for responsibility in profile.responsibilities:
            st.write(f"- {responsibility}")

    with st.expander("岗位要求", expanded=True):
        if not profile.requirements:
            st.caption("未从 JD 中提取到明确要求。")
        for requirement in profile.requirements:
            hard_condition = " · 硬性条件" if requirement.is_hard_condition else ""
            st.markdown(
                f"**{requirement.id} · {requirement.normalized_name}**  "
                f"`{requirement.importance.value}` · `{requirement.category.value}`{hard_condition}"
            )
            st.write(requirement.original_text)

    if profile.domain_background:
        with st.expander("领域背景"):
            for item in profile.domain_background:
                st.write(f"- {item}")


def _render_match_analysis(
    analysis: MatchAnalysis,
    score: ScoreResult,
    job_profile: JobProfile,
) -> None:
    st.warning("证据匹配度不是录取概率，也不是 ATS 通过率。请结合逐项证据和信息完整度判断。")

    score_columns = st.columns(2)
    score_columns[0].metric(
        "证据匹配度",
        "暂不可计算" if score.match_score is None else f"{score.match_score:.1f}%",
    )
    score_columns[1].metric("信息完整度", f"{score.information_completeness:.1f}%")
    st.caption(
        f"已知要求权重：{score.known_weight}/{score.total_weight} ｜ "
        f"评分规则版本：{score.calculation_version}"
    )

    status_counts = {
        status: sum(match.status is status for match in analysis.matches)
        for status in MatchStatus
    }
    count_columns = st.columns(4)
    for column, status in zip(count_columns, MatchStatus):
        icon, label = STATUS_DISPLAY[status]
        column.metric(f"{icon} {label}", status_counts[status])

    requirements_by_id = {
        requirement.id: requirement for requirement in job_profile.requirements
    }
    for match in analysis.matches:
        requirement = requirements_by_id[match.requirement_id]
        icon, label = STATUS_DISPLAY[match.status]
        with st.expander(
            f"{icon} {label} · {requirement.normalized_name}",
            expanded=match.status in {MatchStatus.missing, MatchStatus.unknown},
        ):
            st.caption(
                f"重要程度：{requirement.importance.value} ｜ "
                f"类别：{requirement.category.value} ｜ "
                f"置信度：{match.confidence:.0%}"
            )
            st.markdown("**JD 原文**")
            st.write(requirement.original_text)
            st.markdown("**判断说明**")
            st.write(match.explanation)
            st.markdown("**匹配证据**")
            if match.evidence:
                for evidence in match.evidence:
                    source = "用户确认" if evidence.source.value == "user_confirmed" else "简历"
                    st.write(f"> [{source}] {evidence.text}")
            elif match.resume_evidence:
                for evidence in match.resume_evidence:
                    st.write(f"> [简历] {evidence}")
            else:
                st.caption("没有可验证的简历原文证据。")
            if match.status is MatchStatus.unknown:
                st.info(f"待确认：你是否满足“{requirement.original_text}”？")


def _invalidate_final_derivatives(bundle: dict) -> None:
    for key in [
        "resume_edits",
        "tailored_resume_file",
        "cover_letters",
        "interview_preparations",
        "interview_feedback",
        "report_files",
        "application_package",
    ]:
        bundle.pop(key, None)


def _render_ats_report(
    candidate_profile: dict,
    job_profile: JobProfile,
    analysis: MatchAnalysis,
) -> None:
    layout = PdfLayoutSignals.model_validate(
        candidate_profile.get(
            "pdf_layout",
            {
                "page_count": candidate_profile.get("page_count", 0) or 0,
                "readable": candidate_profile.get("resume_source") == "pdf",
            },
        )
    )
    report = build_ats_report(
        candidate_profile["resume_text"],
        job_profile,
        analysis,
        layout,
        candidate_profile.get("resume_source", "pdf"),
    )
    st.warning(
        "ATS 体检是本地启发式检查，不代表任何招聘平台的实际筛选结果。"
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("ATS 可读性评分", f"{report.score}/100")
    metric_columns[1].metric("JD 关键词直接覆盖", f"{report.keyword_coverage:.1f}%")
    metric_columns[2].metric(
        "需要处理",
        sum(item.severity != "passed" for item in report.checks),
    )
    labels = {
        "critical": "🔴 严重",
        "warning": "🟡 建议修改",
        "passed": "✅ 正常",
    }
    for severity in ["critical", "warning", "passed"]:
        checks = [item for item in report.checks if item.severity == severity]
        with st.expander(f"{labels[severity]} · {len(checks)} 项", expanded=severity == "critical"):
            if not checks:
                st.caption("无")
            for item in checks:
                st.markdown(f"**{item.title}**")
                st.write(item.detail)
                if item.recommendation:
                    st.caption(f"建议：{item.recommendation}")


def _render_important_supplements(
    bundle: dict,
    analysis: MatchAnalysis,
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
    candidate_profile: dict,
) -> None:
    answers = [
        ClarificationAnswer.model_validate(item)
        for item in bundle.get("clarification_answers", [])
    ]
    requirements = select_important_supplements(job_profile, analysis, answers)
    if not requirements:
        return

    saved_details = {
        item["requirement_id"]: SupplementDetail.model_validate(item)
        for item in bundle.get("supplement_details", [])
    }
    st.markdown("### 重要信息待补充")
    st.caption(
        "这些项目对岗位较重要，但现有证据仍不够具体。你可以先保存草稿；"
        "准备好后一次批量生成强化表述。"
    )
    submitted: list[SupplementDetail] = []
    with st.form(f"important_supplements_{bundle['fingerprint']}"):
        for index, requirement in enumerate(requirements, start=1):
            saved = saved_details.get(requirement.id)
            st.markdown(f"#### {index}. {requirement.normalized_name}")
            st.write(requirement.original_text)
            st.caption(
                f"安全写法：具备“{requirement.normalized_name}”相关能力（用户确认）。"
                f"强化草稿：[具体情境]中通过[行动]完成[结果]。"
            )
            situation = st.text_input(
                f"具体情境或项目（补充 {index}）",
                value=saved.situation if saved else "",
                key=f"supplement_situation_{bundle['fingerprint']}_{requirement.id}",
            )
            action = st.text_area(
                f"你的行动（补充 {index}）",
                value=saved.action if saved else "",
                height=90,
                key=f"supplement_action_{bundle['fingerprint']}_{requirement.id}",
            )
            result = st.text_input(
                f"结果（补充 {index}）",
                value=saved.result if saved else "",
                key=f"supplement_result_{bundle['fingerprint']}_{requirement.id}",
            )
            metrics = st.text_input(
                f"可核对的数据（可选，补充 {index}）",
                value=saved.metrics or "" if saved else "",
                key=f"supplement_metrics_{bundle['fingerprint']}_{requirement.id}",
            )
            submitted.append(
                SupplementDetail(
                    requirement_id=requirement.id,
                    situation=situation,
                    action=action,
                    result=result,
                    metrics=metrics or None,
                )
            )
        save_drafts = st.form_submit_button("保存补充草稿")
        generate_batch = st.form_submit_button("AI 批量优化已填写内容", type="primary")

    allowed_ids = {item.id for item in requirements}
    if save_drafts:
        try:
            cleaned = sanitise_supplement_drafts(
                submitted,
                job_profile,
                allowed_ids,
            )
            bundle["supplement_details"] = [
                item.model_dump(mode="json") for item in cleaned
            ]
            _save_active_job_analysis(bundle)
            st.session_state["workspace_notice"] = "重点补充草稿已保存在当前会话。"
            st.rerun()
        except EvidenceFlowError as exc:
            st.error(str(exc))

    if generate_batch:
        try:
            cleaned = sanitise_supplement_drafts(
                submitted,
                job_profile,
                allowed_ids,
                require_complete=True,
            )
            preliminary = PreliminaryAnalysis.model_validate(bundle["preliminary_analysis"])
            choice_facts = facts_from_answers(answers, job_profile, bundle["job_id"])
            detailed_facts = facts_from_supplement_details(
                cleaned,
                job_profile,
                bundle["job_id"],
            )
            replacements = {
                item.source_requirement_text: item for item in choice_facts
            }
            replacements.update(
                {item.source_requirement_text: item for item in detailed_facts}
            )
            existing_facts = [
                CandidateFact.model_validate(item)
                for item in candidate_profile.get("facts", [])
            ]
            candidate_facts = merge_candidate_facts(
                existing_facts,
                list(replacements.values()),
                bundle["job_id"],
            )
            local_analysis = apply_answers_to_final_analysis(
                MatchAnalysis(
                    matches=preliminary.matches,
                    resume_suggestions=preliminary.resume_suggestions,
                    interview_questions=preliminary.interview_questions,
                ),
                answers,
                candidate_facts,
                job_profile,
                preliminary,
            )
            provider = _create_counted_provider()
            with st.spinner("正在批量生成基于真实补充内容的简历表述……"):
                generated = generate_supplement_resume_suggestions(
                    resume_profile,
                    job_profile,
                    candidate_profile["resume_text"],
                    local_analysis,
                    candidate_facts,
                    [item.requirement_id for item in cleaned],
                    provider=provider,
                )
            combined_suggestions = list(preliminary.resume_suggestions)
            seen = {
                (item.original_text.casefold(), item.suggested_text.casefold())
                for item in combined_suggestions
            }
            for item in generated:
                identity = (item.original_text.casefold(), item.suggested_text.casefold())
                if identity not in seen:
                    combined_suggestions.append(item)
                    seen.add(identity)
            final_analysis = local_analysis.model_copy(
                update={"resume_suggestions": combined_suggestions}
            )
            facts_changed = [
                item.model_dump(mode="json") for item in existing_facts
            ] != [item.model_dump(mode="json") for item in candidate_facts]
            candidate_profile["facts"] = [
                item.model_dump(mode="json") for item in candidate_facts
            ]
            st.session_state["candidate_profile"] = candidate_profile
            if facts_changed:
                for job_id, other_bundle in st.session_state.setdefault(
                    "job_analyses", {}
                ).items():
                    if job_id != bundle["job_id"] and other_bundle.get("stage") == "final":
                        invalidate_generated_materials(other_bundle)
            bundle["supplement_details"] = [
                item.model_dump(mode="json") for item in cleaned
            ]
            bundle["final_analysis"] = final_analysis.model_dump(mode="json")
            _invalidate_final_derivatives(bundle)
            bundle["supplement_suggestions"] = [
                item.model_dump(mode="json") for item in generated
            ]
            _save_active_job_analysis(bundle)
            st.session_state["workspace_notice"] = (
                f"已用 1 次模型调用批量优化 {len(generated)} 条简历表述。"
            )
            st.rerun()
        except (EvidenceFlowError, AiParserError, AiProviderError) as exc:
            st.error(str(exc))


def _render_resume_actions(
    bundle: dict,
    analysis: MatchAnalysis,
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
    fingerprint: str,
    candidate_profile: dict,
) -> None:
    requirements_by_id = {
        requirement.id: requirement for requirement in job_profile.requirements
    }
    user_fact_texts = [
        evidence.text.casefold()
        for match in analysis.matches
        for evidence in match.evidence
        if evidence.source.value == "user_confirmed"
    ]

    _render_important_supplements(
        bundle,
        analysis,
        resume_profile,
        job_profile,
        candidate_profile,
    )
    st.markdown("### 简历优化建议")
    st.caption(
        "建议只能重组或强化已有事实。你可以逐条采纳、编辑或忽略，"
        "只有明确采纳的内容才会进入定制简历。"
    )
    if not analysis.resume_suggestions:
        st.caption("本次没有通过证据校验的简历改写建议。")
    decisions = bundle.setdefault("resume_edits", {})
    if analysis.resume_suggestions:
        decision_options = ["待决定", "采纳", "忽略"]
        decision_values = {"待决定": "pending", "采纳": "accepted", "忽略": "ignored"}
        reverse_decisions = {value: label for label, value in decision_values.items()}
        submitted_decisions: dict[str, dict[str, str]] = {}
        with st.form(f"resume_workspace_{fingerprint}"):
            for index, suggestion in enumerate(analysis.resume_suggestions, start=1):
                key = str(index - 1)
                related_names = [
                    requirements_by_id[identifier].normalized_name
                    for identifier in suggestion.requirement_ids
                ]
                st.markdown(f"#### 建议 {index} · {'、'.join(related_names)}")
                source_label = (
                    "用户确认事实"
                    if any(suggestion.original_text.casefold() in text for text in user_fact_texts)
                    else "简历原文"
                )
                st.write(f"**{source_label}：** {suggestion.original_text}")
                st.caption(f"修改理由：{suggestion.reason}")
                current = decisions.get(key, {})
                selected = st.radio(
                    f"处理建议 {index}",
                    decision_options,
                    index=decision_options.index(
                        reverse_decisions.get(current.get("decision", "pending"), "待决定")
                    ),
                    horizontal=True,
                    key=f"resume_decision_{fingerprint}_{key}",
                )
                edited_text = st.text_area(
                    f"建议 {index} 的最终表述",
                    value=current.get("text", suggestion.suggested_text),
                    height=110,
                    key=f"resume_edit_{fingerprint}_{key}",
                )
                with st.expander(f"查看建议 {index} 的修改对比"):
                    st.markdown(
                        render_resume_diff_html(suggestion.original_text, edited_text),
                        unsafe_allow_html=True,
                    )
                    st.caption("删除内容使用删除线，新增内容使用高亮显示。")
                submitted_decisions[key] = {
                    "decision": decision_values[selected],
                    "text": edited_text,
                }
                if suggestion.follow_up_question:
                    st.info(f"需要你补充确认：{suggestion.follow_up_question}")
            save_decisions = st.form_submit_button("保存简历优化选择", type="primary")
        if save_decisions:
            bundle["resume_edits"] = submitted_decisions
            bundle.pop("tailored_resume_file", None)
            bundle.pop("application_package", None)
            _save_active_job_analysis(bundle)
            accepted_count = sum(
                item["decision"] == "accepted" for item in submitted_decisions.values()
            )
            st.session_state["workspace_notice"] = (
                f"已保存简历优化选择，当前采纳 {accepted_count} 项。"
            )
            st.rerun()

    accepted = accepted_resume_suggestions(analysis, decisions)
    blocked_placeholders = sum(
        record.get("decision") == "accepted"
        and contains_unresolved_placeholder(record.get("text", ""))
        for record in decisions.values()
    )
    if blocked_placeholders:
        st.warning(
            f"有 {blocked_placeholders} 条已采纳内容仍包含待填写占位符，"
            "系统不会把它们写入定制简历。"
        )
    accepted_decision_count = sum(
        record.get("decision") == "accepted" for record in decisions.values()
    )
    submission_confirmed = False
    checklist = None
    if accepted_decision_count:
        checklist = build_submission_checklist(
            analysis,
            decisions,
            job_profile,
            [
                CandidateFact.model_validate(item)
                for item in candidate_profile.get("facts", [])
            ],
        )
        st.markdown("### 投递前检查")
        for item in checklist.items:
            icon = "✅" if item.passed else "🔴" if item.blocking else "🟡"
            st.write(f"{icon} **{item.label}**：{item.detail}")
        if checklist.ready:
            submission_confirmed = st.checkbox(
                "我已核对以上内容，确认所有经历和数字真实准确",
                key=f"submission_confirmed_{fingerprint}_{hashlib.sha256(json.dumps(decisions, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()[:12]}",
            )
        else:
            st.error("请先修正阻断项，再下载定制简历。")

    if accepted and checklist is not None and checklist.ready and submission_confirmed:
        tailored_version = hashlib.sha256(
            json.dumps(accepted, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        tailored_file = bundle.get("tailored_resume_file", {})
        if tailored_file.get("version") != tailored_version:
            tailored_file = {
                "version": tailored_version,
                "docx": build_tailored_resume_docx(
                    resume_profile,
                    job_profile,
                    accepted,
                ),
            }
            bundle["tailored_resume_file"] = tailored_file
            _save_active_job_analysis(bundle)
        st.download_button(
            "下载定制简历 Word 草稿",
            data=tailored_file["docx"],
            file_name=safe_resume_filename(job_profile),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
        st.caption("导出文档不保留原 PDF 版式或联系方式，请在 Word 中核对并补充。")
        version_label = st.text_input(
            "版本名称",
            value=f"{job_profile.company}-{job_profile.title}-v{len(bundle.get('resume_versions', [])) + 1}",
            max_chars=60,
            key=f"resume_version_label_{fingerprint}",
            help="版本只保存在当前会话；可通过求职档案 JSON 备份版本记录。",
        )
        if st.button(
            "保存为新的简历版本",
            key=f"save_resume_version_{fingerprint}",
            use_container_width=True,
        ):
            try:
                version = create_resume_version(
                    bundle["job_id"],
                    version_label,
                    decisions,
                    accepted,
                )
                bundle["resume_versions"] = add_resume_version(
                    bundle.get("resume_versions", []),
                    version,
                )
                bundle.pop("application_package", None)
                _save_active_job_analysis(bundle)
                st.session_state["workspace_notice"] = (
                    f"已保存简历版本“{version.label}”。"
                )
                st.rerun()
            except ResumeVersionError as exc:
                st.error(str(exc))
    elif accepted and checklist is not None and checklist.ready:
        st.info("完成投递前人工核对后即可下载定制简历。")

    versions = [
        ResumeVersion.model_validate(item)
        for item in bundle.get("resume_versions", [])
    ]
    if versions:
        st.markdown("### 已保存的简历版本")
        st.caption("历史版本按岗位隔离；恢复只会更新当前编辑选择，不会调用模型。")
        for index, version in enumerate(versions, start=1):
            with st.expander(f"{version.label} · {version.created_at}"):
                st.write(f"已采纳 {len(version.accepted_suggestions)} 条建议")
                version_docx = build_tailored_resume_docx(
                    resume_profile,
                    job_profile,
                    version.accepted_suggestions,
                )
                columns = st.columns(3)
                columns[0].download_button(
                    "下载此版本",
                    data=version_docx,
                    file_name=safe_resume_filename(job_profile).replace(
                        ".docx", f"-version-{index}.docx"
                    ),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"download_resume_version_{fingerprint}_{version.id}",
                    use_container_width=True,
                )
                if columns[1].button(
                    "恢复到编辑器",
                    key=f"restore_resume_version_{fingerprint}_{version.id}",
                    use_container_width=True,
                ):
                    bundle["resume_edits"] = restore_resume_decisions(
                        version.model_dump(mode="json")
                    )
                    bundle.pop("tailored_resume_file", None)
                    bundle.pop("application_package", None)
                    for suggestion_index in range(len(analysis.resume_suggestions)):
                        st.session_state.pop(
                            f"resume_decision_{fingerprint}_{suggestion_index}", None
                        )
                        st.session_state.pop(
                            f"resume_edit_{fingerprint}_{suggestion_index}", None
                        )
                    _save_active_job_analysis(bundle)
                    st.session_state["workspace_notice"] = (
                        f"已恢复“{version.label}”的优化选择。"
                    )
                    st.rerun()
                if columns[2].button(
                    "删除此版本",
                    key=f"delete_resume_version_{fingerprint}_{version.id}",
                    use_container_width=True,
                ):
                    bundle["resume_versions"] = [
                        item
                        for item in bundle.get("resume_versions", [])
                        if item.get("id") != version.id
                    ]
                    bundle.pop("application_package", None)
                    _save_active_job_analysis(bundle)
                    st.session_state["workspace_notice"] = (
                        f"已从当前会话删除“{version.label}”。"
                    )
                    st.rerun()



def _render_keyword_gaps(job_profile: JobProfile, analysis: MatchAnalysis) -> None:
    gaps = analyse_keyword_gaps(job_profile, analysis)
    labels = {
        MatchStatus.matched: ("✅ 已覆盖", "简历有直接证据"),
        MatchStatus.partial: ("🟡 待强化", "有相关经验，但表述或深度不足"),
        MatchStatus.missing: ("🔴 能力缺口", "不建议靠改写弥补，需要真实学习或经验"),
        MatchStatus.unknown: ("❓ 待确认", "简历无法判断，需要本人确认"),
    }
    counts = {status: sum(item.status is status for item in gaps) for status in MatchStatus}
    columns = st.columns(4)
    for column, status in zip(columns, MatchStatus):
        column.metric(labels[status][0], counts[status])
    for status in [MatchStatus.missing, MatchStatus.partial, MatchStatus.unknown, MatchStatus.matched]:
        items = [item for item in gaps if item.status is status]
        with st.expander(f"{labels[status][0]} · {len(items)} 项", expanded=status is MatchStatus.missing):
            st.caption(labels[status][1])
            if not items:
                st.write("无")
            for item in items:
                st.markdown(f"**{item.keyword}**")
                st.write(item.explanation)


def _render_cover_letter(
    bundle: dict,
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
    analysis: MatchAnalysis,
) -> None:
    st.caption(
        "求职信每段都必须绑定简历证据；无证据段落和未经支持的数字会被自动移除。"
    )
    language_label = st.selectbox(
        "求职信语言",
        ["中文", "English"],
        key=f"cover_letter_language_{bundle['fingerprint']}",
    )
    language = "zh" if language_label == "中文" else "en"
    drafts = bundle.setdefault("cover_letters", {})
    if st.button(
        "生成求职信",
        type="primary",
        key=f"generate_cover_letter_{bundle['fingerprint']}_{language}",
    ):
        try:
            provider = _create_counted_provider()
            with st.spinner("正在基于简历证据生成求职信……"):
                draft, evidence = generate_cover_letter(
                    resume_profile,
                    job_profile,
                    analysis,
                    language,
                    provider=provider,
                )
            drafts[language] = {
                "draft": draft.model_dump(mode="json"),
                "evidence": [item.__dict__ for item in evidence],
            }
            bundle.pop("application_package", None)
            _save_active_job_analysis(bundle)
            st.session_state["workspace_notice"] = "求职信已生成并通过基础证据校验。"
            st.rerun()
        except (AiParserError, AiProviderError) as exc:
            st.error(str(exc))

    record = drafts.get(language)
    if not record:
        st.info("当前语言还没有求职信草稿。")
        return
    draft = CoverLetterDraft.model_validate(record["draft"])
    st.markdown(f"**{draft.salutation}**")
    for paragraph in draft.paragraphs:
        st.write(paragraph.text)
        st.caption("证据：" + "、".join(paragraph.evidence_ids))
    st.write(draft.closing)
    for note in draft.caution_notes:
        st.warning(note)

    if "docx" not in record or "pdf" not in record:
        record["docx"] = build_cover_letter_docx(job_profile, draft)
        record["pdf"] = build_cover_letter_pdf(job_profile, draft)
        _save_active_job_analysis(bundle)
    columns = st.columns(2)
    columns[0].download_button(
        "下载求职信 Word",
        data=record["docx"],
        file_name=f"cover-letter-{language}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )
    columns[1].download_button(
        "下载求职信 PDF",
        data=record["pdf"],
        file_name=f"cover-letter-{language}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def _render_preparation(
    preparation: InterviewPreparation,
    evidence_records: list[dict],
) -> None:
    if preparation.personalized_answer:
        st.markdown("**个性化回答草稿（可复制）**")
        st.code(preparation.personalized_answer, language=None)
    else:
        st.warning("当前个人资料证据不足，系统没有生成第一人称回答草稿。")

    if preparation.key_points:
        st.markdown("**回答要点**")
        for item in preparation.key_points:
            st.write(f"- {item}")

    star_values = preparation.star_outline.model_dump()
    if any(star_values.values()):
        st.markdown("**STAR 组织方式**")
        for label, key in [("情境", "situation"), ("任务", "task"), ("行动", "action"), ("结果", "result")]:
            if star_values[key]:
                st.write(f"- {label}：{star_values[key]}")

    evidence_by_id = {item["id"]: item for item in evidence_records}
    if preparation.evidence_ids:
        st.markdown("**引用的个人证据**")
        for identifier in preparation.evidence_ids:
            record = evidence_by_id.get(identifier)
            if record:
                st.write(f"> [{identifier} · {record['source']}] {record['text']}")
    for item in preparation.missing_information:
        st.warning(item)
    for item in preparation.caution_notes:
        st.info(item)


def _render_feedback(feedback: InterviewFeedback) -> None:
    columns = st.columns(4)
    columns[0].metric("完整性", f"{feedback.completeness_score}/5")
    columns[1].metric("STAR", f"{feedback.star_score}/5")
    columns[2].metric("岗位相关性", f"{feedback.relevance_score}/5")
    columns[3].metric("表达清晰度", f"{feedback.clarity_score}/5")
    if feedback.strengths:
        st.markdown("**做得好的地方**")
        for item in feedback.strengths:
            st.write(f"- {item}")
    if feedback.improvements:
        st.markdown("**改进建议**")
        for item in feedback.improvements:
            st.write(f"- {item}")
    if feedback.unsupported_claims:
        st.markdown("**需要本人确认的陈述**")
        for item in feedback.unsupported_claims:
            st.write(f"- {item}")
    if feedback.improved_structure:
        st.markdown("**更好的回答结构**")
        for item in feedback.improved_structure:
            st.write(f"- {item}")
    if feedback.follow_up_question:
        st.info(f"建议继续练习：{feedback.follow_up_question}")


def _render_interview_center(
    bundle: dict,
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
    analysis: MatchAnalysis,
) -> None:
    st.caption(
        "回答草稿要求基于简历证据，并校验证据编号和新增数字；"
        "证据不足时不生成个性化回答。练习反馈仅保存在当前会话。"
    )
    if not analysis.interview_questions:
        st.info("本次没有生成可用的面试问题。")
        return

    category_labels = {
        "job_knowledge": "岗位知识",
        "behavioral": "行为面试",
        "project_deep_dive": "项目深挖",
        "capability_gap": "能力缺口",
    }
    preparations = bundle.setdefault("interview_preparations", {})
    feedback_records = bundle.setdefault("interview_feedback", {})

    for index, question in enumerate(analysis.interview_questions, start=1):
        key = str(index - 1)
        category = category_labels[question.category.value]
        with st.expander(f"{category} {index} · {question.question}", expanded=index == 1):
            st.write(question.why_asked)
            st.markdown("**基础答题思路**")
            for item in question.answer_outline:
                st.write(f"- {item}")

            stored_preparation = preparations.get(key)
            if not stored_preparation:
                if st.button(
                    "生成证据化回答思路",
                    key=f"prepare_{bundle['fingerprint']}_{key}",
                ):
                    try:
                        provider = _create_counted_provider()
                        with st.spinner("正在根据简历证据准备回答……"):
                            preparation, evidence = prepare_interview_answer(
                                question,
                                resume_profile,
                                analysis,
                                provider=provider,
                            )
                        preparations[key] = {
                            "preparation": preparation.model_dump(mode="json"),
                            "evidence": [item.__dict__ for item in evidence],
                        }
                        _save_active_job_analysis(bundle)
                        st.rerun()
                    except (AiParserError, AiProviderError) as exc:
                        st.error(str(exc))
            else:
                preparation = InterviewPreparation.model_validate(
                    stored_preparation["preparation"]
                )
                _render_preparation(preparation, stored_preparation["evidence"])

            with st.form(f"mock_interview_{bundle['fingerprint']}_{key}"):
                answer = st.text_area(
                    "输入你的练习回答",
                    key=f"mock_answer_{bundle['fingerprint']}_{key}",
                    height=160,
                    placeholder="建议至少 20 个字符，可按 STAR 结构回答……",
                )
                review_requested = st.form_submit_button("提交回答并获取点评")
            if review_requested:
                evidence = (
                    [InterviewEvidence(**item) for item in stored_preparation["evidence"]]
                    if stored_preparation
                    else collect_interview_evidence(question, resume_profile, analysis)
                )
                try:
                    provider = _create_counted_provider()
                    with st.spinner("正在点评你的练习回答……"):
                        feedback = review_interview_answer(
                            question,
                            answer,
                            job_profile,
                            evidence,
                            provider=provider,
                        )
                    feedback_records[key] = {
                        "question": question.question,
                        "feedback": feedback.model_dump(mode="json"),
                    }
                    _save_active_job_analysis(bundle)
                    st.rerun()
                except (AiParserError, AiProviderError) as exc:
                    st.error(str(exc))

            if key in feedback_records:
                st.markdown("### 本题复盘")
                _render_feedback(
                    InterviewFeedback.model_validate(feedback_records[key]["feedback"])
                )


def _render_report_download(
    bundle: dict,
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
    analysis: MatchAnalysis,
    score: ScoreResult,
) -> None:
    report_source = {
        "fingerprint": bundle["fingerprint"],
        "resume_profile": resume_profile.model_dump(mode="json"),
        "job_profile": bundle["job_profile"],
        "match_analysis": analysis.model_dump(mode="json"),
        "interview_feedback": bundle.get("interview_feedback", {}),
    }
    report_version = hashlib.sha256(
        json.dumps(
            report_source,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    report_files = bundle.get("report_files", {})
    if report_files.get("version") != report_version:
        report_files = {
            "version": report_version,
            "docx": build_docx_report(
                resume_profile,
                job_profile,
                analysis,
                score,
                bundle.get("interview_feedback", {}),
            ),
            "pdf": build_pdf_report(
                resume_profile,
                job_profile,
                analysis,
                score,
                bundle.get("interview_feedback", {}),
            ),
        }
        bundle["report_files"] = report_files
        _save_active_job_analysis(bundle)

    st.caption("Word 和 PDF 均在内存中生成，不包含原始简历全文；下载由你的浏览器完成。")
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "下载 Word 报告",
        data=report_files["docx"],
        file_name="ai-job-analysis-report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )
    download_columns[1].download_button(
        "下载 PDF 报告",
        data=report_files["pdf"],
        file_name="ai-job-analysis-report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
    st.markdown("**简历改写建议汇总（点击代码框右上角复制）**")
    suggestions = "\n\n".join(
        f"{index}. {item.suggested_text}"
        for index, item in enumerate(analysis.resume_suggestions, start=1)
    )
    st.code(suggestions or "本次没有通过证据校验的简历改写建议。", language=None)


def _render_application_package(
    bundle: dict,
    candidate_profile: dict,
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
    analysis: MatchAnalysis,
    score: ScoreResult,
) -> None:
    cover_letter_drafts = {
        language: record.get("draft")
        for language, record in bundle.get("cover_letters", {}).items()
        if isinstance(record, dict) and record.get("draft")
    }
    package_source = {
        "fingerprint": bundle["fingerprint"],
        "analysis": analysis.model_dump(mode="json"),
        "resume_edits": bundle.get("resume_edits", {}),
        "resume_versions": bundle.get("resume_versions", []),
        "application_tracking": bundle.get("application_tracking", {}),
        "cover_letters": cover_letter_drafts,
        "interview_feedback": bundle.get("interview_feedback", {}),
        "candidate_facts": candidate_profile.get("facts", []),
    }
    package_version = hashlib.sha256(
        json.dumps(
            package_source,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    package = bundle.get("application_package", {})
    if package.get("version") != package_version:
        facts = [
            CandidateFact.model_validate(item)
            for item in candidate_profile.get("facts", [])
        ]
        with st.spinner("正在本地整理投递材料包……"):
            result = build_application_package(
                resume_profile,
                job_profile,
                analysis,
                score,
                bundle,
                facts,
            )
        package = {
            "version": package_version,
            "data": result.data,
            "files": list(result.files),
            "warnings": list(result.warnings),
        }
        bundle["application_package"] = package
        _save_active_job_analysis(bundle)

    st.markdown("### 当前投递材料包")
    st.caption(
        "材料只从已完成的分析、已保存的简历版本和已生成的求职信中整理；"
        "不会加入原始 PDF，也不会调用模型。"
    )
    for filename in package["files"]:
        st.write(f"- {filename}")
    for warning in package["warnings"]:
        st.warning(warning)
    st.download_button(
        "下载当前投递材料包 ZIP",
        data=package["data"],
        file_name=f"application-package-{bundle['job_id'][:12]}.zip",
        mime="application/zip",
        use_container_width=True,
    )
    st.caption("ZIP 在内存中生成并由浏览器下载；请在正式投递前逐份核对。")


def _date_input_value(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _render_application_tracking(bundle: dict, job_profile: JobProfile) -> None:
    record = get_application_record(bundle)
    versions = [
        ResumeVersion.model_validate(item)
        for item in bundle.get("resume_versions", [])
    ]
    version_labels = {item.id: item.label for item in versions}
    st.caption(
        "投递状态、日期和材料绑定仅在本地会话更新，不调用模型。"
        "如需下次继续，请返回岗位对比并导出脱敏档案。"
    )
    with st.form(f"application_tracking_form_{bundle['job_id']}"):
        status = st.selectbox(
            "投递状态",
            options=list(ApplicationStatus),
            index=list(ApplicationStatus).index(record.status),
            format_func=lambda value: APPLICATION_STATUS_LABELS[value],
            key=f"application_status_{bundle['job_id']}",
        )
        date_columns = st.columns(2)
        applied_on = date_columns[0].date_input(
            "投递日期",
            value=_date_input_value(record.applied_on),
            format="YYYY-MM-DD",
            key=f"application_applied_on_{bundle['job_id']}",
        )
        deadline = date_columns[1].date_input(
            "申请截止日期",
            value=_date_input_value(record.deadline),
            format="YYYY-MM-DD",
            key=f"application_deadline_{bundle['job_id']}",
        )
        action_columns = st.columns(2)
        interview_on = action_columns[0].date_input(
            "面试日期",
            value=_date_input_value(record.interview_on),
            format="YYYY-MM-DD",
            key=f"application_interview_on_{bundle['job_id']}",
        )
        follow_up_on = action_columns[1].date_input(
            "跟进日期",
            value=_date_input_value(record.follow_up_on),
            format="YYYY-MM-DD",
            key=f"application_follow_up_on_{bundle['job_id']}",
        )
        resume_version_id = st.selectbox(
            "本次投递使用的简历版本",
            options=[None, *version_labels],
            index=(
                [None, *version_labels].index(record.resume_version_id)
                if record.resume_version_id in version_labels
                else 0
            ),
            format_func=lambda value: "暂未绑定" if value is None else version_labels[value],
            key=f"application_resume_version_{bundle['job_id']}",
        )
        job_url = st.text_input(
            "岗位链接（选填）",
            value=record.job_url or "",
            placeholder="https://...",
            key=f"application_job_url_{bundle['job_id']}",
        )
        notes = st.text_area(
            "投递备注（选填）",
            value=record.notes,
            max_chars=2_000,
            height=130,
            placeholder="例如：内推渠道、需要准备的材料、面试反馈……",
            key=f"application_notes_{bundle['job_id']}",
        )
        saved = st.form_submit_button("保存投递记录", type="primary")
    if saved:
        try:
            updated = build_application_record(
                {
                    "status": status.value,
                    "applied_on": applied_on.isoformat() if applied_on else None,
                    "deadline": deadline.isoformat() if deadline else None,
                    "interview_on": interview_on.isoformat() if interview_on else None,
                    "follow_up_on": follow_up_on.isoformat() if follow_up_on else None,
                    "job_url": job_url,
                    "notes": notes,
                    "resume_version_id": resume_version_id,
                },
                available_resume_version_ids=set(version_labels),
            )
            bundle["application_tracking"] = updated.model_dump(mode="json")
            bundle.pop("application_package", None)
            _save_active_job_analysis(bundle)
            st.session_state["workspace_notice"] = (
                f"已保存 {job_profile.company} · {job_profile.title} 的投递记录。"
            )
            st.rerun()
        except ApplicationTrackerError as exc:
            st.error(str(exc))


def _render_job_input_fields(prefix: str, index: int) -> dict[str, str]:
    number = index + 1
    st.markdown(f"#### 岗位 {number}")
    url_columns = st.columns([4, 1])
    job_url = url_columns[0].text_input(
        f"岗位链接（岗位 {number}，选填）",
        placeholder="https://company.example/jobs/123",
        key=f"{prefix}_url_{index}",
        help="可尝试读取公开招聘页；登录页、动态页面或反爬页面可能需要手动粘贴。",
    )
    if url_columns[1].button(
        "读取链接",
        key=f"{prefix}_fetch_url_{index}",
        use_container_width=True,
    ):
        try:
            with st.spinner(f"正在读取岗位 {number}……"):
                imported = fetch_job_posting(job_url)
            imported_values = {
                f"{prefix}_company_{index}": imported.company,
                f"{prefix}_title_{index}": imported.title,
                f"{prefix}_location_{index}": imported.location,
                f"{prefix}_jd_{index}": imported.description,
            }
            for key, value in imported_values.items():
                if value:
                    st.session_state[key] = value
            if imported.job_type in {"全职", "实习", "兼职", "合同", "其他"}:
                st.session_state[f"{prefix}_type_{index}"] = imported.job_type
            st.success(f"岗位 {number} 已从链接读取，请核对下方内容。")
        except JobLinkError as exc:
            st.error(str(exc))
    columns = st.columns(2)
    company = columns[0].text_input(
        f"公司名称（岗位 {number}）",
        key=f"{prefix}_company_{index}",
    )
    title = columns[1].text_input(
        f"岗位名称（岗位 {number}）",
        key=f"{prefix}_title_{index}",
    )
    detail_columns = st.columns(2)
    location = detail_columns[0].text_input(
        f"工作地点（岗位 {number}，选填）",
        key=f"{prefix}_location_{index}",
    )
    job_type = detail_columns[1].selectbox(
        f"岗位类型（岗位 {number}，选填）",
        options=["", "全职", "实习", "兼职", "合同", "其他"],
        format_func=lambda value: "请选择" if value == "" else value,
        key=f"{prefix}_type_{index}",
    )
    jd_text = st.text_area(
        f"岗位 JD（岗位 {number}）",
        height=210,
        placeholder="请粘贴完整岗位描述，至少 50 个非空白字符……",
        key=f"{prefix}_jd_{index}",
    )
    return {
        "company": company,
        "job_title": title,
        "location": location,
        "job_type": job_type,
        "job_url": job_url,
        "jd_text": jd_text,
    }


def _validate_job_records(records: list[dict[str, str]]) -> tuple[list[JobInput], list[str]]:
    jobs: list[JobInput] = []
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        try:
            jobs.append(JobInput(**record))
        except ValidationError as exc:
            errors.extend(
                f"岗位 {index} · {message}" for message in _format_job_errors(exc)
            )
    return jobs, errors


def _bundle_analysis(bundle: dict) -> MatchAnalysis:
    if bundle.get("stage") == "final" and bundle.get("final_analysis"):
        return MatchAnalysis.model_validate(bundle["final_analysis"])
    preliminary = PreliminaryAnalysis.model_validate(bundle["preliminary_analysis"])
    return MatchAnalysis(
        matches=preliminary.matches,
        resume_suggestions=preliminary.resume_suggestions,
        interview_questions=preliminary.interview_questions,
    )


def _jobs_need_model(
    resume_id: str,
    jobs: list[JobInput],
    *,
    resume_profile_available: bool = False,
) -> bool:
    cache = _analysis_cache()
    resume_missing = (
        not resume_profile_available
        and _model_cache_key("resume", resume_id) not in cache["resumes"]
    )
    for job in jobs:
        job_id = _job_fingerprint(job)
        if (
            _model_cache_key("job", job_id) not in cache["jobs"]
            or _model_cache_key("initial", resume_id, job_id) not in cache["initial"]
        ):
            return True
    return resume_missing


def _add_jobs_for_candidate(
    candidate_profile: dict,
    jobs: list[JobInput],
) -> tuple[int, int]:
    resume_id = candidate_profile["resume_id"]
    resume_profile = ResumeProfile.model_validate(candidate_profile["resume_profile"])
    provider = _LazySessionProvider()
    existing = st.session_state.setdefault("job_analyses", {})
    additions: dict[str, dict] = {}
    duplicates = 0
    for job in jobs:
        job_id = _job_fingerprint(job)
        if job_id in existing or job_id in additions:
            duplicates += 1
            continue
        job_id, job_profile, preliminary, _ = _cached_job_analysis(
            resume_id,
            resume_profile,
            candidate_profile["resume_text"],
            job,
            provider,
        )
        additions[job_id] = _new_job_bundle(
            job_id,
            job_profile,
            preliminary,
            job.job_url,
        )
    existing.update(additions)
    st.session_state["job_analyses"] = existing
    return len(additions), duplicates


def _render_job_workspace(candidate_profile: dict, job_analyses: dict[str, dict]) -> None:
    st.subheader("岗位对比工作台")
    notice = st.session_state.pop("workspace_notice", None)
    if notice:
        st.success(notice)
    st.caption(
        f"当前简历：{candidate_profile.get('filename', '已解析简历')} ｜ "
        f"已分析 {len(job_analyses)}/{MAX_JOBS_PER_SESSION} 个岗位。"
    )
    tracking_metrics = build_application_metrics(job_analyses)
    metric_columns = st.columns(5)
    metric_columns[0].metric("岗位总数", tracking_metrics.total_jobs)
    metric_columns[1].metric("已投递", tracking_metrics.submitted)
    metric_columns[2].metric("回复率", f"{tracking_metrics.response_rate:.1f}%")
    metric_columns[3].metric("面试率", f"{tracking_metrics.interview_rate:.1f}%")
    metric_columns[4].metric("Offer", tracking_metrics.offers)
    actions = upcoming_application_actions(job_analyses)
    if actions:
        with st.expander(f"截止日期与跟进提醒 · {len(actions)} 项", expanded=True):
            st.caption("显示已逾期或未来 30 天内的申请截止、面试和跟进日期。")
            for action in actions:
                icon = "🔴" if action["timing"] == "已逾期" else "🗓️"
                st.write(
                    f"{icon} {action['date']} · {action['kind']} · "
                    f"{action['company']} · {action['title']}（{action['timing']}）"
                )
    comparison = build_job_comparison(candidate_profile, job_analyses)
    display_rows = [
        {
            "公司": item.company,
            "岗位": item.title,
            "阶段": "最终分析" if item.stage == "final" else "待补充确认",
            "投递状态": APPLICATION_STATUS_LABELS[
                ApplicationStatus(item.application_status)
            ],
            "匹配度": "--" if item.match_score is None else f"{item.match_score:.1f}%",
            "完整度": f"{item.information_completeness:.1f}%",
            "ATS": f"{item.ats_score}/100",
            "硬性风险": item.hard_risks,
            "必须项缺口": item.must_have_gaps,
            "推荐值": item.recommendation_score,
        }
        for item in comparison
    ]
    st.warning("推荐值由本地规则计算，用于整理投递顺序，不代表录取概率。")
    st.dataframe(
        display_rows,
        hide_index=True,
        use_container_width=True,
    )

    comparison_source = [item.model_dump(mode="json") for item in comparison]
    comparison_version = hashlib.sha256(
        json.dumps(comparison_source, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    comparison_files = st.session_state.get("comparison_report_files", {})
    if comparison_files.get("version") != comparison_version:
        comparison_files = {
            "version": comparison_version,
            "docx": build_job_comparison_docx(comparison),
            "pdf": build_job_comparison_pdf(comparison),
        }
        st.session_state["comparison_report_files"] = comparison_files
    with st.expander("下载对比报告与备份档案"):
        st.caption(
            "对比报告不包含原始简历全文。求职档案是脱敏 JSON，保存分析记录、"
            "补充事实和简历版本信息；文件由浏览器下载，本应用不会自动长期保存。"
        )
        download_columns = st.columns(3)
        download_columns[0].download_button(
            "下载对比 Word",
            data=comparison_files["docx"],
            file_name="job-comparison-report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
        download_columns[1].download_button(
            "下载对比 PDF",
            data=comparison_files["pdf"],
            file_name="job-comparison-report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        try:
            archive_bytes = build_workspace_archive(candidate_profile, job_analyses)
            download_columns[2].download_button(
                "导出脱敏档案",
                data=archive_bytes,
                file_name="ai-job-assistant-archive.json",
                mime="application/json",
                use_container_width=True,
            )
        except WorkspaceArchiveError as exc:
            st.error(str(exc))

    st.markdown("### 选择岗位进入详细流程")
    for index, item in enumerate(comparison, start=1):
        columns = st.columns([4, 1])
        columns[0].write(
            f"**{index}. {item.company} · {item.title}**  "
            f"推荐值 {item.recommendation_score:.1f} · "
            f"{APPLICATION_STATUS_LABELS[ApplicationStatus(item.application_status)]}"
        )
        if columns[1].button(
            "打开岗位",
            key=f"open_job_{item.job_id}",
            use_container_width=True,
        ):
            st.session_state["active_job_id"] = item.job_id
            st.rerun()

    remaining = MAX_JOBS_PER_SESSION - len(job_analyses)
    if remaining > 0:
        with st.expander("继续添加岗位 JD"):
            batch_prefix = f"additional_{len(job_analyses)}"
            add_count = int(
                st.number_input(
                    "本次添加岗位数量",
                    min_value=1,
                    max_value=remaining,
                    value=1,
                    step=1,
                    key=f"{batch_prefix}_count",
                )
            )
            records = [
                _render_job_input_fields(batch_prefix, index)
                for index in range(add_count)
            ]
            st.caption(
                f"若均为新岗位，本次最多增加 {add_count * 2} 次模型调用；"
                "简历解析不会重复调用。"
            )
            if st.button("分析并加入对比", type="primary", use_container_width=True):
                jobs, errors = _validate_job_records(records)
                if errors:
                    st.error("请修正以下问题：")
                    for error in errors:
                        st.write(f"- {error}")
                elif _jobs_need_model(
                    candidate_profile["resume_id"],
                    jobs,
                    resume_profile_available=True,
                ) and not has_api_key():
                    st.error("尚未配置 OPENAI_API_KEY，无法分析新岗位。")
                else:
                    try:
                        with st.spinner("正在分析新增岗位；简历解析结果将直接复用……"):
                            added, duplicates = _add_jobs_for_candidate(candidate_profile, jobs)
                        st.session_state["workspace_notice"] = (
                            f"已新增 {added} 个岗位。"
                            + (f"另有 {duplicates} 个重复岗位已跳过。" if duplicates else "")
                        )
                        st.rerun()
                    except (AiParserError, AiProviderError) as exc:
                        st.error(str(exc))
    else:
        st.info("当前会话已达到 5 个岗位的比较上限。")

    if st.button("上传新简历并清空当前工作台"):
        st.session_state.clear()
        st.rerun()


def _render_final_results(bundle: dict, candidate_profile: dict) -> None:
    resume_profile = ResumeProfile.model_validate(candidate_profile["resume_profile"])
    job_profile = JobProfile.model_validate(bundle["job_profile"])
    match_analysis = MatchAnalysis.model_validate(bundle["final_analysis"])
    score = calculate_scores(job_profile, match_analysis)

    _render_step_progress(4)
    st.subheader("第 4 步：最终建议与材料")
    workspace_notice = st.session_state.pop("workspace_notice", None)
    if workspace_notice:
        st.success(workspace_notice)
    action_columns = st.columns(3)
    if action_columns[0].button("返回岗位对比", use_container_width=True):
        st.session_state["active_job_id"] = None
        st.rerun()
    if action_columns[1].button("修改补充信息", use_container_width=True):
        # Keep the current final result available until the user actually submits
        # changed answers. This makes entering the edit screen fully reversible.
        bundle["stage"] = "clarification"
        bundle["editing_clarifications"] = True
        _save_active_job_analysis(bundle)
        st.rerun()
    if action_columns[2].button("重新开始", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    job_facts = [
        CandidateFact.model_validate(item)
        for item in candidate_profile.get("facts", [])
        if item.get("source_job_id") == bundle["job_id"]
    ]
    if job_facts:
        with st.expander(f"本岗位已确认补充信息 · {len(job_facts)} 项"):
            st.caption("以下内容来自你的明确填写，不会标记为 PDF 简历原文。")
            for fact in job_facts:
                st.write(f"- {fact.statement}")
                if fact.metrics:
                    st.caption(f"成果或数据：{fact.metrics}")

    sections = [
        "岗位匹配",
        "关键词缺口",
        "ATS 体检",
        "简历优化",
        "投递管理",
        "材料包",
        "求职信",
        "面试准备",
        "报告下载",
        "简历结构",
        "JD 结构",
    ]
    selected_section = st.segmented_control(
        "选择功能",
        sections,
        default="岗位匹配",
        key=f"final_section_{bundle['job_id']}",
        selection_mode="single",
    ) or "岗位匹配"
    st.divider()
    if selected_section == "岗位匹配":
        _render_match_analysis(match_analysis, score, job_profile)
    elif selected_section == "关键词缺口":
        _render_keyword_gaps(job_profile, match_analysis)
    elif selected_section == "ATS 体检":
        _render_ats_report(candidate_profile, job_profile, match_analysis)
    elif selected_section == "简历优化":
        _render_resume_actions(
            bundle,
            match_analysis,
            resume_profile,
            job_profile,
            bundle["fingerprint"],
            candidate_profile,
        )
    elif selected_section == "投递管理":
        _render_application_tracking(bundle, job_profile)
    elif selected_section == "材料包":
        _render_application_package(
            bundle,
            candidate_profile,
            resume_profile,
            job_profile,
            match_analysis,
            score,
        )
    elif selected_section == "求职信":
        _render_cover_letter(bundle, resume_profile, job_profile, match_analysis)
    elif selected_section == "面试准备":
        _render_interview_center(bundle, resume_profile, job_profile, match_analysis)
    elif selected_section == "报告下载":
        _render_report_download(bundle, resume_profile, job_profile, match_analysis, score)
    elif selected_section == "简历结构":
        _render_resume_profile(resume_profile)
    elif selected_section == "JD 结构":
        _render_job_profile(job_profile)
    st.caption("本阶段不预测录取概率，不补写不存在的经历，也不使用 RAG 或长期保存数据。")


def _complete_clarification(
    bundle: dict,
    candidate_profile: dict,
    answers: list[ClarificationAnswer],
) -> None:
    job_profile = JobProfile.model_validate(bundle["job_profile"])
    preliminary = PreliminaryAnalysis.model_validate(bundle["preliminary_analysis"])
    answer_payload = [item.model_dump(mode="json") for item in answers]
    if bundle.get("final_analysis") and bundle.get("clarification_answers", []) == answer_payload:
        bundle["stage"] = "final"
        bundle.pop("editing_clarifications", None)
        _save_active_job_analysis(bundle)
        st.session_state["workspace_notice"] = (
            "补充选项没有变化，已保留并返回上一次生成的结果。"
        )
        st.rerun()

    existing_facts = [
        CandidateFact.model_validate(item) for item in candidate_profile.get("facts", [])
    ]
    choice_facts = facts_from_answers(answers, job_profile, bundle["job_id"])
    saved_details = [
        SupplementDetail.model_validate(item)
        for item in bundle.get("supplement_details", [])
    ]
    detailed_facts = facts_from_supplement_details(
        saved_details,
        job_profile,
        bundle["job_id"],
    )
    replacement_by_requirement = {
        item.source_requirement_text: item for item in choice_facts
    }
    have_requirement_texts = {
        requirement.original_text
        for requirement in job_profile.requirements
        if any(
            answer.requirement_id == requirement.id and answer.status == "have"
            for answer in answers
        )
    }
    replacement_by_requirement.update(
        {
            item.source_requirement_text: item
            for item in detailed_facts
            if item.source_requirement_text in have_requirement_texts
        }
    )
    replacement_facts = list(replacement_by_requirement.values())
    candidate_facts = merge_candidate_facts(
        existing_facts,
        replacement_facts,
        bundle["job_id"],
    )

    initial_analysis = MatchAnalysis(
        matches=preliminary.matches,
        resume_suggestions=preliminary.resume_suggestions,
        interview_questions=preliminary.interview_questions,
    )
    final_analysis = apply_answers_to_final_analysis(
        initial_analysis,
        answers,
        candidate_facts,
        job_profile,
        preliminary,
    )

    facts_changed = [item.model_dump(mode="json") for item in existing_facts] != [
        item.model_dump(mode="json") for item in candidate_facts
    ]
    candidate_profile["facts"] = [item.model_dump(mode="json") for item in candidate_facts]
    st.session_state["candidate_profile"] = candidate_profile
    if facts_changed:
        for job_id, other_bundle in st.session_state.setdefault("job_analyses", {}).items():
            if job_id != bundle["job_id"] and other_bundle.get("stage") == "final":
                invalidate_generated_materials(other_bundle)

    # The old final result and its derived downloads remain usable while editing.
    # Invalidate them only after a replacement analysis has succeeded.
    invalidate_generated_materials(bundle)
    bundle.pop("supplement_suggestions", None)
    bundle.pop("editing_clarifications", None)
    bundle["clarification_answers"] = answer_payload
    bundle["final_analysis"] = final_analysis.model_dump(mode="json")
    bundle["stage"] = "final"
    bundle["resume_edits"] = {}
    bundle["cover_letters"] = {}
    bundle["interview_preparations"] = {}
    bundle["interview_feedback"] = {}
    _save_active_job_analysis(bundle)
    st.session_state["workspace_notice"] = (
        f"最终分析已在本地更新，其中包含 {len(replacement_facts)} 项用户确认信息，"
        "本步骤没有调用模型。"
    )
    st.rerun()


def _render_clarification_stage(bundle: dict, candidate_profile: dict) -> None:
    job_profile = JobProfile.model_validate(bundle["job_profile"])
    preliminary = PreliminaryAnalysis.model_validate(bundle["preliminary_analysis"])
    questions = [
        ClarificationQuestion.model_validate(item)
        for item in bundle.get("clarification_questions", [])
    ]
    saved_answers = {
        item["question_id"]: ClarificationAnswer.model_validate(item)
        for item in bundle.get("clarification_answers", [])
    }

    _render_step_progress(3)
    st.subheader("第 3 步：补充真实信息")
    st.caption(
        "初步匹配已完成。这些问题只针对部分匹配、缺失或待确认项目，"
        "最多 5 题。这里仅需选择；具体经历可稍后在“简历优化”中按需补充。"
    )
    with st.expander("查看初步匹配结果"):
        initial_analysis = MatchAnalysis(matches=preliminary.matches)
        _render_match_analysis(
            initial_analysis,
            calculate_scores(job_profile, initial_analysis),
            job_profile,
        )

    if not questions:
        st.success("所有岗位要求都已有可验证证据，无需额外补充。")

    requirement_by_id = {item.id: item for item in job_profile.requirements}
    option_labels = ["尚未回答", "具备", "不具备", "不确定"]
    option_values = {
        "尚未回答": "unanswered",
        "具备": "have",
        "不具备": "not_have",
        "不确定": "unsure",
    }
    reverse_options = {value: label for label, value in option_values.items()}
    submitted_answers: list[ClarificationAnswer] = []
    with st.form(f"clarification_{bundle['job_id']}"):
        for index, question in enumerate(questions, start=1):
            requirement = requirement_by_id[question.requirement_id]
            saved = saved_answers.get(question.id)
            st.markdown(f"### {index}. {requirement.normalized_name}")
            st.write(question.prompt)
            status_label = st.radio(
                f"你目前是否具备该条件？（问题 {index}）",
                option_labels,
                index=option_labels.index(
                    reverse_options.get(saved.status if saved else "unanswered", "尚未回答")
                ),
                horizontal=True,
                key=f"clarification_status_{bundle['job_id']}_{question.id}",
            )
            submitted_answers.append(
                ClarificationAnswer(
                    question_id=question.id,
                    requirement_id=question.requirement_id,
                    status=option_values[status_label],
                )
            )
        submit = st.form_submit_button("完成选择并查看结果", type="primary")
        if bundle.get("editing_clarifications"):
            cancel_edit = st.form_submit_button("取消修改，返回原结果")
            skip = False
        else:
            skip = st.form_submit_button("暂不选择，直接查看结果")
            cancel_edit = False

    if cancel_edit:
        bundle["stage"] = "final"
        bundle.pop("editing_clarifications", None)
        _save_active_job_analysis(bundle)
        st.session_state["workspace_notice"] = "未修改补充信息，已返回上一次生成的结果。"
        st.rerun()

    if submit or skip:
        try:
            answers = (
                []
                if skip
                else validate_clarification_answers(
                    submitted_answers,
                    questions,
                    job_profile,
                )
            )
            _complete_clarification(bundle, candidate_profile, answers)
        except (EvidenceFlowError, AiParserError, AiProviderError) as exc:
            st.error(str(exc))

    if st.button("返回岗位对比"):
        st.session_state["active_job_id"] = None
        st.rerun()


def _render_archive_import() -> None:
    with st.expander("导入已有求职档案"):
        st.caption(
            "仅支持本应用导出的脱敏 JSON。导入内容只进入当前会话，"
            "不会写入数据库或浏览器长期存储。"
        )
        archive_file = st.file_uploader(
            "选择求职档案 JSON（最大 5 MB）",
            type=["json"],
            accept_multiple_files=False,
            key="workspace_archive_upload",
        )
        if st.button(
            "导入并恢复工作台",
            key="load_workspace_archive",
            use_container_width=True,
        ):
            try:
                candidate, jobs = load_workspace_archive(
                    archive_file.getvalue() if archive_file is not None else b""
                )
                st.session_state.clear()
                st.session_state["candidate_profile"] = candidate
                st.session_state["job_analyses"] = jobs
                st.session_state["active_job_id"] = None
                st.session_state["model_call_count"] = 0
                st.session_state["workspace_notice"] = (
                    f"已从脱敏档案恢复 {len(jobs)} 个岗位；未调用模型。"
                )
                st.rerun()
            except WorkspaceArchiveError as exc:
                st.error(str(exc))


def main() -> None:
    load_dotenv(override=True)
    st.set_page_config(page_title="AI 求职助手", page_icon="📄", layout="centered")

    st.title("AI 求职助手")
    st.caption("交互式补充真实证据，再生成岗位匹配、简历、求职信与面试材料")
    st.info(
        "隐私说明：原始 PDF、提取文字和粘贴内容仅用于当前会话，不会被本应用长期保存。"
        "预览中的电话、邮箱和详细地址会被自动脱敏。开始分析后，脱敏简历文本和岗位信息"
        "将发送给 OpenAI 进行结构化解析，并在 API 请求中设置 store=False。"
    )
    st.caption(f"当前会话模型调用次数：{st.session_state.get('model_call_count', 0)}")

    active_job_id = st.session_state.get("active_job_id")
    job_analyses = st.session_state.setdefault("job_analyses", {})
    candidate_profile = st.session_state.get("candidate_profile")
    if active_job_id and active_job_id in job_analyses and candidate_profile:
        active_job = job_analyses[active_job_id]
        if active_job.get("stage") == "final" and active_job.get("final_analysis"):
            _render_final_results(active_job, candidate_profile)
        else:
            _render_clarification_stage(active_job, candidate_profile)
        return
    if candidate_profile and job_analyses:
        _render_job_workspace(candidate_profile, job_analyses)
        return

    _render_archive_import()
    _render_step_progress(1)
    st.subheader("第 1 步：上传简历并填写岗位")

    uploaded_file = st.file_uploader(
        "上传一份 PDF 简历（必填，最大 10 MB）",
        type=["pdf"],
        accept_multiple_files=False,
        key="resume_upload",
    )

    pdf_result, pdf_error = _parse_uploaded_pdf(uploaded_file)
    pdf_text_is_valid = bool(pdf_result and has_valid_resume_text(pdf_result.text))

    if uploaded_file is None:
        st.session_state["resume_fallback"] = ""
    elif pdf_text_is_valid:
        st.session_state["resume_fallback"] = ""
        st.success(f"PDF 解析成功：共 {pdf_result.page_count} 页，将优先使用 PDF 文本。")
    else:
        if pdf_result is not None:
            pdf_error = "PDF 中提取到的有效文字少于 50 个字符。"
        st.warning(
            f"{pdf_error or 'PDF 无法提取有效文字。'} 请粘贴简历内容，或重新上传文本型 PDF。"
        )
        st.text_area(
            "粘贴简历内容（备用入口）",
            key="resume_fallback",
            height=220,
            placeholder="请粘贴至少 50 个非空白字符……",
            help="仅当 PDF 无法提取有效文字时使用。内容仅用于当前会话，不会被长期保存。",
        )
        st.caption("粘贴内容不会被长期保存；若重新上传有效 PDF，将自动改用 PDF 文本。")

    st.markdown("### 岗位信息")
    job_count = int(
        st.number_input(
            "本次提交岗位数量",
            min_value=1,
            max_value=MAX_JOBS_PER_SESSION,
            value=1,
            step=1,
            key="initial_job_count",
        )
    )
    job_records = [
        _render_job_input_fields("initial", index) for index in range(job_count)
    ]
    st.caption(
        f"全部为新内容时预计调用 {1 + job_count * 2} 次模型："
        "简历解析 1 次，每个 JD 解析和匹配各 1 次。缓存命中时会更少。"
    )

    fallback_text = st.session_state.get("resume_fallback", "") if not pdf_text_is_valid else ""
    st.caption(
        f"AI 提供方：{get_provider_name()} ｜ 模型：{get_model_name()}（低推理强度）；"
        "简历只解析一次，每个岗位独立解析和匹配"
    )
    analysis_requested = st.button(
        "批量分析并进入岗位对比",
        type="primary",
        use_container_width=True,
    )
    if not analysis_requested:
        return

    errors: list[str] = []
    if uploaded_file is None:
        errors.append("请先上传一份 PDF 简历。")
    else:
        try:
            validate_pdf_upload(uploaded_file.name, uploaded_file.size)
        except InputValidationError as exc:
            errors.append(str(exc))
    jobs, job_errors = _validate_job_records(job_records)
    errors.extend(job_errors)

    resume_text: Optional[str] = None
    resume_source: Optional[str] = None
    if uploaded_file is not None:
        try:
            resume_text, resume_source = select_resume_text(
                pdf_result.text if pdf_text_is_valid and pdf_result else None,
                st.session_state.get("resume_fallback", "") if not pdf_text_is_valid else None,
            )
        except InputValidationError as exc:
            errors.append(str(exc))
    if errors:
        st.error("请修正以下问题：")
        for error in dict.fromkeys(errors):
            st.write(f"- {error}")
        return

    assert resume_text is not None and resume_source is not None
    resume_id = _resume_fingerprint(uploaded_file, fallback_text)
    if _jobs_need_model(resume_id, jobs) and not has_api_key():
        st.warning(
            "输入已通过校验，但尚未配置 OPENAI_API_KEY，无法分析未缓存的岗位。"
        )
        return

    st.success(f"输入校验通过，将分析 {len(jobs)} 个岗位。")
    with st.expander("查看脱敏输入预览"):
        st.markdown("**简历文本预览**")
        st.text(_preview(redact_sensitive_info(resume_text), RESUME_PREVIEW_LIMIT))
        for index, job in enumerate(jobs, start=1):
            st.markdown(f"**岗位 {index}：{job.company} · {job.job_title}**")
            st.text(_preview(job.jd_text, JD_PREVIEW_LIMIT))

    try:
        provider = _LazySessionProvider()
        with st.status("正在批量分析岗位……", expanded=True) as status:
            resume_profile, resume_cached = _cached_resume_profile(
                resume_id,
                resume_text,
                provider,
            )
            st.write("已复用简历解析。" if resume_cached else "简历解析完成。")
            analysed_jobs: list[
                tuple[str, JobProfile, PreliminaryAnalysis, str | None]
            ] = []
            for index, job in enumerate(jobs, start=1):
                job_id, job_profile, preliminary, cache_state = _cached_job_analysis(
                    resume_id,
                    resume_profile,
                    resume_text,
                    job,
                    provider,
                )
                analysed_jobs.append((job_id, job_profile, preliminary, job.job_url))
                st.write(
                    f"岗位 {index}/{len(jobs)}：{job.company} · {job.job_title} "
                    + ("已从缓存恢复。" if all(cache_state) else "分析完成。")
                )
            status.update(label="批量分析完成", state="complete", expanded=False)
    except (AiParserError, AiProviderError) as exc:
        st.error(str(exc))
        st.info("已完成的解析已保存在当前会话缓存中，再次提交会从中断处继续。")
        return

    previous_candidate = st.session_state.get("candidate_profile")
    same_resume = bool(
        previous_candidate and previous_candidate.get("resume_id") == resume_id
    )
    preserved_facts = previous_candidate.get("facts", []) if same_resume else []
    preserved_job_analyses = (
        st.session_state.get("job_analyses", {}) if same_resume else {}
    )
    pdf_layout = inspect_pdf_layout(uploaded_file.getvalue()).model_copy(
        update={"has_contact_details": contains_contact_details(resume_text)}
    )
    candidate_profile = {
        "resume_id": resume_id,
        "resume_profile": resume_profile.model_dump(mode="json"),
        "resume_text": redact_sensitive_info(resume_text),
        "resume_source": resume_source,
        "filename": uploaded_file.name,
        "page_count": pdf_result.page_count if pdf_result else None,
        "pdf_layout": pdf_layout.model_dump(mode="json"),
        "facts": preserved_facts,
    }
    distinct_analysed = {
        job_id: (job_profile, preliminary, job_url)
        for job_id, job_profile, preliminary, job_url in analysed_jobs
    }
    for job_id, (job_profile, preliminary, job_url) in distinct_analysed.items():
        preserved_job_analyses.setdefault(
            job_id,
            _new_job_bundle(job_id, job_profile, preliminary, job_url),
        )
    st.session_state["candidate_profile"] = candidate_profile
    st.session_state["job_analyses"] = preserved_job_analyses
    st.session_state["active_job_id"] = None
    st.session_state["workspace_notice"] = (
        f"已完成 {len(distinct_analysed)} 个不同岗位的初步分析，请选择岗位进入详细流程。"
    )
    st.rerun()


if __name__ == "__main__":
    main()
