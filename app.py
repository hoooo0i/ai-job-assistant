from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Optional

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.ai_parser import (
    AiParserError,
    create_ai_provider,
    generate_supplement_resume_suggestions,
    get_model_name,
    generate_cover_letter,
    generate_interview_copilot_guidance,
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
from src.job_link import EXTRACTION_LABELS, JobLinkError, fetch_job_posting
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
from src.realtime_transcription import (
    RealtimeAudioProcessor,
    RealtimeTranscriptionSession,
    looks_like_interview_question,
    realtime_max_seconds,
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
    InterviewCopilotGuidance,
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
from src.ui import (
    clear_session_preserving_ui_preferences,
    apply_design_system,
    count_pending_confirmations,
    default_detail_section,
    detail_sections,
    initialise_ui_state,
    normalise_selected_job_id,
    render_app_header,
    render_pill_navigation,
    render_score_donut,
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
    steps = "  /  ".join(
        f"**{index}. {label}**" if index == active_step else f"{index}. {label}"
        for index, label in enumerate(STAGE_STEPS, start=1)
    )
    st.caption(f"当前步骤 {active_step}/{len(STAGE_STEPS)}  ·  {steps}")


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
        "interview_copilot_records",
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


def _render_copilot_guidance(latest: dict) -> None:
    """Render evidence-bound guidance shared by live and manual modes."""
    guidance = InterviewCopilotGuidance.model_validate(latest["guidance"])
    evidence_by_id = {
        item["id"]: item for item in latest.get("evidence", []) if item.get("id")
    }
    st.info(f"识别到的问题：{guidance.detected_question}")
    columns = st.columns(2)
    with columns[0]:
        st.markdown("**回答结构**")
        for item in guidance.answer_framework:
            st.write(f"- {item}")
    with columns[1]:
        st.markdown("**可用要点**")
        if guidance.talking_points:
            for item in guidance.talking_points:
                st.write(f"- {item}")
        else:
            st.caption("暂无足够证据支持的个人要点。")
    if guidance.evidence_ids:
        st.markdown("**真实证据提示**")
        for identifier in guidance.evidence_ids:
            item = evidence_by_id.get(identifier)
            if item:
                st.write(f"> [{item['source']}] {item['text']}")
    if guidance.missing_information:
        st.markdown("**不足与待补充**")
        for item in guidance.missing_information:
            st.write(f"- {item}")
    for note in guidance.caution_notes:
        st.warning(note)


def _realtime_state_key(fingerprint: str) -> str:
    return f"_copilot_live_state_{fingerprint}"


def _realtime_worker_key(fingerprint: str) -> str:
    return f"_copilot_live_worker_{fingerprint}"


def _get_realtime_worker(fingerprint: str) -> RealtimeTranscriptionSession:
    key = _realtime_worker_key(fingerprint)
    worker = st.session_state.get(key)
    if not isinstance(worker, RealtimeTranscriptionSession):
        worker = RealtimeTranscriptionSession()
        st.session_state[key] = worker
    return worker


def _get_realtime_state(fingerprint: str) -> dict:
    return st.session_state.setdefault(
        _realtime_state_key(fingerprint),
        {
            "status": "等待开启麦克风",
            "partials": {},
            "completed": [],
            "analysed": [],
            "session_counted": False,
            "error": "",
        },
    )


def _generate_copilot_record(
    transcript: str,
    bundle: dict,
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
    analysis: MatchAnalysis,
) -> None:
    guidance, evidence = generate_interview_copilot_guidance(
        transcript,
        resume_profile,
        job_profile,
        analysis,
        provider=_create_counted_provider(),
    )
    records = bundle.setdefault("interview_copilot_records", [])
    records.append(
        {
            "transcript": redact_sensitive_info(transcript),
            "guidance": guidance.model_dump(mode="json"),
            "evidence": [item.__dict__ for item in evidence],
        }
    )
    bundle["interview_copilot_records"] = records[-10:]
    _save_active_job_analysis(bundle)


@st.fragment(run_every=1.0)
def _render_realtime_updates(
    job_id: str,
    fingerprint: str,
    auto_guidance: bool,
) -> None:
    worker = _get_realtime_worker(fingerprint)
    live_state = _get_realtime_state(fingerprint)
    for event in worker.drain_events():
        event_type = event.get("type")
        if event_type == "connected":
            live_state["status"] = "实时转写已连接"
            live_state["error"] = ""
            if not live_state.get("session_counted"):
                st.session_state["model_call_count"] = (
                    st.session_state.get("model_call_count", 0) + 1
                )
                live_state["session_counted"] = True
        elif event_type == "delta":
            item_id = event.get("item_id", "current")
            live_state["partials"][item_id] = (
                live_state["partials"].get(item_id, "") + event.get("text", "")
            )
        elif event_type == "completed":
            item_id = event.get("item_id", "current")
            live_state["partials"].pop(item_id, None)
            transcript = redact_sensitive_info(event.get("text", "")).strip()
            if transcript and transcript not in live_state["completed"]:
                live_state["completed"] = (
                    live_state["completed"] + [transcript]
                )[-12:]
        elif event_type == "error":
            live_state["error"] = event.get("text", "实时转写失败。")
            live_state["status"] = "连接异常"
        elif event_type == "limit":
            live_state["error"] = event.get("text", "本次会话已结束。")
        elif event_type == "stopped":
            live_state["status"] = "实时转写已停止"

    job_analyses = st.session_state.get("job_analyses", {})
    candidate_profile = st.session_state.get("candidate_profile")
    bundle = job_analyses.get(job_id)
    if not bundle or not candidate_profile:
        st.info("当前岗位会话已结束。")
        return
    resume_profile = ResumeProfile.model_validate(candidate_profile["resume_profile"])
    job_profile = JobProfile.model_validate(bundle["job_profile"])
    analysis = _bundle_analysis(bundle)

    latest_turn = live_state["completed"][-1] if live_state["completed"] else ""
    should_generate = (
        auto_guidance
        and latest_turn
        and latest_turn not in live_state["analysed"]
        and looks_like_interview_question(latest_turn)
    )
    manual_generate = False
    if latest_turn and not auto_guidance:
        manual_generate = st.button(
            "为最新转写生成证据提示",
            key=f"live_manual_generate_{fingerprint}",
            use_container_width=True,
        )
    if should_generate or manual_generate:
        live_state["analysed"] = (live_state["analysed"] + [latest_turn])[-20:]
        try:
            with st.spinner("已识别到面试问题，正在匹配简历证据……"):
                _generate_copilot_record(
                    latest_turn,
                    bundle,
                    resume_profile,
                    job_profile,
                    analysis,
                )
        except (AiParserError, AiProviderError) as exc:
            live_state["error"] = str(exc)

    status_icon = "🟢" if worker.running else "⚪"
    st.caption(f"{status_icon} {live_state['status']}")
    if live_state.get("error"):
        st.error(live_state["error"])
    partial_text = " ".join(live_state["partials"].values()).strip()
    if partial_text:
        st.markdown("**正在识别**")
        st.info(redact_sensitive_info(partial_text))
    if live_state["completed"]:
        with st.expander("最近实时转写", expanded=False):
            for turn in live_state["completed"][-5:]:
                st.write(f"- {turn}")

    records = bundle.get("interview_copilot_records", [])
    if records:
        st.markdown("### 最新即时提示")
        _render_copilot_guidance(records[-1])
        st.caption(f"当前会话已保留 {len(records)} 次提示，最多 10 次。")


def _stop_realtime_workers() -> None:
    for value in list(st.session_state.values()):
        if isinstance(value, RealtimeTranscriptionSession):
            value.stop()


def _render_interview_copilot(
    bundle: dict,
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
    analysis: MatchAnalysis,
) -> None:
    st.subheader("实时面试辅助")
    st.caption(
        "开启后会连续接收麦克风音频、实时显示转写，并在识别到问题时"
        "自动给出回答结构、关键词和可核对证据；不需要逐段录制或上传。"
    )
    st.warning(
        "只能在已获得所有面试参与者同意的情况下使用。"
        "系统不生成整段代答；原始音频不会写入文件、岗位档案或下载报告。"
    )
    consent = st.checkbox(
        "我确认已获得录音与实时转写所需的同意",
        key=f"copilot_consent_{bundle['fingerprint']}",
    )
    auto_guidance = st.toggle(
        "自动识别面试问题并生成证据提示",
        value=True,
        disabled=not consent,
        key=f"copilot_auto_{bundle['fingerprint']}",
        help="麦克风无法完全区分面试官和候选人；关闭后可手动确认最新转写。",
    )
    if consent:
        worker = _get_realtime_worker(bundle["fingerprint"])
        processor_key = f"_copilot_audio_processor_{bundle['fingerprint']}"
        processor = st.session_state.get(processor_key)
        if not isinstance(processor, RealtimeAudioProcessor):
            processor = RealtimeAudioProcessor(worker)
            st.session_state[processor_key] = processor
        webrtc_streamer(
            key=f"copilot_live_mic_{bundle['fingerprint']}",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": False, "audio": True},
            audio_frame_callback=processor,
            on_audio_ended=worker.stop,
            async_processing=True,
            sendback_audio=False,
        )
        st.caption(
            "点击上方 START 开始、STOP 停止。首次使用请允许浏览器访问麦克风。"
            f"单次最长 {realtime_max_seconds() // 60} 分钟，停止后可再次开始。"
        )
        _render_realtime_updates(
            bundle["job_id"],
            bundle["fingerprint"],
            auto_guidance,
        )
    else:
        worker = st.session_state.get(_realtime_worker_key(bundle["fingerprint"]))
        if isinstance(worker, RealtimeTranscriptionSession) and worker.running:
            worker.stop()
        st.info("勾选同意后才会显示实时麦克风开关。")

    with st.expander("备用：手动输入面试问题"):
        typed_question = st.text_area(
            "输入面试官的问题",
            height=100,
            disabled=not consent,
            placeholder="例如：请介绍一个你用数据推动产品决策的例子。",
            key=f"copilot_typed_{bundle['fingerprint']}",
        )
        if st.button(
            "生成证据提示",
            type="primary",
            disabled=(not consent or not typed_question.strip()),
            key=f"copilot_generate_{bundle['fingerprint']}",
        ):
            try:
                with st.spinner("正在匹配 JD 与简历证据……"):
                    _generate_copilot_record(
                        typed_question.strip(),
                        bundle,
                        resume_profile,
                        job_profile,
                        analysis,
                    )
                st.rerun()
            except (AiParserError, AiProviderError) as exc:
                st.error(str(exc))

    records = bundle.get("interview_copilot_records", [])
    if records and st.button(
        "清空面试辅助记录",
        key=f"copilot_clear_{bundle['fingerprint']}",
    ):
        bundle.pop("interview_copilot_records", None)
        live_state = _get_realtime_state(bundle["fingerprint"])
        live_state.update({"partials": {}, "completed": [], "analysed": []})
        _save_active_job_analysis(bundle)
        st.rerun()


def _render_interview_center(
    bundle: dict,
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
    analysis: MatchAnalysis,
) -> None:
    live_column, practice_column = st.columns([1.08, 1], gap="large")
    with live_column:
        with st.container(key="interview_live_panel"):
            _render_interview_copilot(
                bundle,
                resume_profile,
                job_profile,
                analysis,
            )

    with practice_column:
        with st.container(key="interview_practice_panel"):
            st.subheader("题目练习与复盘")
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
                with st.expander(
                    f"{category} {index} · {question.question}",
                    expanded=index == 1,
                ):
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
                        _render_preparation(
                            preparation,
                            stored_preparation["evidence"],
                        )

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
                            [
                                InterviewEvidence(**item)
                                for item in stored_preparation["evidence"]
                            ]
                            if stored_preparation
                            else collect_interview_evidence(
                                question,
                                resume_profile,
                                analysis,
                            )
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
                            InterviewFeedback.model_validate(
                                feedback_records[key]["feedback"]
                            )
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
        help=(
            "支持字节跳动、腾讯、小米、Moka、智联招聘及其他公开岗位详情页。"
            "小米请先在职位列表打开具体岗位，再复制详情页链接；"
            "Moka 请复制地址中包含 #/job/ 的具体岗位链接；"
            "如网站要求登录或验证，请手动粘贴 JD。"
        ),
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
            method = EXTRACTION_LABELS.get(imported.extraction_method, "网页内容")
            st.success(
                f"岗位 {number} 已读取（{method}），请核对下方自动填写内容。"
            )
            missing = [
                label
                for label, value in (
                    ("公司名称", imported.company),
                    ("岗位名称", imported.title),
                )
                if not value
            ]
            if missing:
                st.info(f"网页没有明确提供{'、'.join(missing)}，请手动补充。")
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


def _must_have_coverage(bundle: dict) -> float:
    job_profile = JobProfile.model_validate(bundle["job_profile"])
    analysis = _bundle_analysis(bundle)
    matches = {item.requirement_id: item for item in analysis.matches}
    must_have = [
        item for item in job_profile.requirements if item.importance.value == "must_have"
    ]
    if not must_have:
        return 100.0
    weights = {
        MatchStatus.matched: 1.0,
        MatchStatus.partial: 0.5,
        MatchStatus.missing: 0.0,
        MatchStatus.unknown: 0.0,
    }
    covered = sum(
        weights.get(matches[item.id].status, 0.0)
        for item in must_have
        if item.id in matches
    )
    return round(covered / len(must_have) * 100, 1)


def _top_requirement_gaps(bundle: dict, limit: int = 2) -> list[str]:
    job_profile = JobProfile.model_validate(bundle["job_profile"])
    analysis = _bundle_analysis(bundle)
    matches = {item.requirement_id: item for item in analysis.matches}
    importance_rank = {"must_have": 0, "preferred": 1, "other": 2}
    gaps = [
        requirement
        for requirement in job_profile.requirements
        if requirement.id in matches
        and matches[requirement.id].status
        in {MatchStatus.missing, MatchStatus.unknown, MatchStatus.partial}
    ]
    gaps.sort(
        key=lambda item: (
            not item.is_hard_condition,
            importance_rank.get(item.importance.value, 3),
            item.normalized_name,
        )
    )
    return [item.normalized_name for item in gaps[:limit]]


def _job_risk_summary(item) -> str:
    if item.hard_risks:
        return f"{item.hard_risks} 项硬性风险"
    if item.must_have_gaps:
        return f"{item.must_have_gaps} 项必须能力缺口"
    if item.stage != "final":
        return "待补充确认"
    return "暂无明显硬性风险"


def _enter_job_detail(job_id: str) -> None:
    st.session_state["active_job_id"] = job_id
    st.session_state["workspace_section"] = "岗位"


def _select_comparison_job(job_id: str) -> None:
    st.session_state["comparison_selected_job_id"] = job_id


def _return_to_job_workspace(job_id: str) -> None:
    st.session_state["comparison_selected_job_id"] = job_id
    st.session_state["workspace_section"] = "岗位"
    st.session_state["active_job_id"] = None


@st.dialog("添加岗位 JD", width="large")
def _render_add_jobs_dialog(
    candidate_profile: dict,
    job_analyses: dict[str, dict],
    remaining: int,
) -> None:
    st.caption(
        f"当前还可添加 {remaining} 个岗位。新增岗位会复用现有简历解析结果。"
    )
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
        _render_job_input_fields(batch_prefix, index) for index in range(add_count)
    ]
    st.caption(
        f"若均为新岗位，本次最多增加 {add_count * 2} 次模型调用；"
        "简历解析不会重复调用。"
    )
    if not st.button(
        "分析并加入对比",
        type="primary",
        icon=":material/add_task:",
        use_container_width=True,
    ):
        return

    jobs, errors = _validate_job_records(records)
    if errors:
        st.error("请修正以下问题：")
        for error in errors:
            st.write(f"- {error}")
        return
    if _jobs_need_model(
        candidate_profile["resume_id"],
        jobs,
        resume_profile_available=True,
    ) and not has_api_key():
        st.error("尚未配置 OPENAI_API_KEY，无法分析新岗位。")
        return
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


@st.fragment
def _render_job_decision_board(
    candidate_profile: dict,
    job_analyses: dict[str, dict],
    comparison: list,
) -> None:
    job_ids = [item.job_id for item in comparison]
    selected_job_id = normalise_selected_job_id(
        job_ids,
        st.session_state.get("comparison_selected_job_id"),
    )
    st.session_state["comparison_selected_job_id"] = selected_job_id
    if not selected_job_id:
        st.info("还没有可比较的岗位。")
        return

    list_column, detail_column = st.columns([1.85, 1.05], gap="large")
    with list_column:
        with st.container(key="job_board_list"):
            with st.container(key="job_board_header"):
                header = st.columns([0.45, 2.15, 1, 1.2, 1.3, 0.8])
                for column, label in zip(
                    header,
                    ["排名", "岗位信息", "总体匹配", "必须项覆盖", "关键风险", "操作"],
                ):
                    column.caption(label)

            for index, item in enumerate(comparison, start=1):
                selected = item.job_id == selected_job_id
                container_key = (
                    "job_row_selected" if selected else f"job_row_{item.job_id[:10]}"
                )
                with st.container(key=container_key):
                    columns = st.columns([0.45, 2.15, 1, 1.2, 1.3, 0.8])
                    columns[0].markdown(f"### {index}")
                    columns[1].markdown(f"**{item.title}**")
                    columns[1].caption(
                        f"{item.company} · "
                        f"{APPLICATION_STATUS_LABELS[ApplicationStatus(item.application_status)]}"
                    )
                    columns[2].metric(
                        "匹配",
                        "--" if item.match_score is None else f"{item.match_score:.0f}%",
                        label_visibility="collapsed",
                    )
                    coverage = _must_have_coverage(job_analyses[item.job_id])
                    columns[3].write(f"**{coverage:.0f}%**")
                    columns[3].progress(coverage / 100)
                    columns[4].write(_job_risk_summary(item))
                    columns[4].caption(
                        "最终分析" if item.stage == "final" else "待补充确认"
                    )
                    columns[5].button(
                        "已选择" if selected else "预览",
                        key=f"preview_job_{item.job_id}",
                        disabled=selected,
                        use_container_width=True,
                        on_click=_select_comparison_job,
                        args=(item.job_id,),
                    )

    selected_item = next(
        item for item in comparison if item.job_id == selected_job_id
    )
    selected_bundle = job_analyses[selected_job_id]
    selected_profile = JobProfile.model_validate(selected_bundle["job_profile"])
    selected_analysis = _bundle_analysis(selected_bundle)
    selected_score = calculate_scores(selected_profile, selected_analysis)
    with detail_column:
        with st.container(key="selected_job_panel"):
            st.caption("当前选择")
            st.markdown(f"## {selected_profile.title}")
            st.write(
                " · ".join(
                    value
                    for value in [
                        selected_profile.company,
                        selected_profile.location,
                        selected_profile.job_type,
                    ]
                    if value
                )
            )
            render_score_donut(selected_score.match_score)
            metric_columns = st.columns(3)
            metric_columns[0].metric(
                "必须项",
                f"{_must_have_coverage(selected_bundle):.0f}%",
            )
            metric_columns[1].metric(
                "完整度", f"{selected_score.information_completeness:.0f}%"
            )
            metric_columns[2].metric("ATS", f"{selected_item.ats_score}")

            if selected_item.recommendation_score >= 75:
                recommendation = "匹配度较高，建议优先投入并针对关键差距优化。"
            elif selected_item.recommendation_score >= 55:
                recommendation = "具备一定基础，建议先补强高影响证据再投递。"
            else:
                recommendation = "当前差距较明显，建议先核对硬性条件与投入成本。"
            st.info(f"AI 建议：{recommendation}")

            gaps = _top_requirement_gaps(selected_bundle)
            st.markdown("#### 关键差距")
            if gaps:
                for index, gap in enumerate(gaps, start=1):
                    st.write(f"{index}. {gap}")
            else:
                st.success("当前未发现高优先级缺口。")

            enter_label = (
                "进入详细分析"
                if selected_item.stage == "final"
                else "进入补充确认"
            )
            if st.button(
                enter_label,
                key=f"enter_job_{selected_job_id}",
                type="primary",
                icon=":material/arrow_forward:",
                icon_position="right",
                use_container_width=True,
            ):
                _enter_job_detail(selected_job_id)
                # This control lives inside a fragment so its default rerun would
                # only repaint the decision board. Entering a job changes the app
                # route and therefore requires an explicit full-app rerun.
                st.rerun(scope="app")


def _render_job_workspace(candidate_profile: dict, job_analyses: dict[str, dict]) -> None:
    notice = st.session_state.pop("workspace_notice", None)
    if notice:
        st.success(notice)

    comparison = build_job_comparison(candidate_profile, job_analyses)
    highest_match = max(
        (item.match_score for item in comparison if item.match_score is not None),
        default=None,
    )
    pending = count_pending_confirmations(job_analyses.values())
    remaining = MAX_JOBS_PER_SESSION - len(job_analyses)

    with st.container(key="workspace_hero"):
        title_column, action_column = st.columns(
            [4, 1], vertical_alignment="bottom"
        )
        with title_column:
            st.title("选择最值得投入的岗位")
            st.caption(
                f"基于 {candidate_profile.get('filename', '当前简历')} 的证据匹配，"
                "先比较机会，再进入单个岗位精细优化。"
            )
        with action_column:
            if remaining > 0 and st.button(
                "添加岗位 JD",
                icon=":material/add:",
                use_container_width=True,
            ):
                _render_add_jobs_dialog(candidate_profile, job_analyses, remaining)

    with st.container(key="workspace_metrics"):
        metric_columns = st.columns(3)
        metric_columns[0].metric("已分析", f"{len(comparison)} 个岗位")
        metric_columns[1].metric(
            "最高匹配",
            "--" if highest_match is None else f"{highest_match:.0f}%",
        )
        metric_columns[2].metric("待确认", f"{pending} 项")

    st.caption("推荐排序由本地规则计算，仅用于安排投递优先级，不代表录取概率。")
    _render_job_decision_board(candidate_profile, job_analyses, comparison)

    tracking_metrics = build_application_metrics(job_analyses)
    actions = upcoming_application_actions(job_analyses)
    with st.expander("投递进度、提醒与下载"):
        metric_columns = st.columns(5)
        metric_columns[0].metric("岗位总数", tracking_metrics.total_jobs)
        metric_columns[1].metric("已投递", tracking_metrics.submitted)
        metric_columns[2].metric("回复率", f"{tracking_metrics.response_rate:.1f}%")
        metric_columns[3].metric("面试率", f"{tracking_metrics.interview_rate:.1f}%")
        metric_columns[4].metric("Offer", tracking_metrics.offers)
        if actions:
            st.markdown("#### 近期提醒")
            for action in actions:
                timing = "已逾期" if action["timing"] == "已逾期" else action["timing"]
                st.write(
                    f"{action['date']} · {action['kind']} · "
                    f"{action['company']} · {action['title']}（{timing}）"
                )

        comparison_source = [item.model_dump(mode="json") for item in comparison]
        comparison_version = hashlib.sha256(
            json.dumps(
                comparison_source, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        comparison_files = st.session_state.get("comparison_report_files", {})
        if comparison_files.get("version") != comparison_version:
            comparison_files = {
                "version": comparison_version,
                "docx": build_job_comparison_docx(comparison),
                "pdf": build_job_comparison_pdf(comparison),
            }
            st.session_state["comparison_report_files"] = comparison_files
        st.caption(
            "报告不包含原始简历全文；脱敏档案由浏览器下载，本应用不会自动长期保存。"
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

    if remaining <= 0:
        st.info("当前会话已达到 5 个岗位的比较上限。")

    if st.button(
        "上传新简历并清空当前工作台",
        type="tertiary",
        icon=":material/restart_alt:",
    ):
        _stop_realtime_workers()
        clear_session_preserving_ui_preferences()
        st.rerun()


def _render_final_results(
    bundle: dict,
    candidate_profile: dict,
    active_group: str,
) -> None:
    resume_profile = ResumeProfile.model_validate(candidate_profile["resume_profile"])
    job_profile = JobProfile.model_validate(bundle["job_profile"])
    match_analysis = MatchAnalysis.model_validate(bundle["final_analysis"])
    score = calculate_scores(job_profile, match_analysis)

    workspace_notice = st.session_state.pop("workspace_notice", None)
    if workspace_notice:
        st.success(workspace_notice)

    st.caption("分析流程  /  上传资料  /  初步匹配  /  补充真实信息  /  最终建议")
    title_column, score_column = st.columns([4, 1], vertical_alignment="bottom")
    with title_column:
        st.title(f"{job_profile.title} · {job_profile.company}")
        st.caption(
            "分析已完成。所有评分均基于简历证据与用户明确确认的信息。"
        )
    with score_column:
        st.metric(
            "证据匹配度",
            "--" if score.match_score is None else f"{score.match_score:.0f}%",
        )

    with st.container(key="detail_actions"):
        action_columns = st.columns(3)
    action_columns[0].button(
        "返回岗位对比",
        icon=":material/arrow_back:",
        use_container_width=True,
        on_click=_return_to_job_workspace,
        args=(bundle["job_id"],),
    )
    if action_columns[1].button(
        "修改补充信息",
        icon=":material/edit_note:",
        use_container_width=True,
    ):
        # Keep the current final result available until the user actually submits
        # changed answers. This makes entering the edit screen fully reversible.
        bundle["stage"] = "clarification"
        bundle["editing_clarifications"] = True
        _save_active_job_analysis(bundle)
        st.rerun()
    if action_columns[2].button(
        "重新开始",
        icon=":material/restart_alt:",
        use_container_width=True,
    ):
        _stop_realtime_workers()
        clear_session_preserving_ui_preferences()
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

    sections = detail_sections(active_group)
    with st.container(key="detail_subnavigation"):
        if len(sections) > 1:
            selected_section = render_pill_navigation(
                sections,
                state_key=f"detail_section_{bundle['job_id']}_{active_group}",
                default=default_detail_section(active_group),
                container_key="secondary_navigation",
            )
        else:
            selected_section = sections[0]
            st.caption(f"当前功能 · {selected_section}")

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
    elif selected_section == "面试辅助":
        _render_interview_center(bundle, resume_profile, job_profile, match_analysis)
    elif selected_section == "报告下载":
        _render_report_download(bundle, resume_profile, job_profile, match_analysis, score)
    elif selected_section == "简历结构":
        _render_resume_profile(resume_profile)
    elif selected_section == "JD 结构":
        _render_job_profile(job_profile)
        st.info(
            "隐私说明：原始 PDF、提取文字和粘贴内容仅用于当前会话；"
            "模型输入、预览和导出内容继续执行敏感信息脱敏。"
        )
    st.caption(
        "本阶段不预测录取概率，不补写不存在的经历，也不使用 RAG 或长期保存数据。"
    )


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
    bundle["interview_copilot_records"] = []
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
                _stop_realtime_workers()
                clear_session_preserving_ui_preferences()
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
    st.set_page_config(
        page_title="AI 求职助手",
        page_icon=":material/work:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    initialise_ui_state()
    apply_design_system()

    active_job_id = st.session_state.get("active_job_id")
    job_analyses = st.session_state.setdefault("job_analyses", {})
    candidate_profile = st.session_state.get("candidate_profile")

    active_job = (
        job_analyses.get(active_job_id) if active_job_id in job_analyses else None
    )
    label_bundle = active_job
    if label_bundle is None and job_analyses:
        selected_for_label = st.session_state.get("comparison_selected_job_id")
        label_bundle = job_analyses.get(selected_for_label)
        if label_bundle is None:
            label_bundle = next(iter(job_analyses.values()))
    job_label = None
    if label_bundle:
        label_profile = JobProfile.model_validate(label_bundle["job_profile"])
        job_label = f"{label_profile.title} · {label_profile.company}"

    detail_is_ready = bool(
        active_job
        and active_job.get("stage") == "final"
        and active_job.get("final_analysis")
    )
    navigation_enabled = bool(
        candidate_profile and job_analyses and (active_job is None or detail_is_ready)
    )
    active_group = render_app_header(
        job_label=job_label,
        model_call_count=st.session_state.get("model_call_count", 0),
        navigation_enabled=navigation_enabled,
        force_group="岗位" if not navigation_enabled else None,
    )

    if active_job_id and active_job_id in job_analyses and candidate_profile:
        assert active_job is not None
        if detail_is_ready:
            _render_final_results(active_job, candidate_profile, active_group)
        else:
            _render_clarification_stage(active_job, candidate_profile)
        return
    if candidate_profile and job_analyses:
        if active_group != "岗位":
            comparison_ids = [
                item.job_id
                for item in build_job_comparison(candidate_profile, job_analyses)
            ]
            selected_job_id = normalise_selected_job_id(
                comparison_ids,
                st.session_state.get("comparison_selected_job_id"),
            )
            if selected_job_id:
                st.session_state["active_job_id"] = selected_job_id
                st.rerun()
        _render_job_workspace(candidate_profile, job_analyses)
        return

    st.title("从一份简历，开始比较机会")
    st.caption("一次提交多个岗位，先比较投入价值，再进入单个岗位精细优化。")
    _render_archive_import()
    _render_step_progress(1)
    resume_column, jobs_column = st.columns([1, 1.65], gap="large")
    with resume_column:
        with st.container(key="upload_panel"):
            st.subheader("上传简历")
            st.caption("优先读取文本型 PDF；文件上限为 10 MB。")
            uploaded_file = st.file_uploader(
                "上传一份 PDF 简历（必填）",
                type=["pdf"],
                accept_multiple_files=False,
                key="resume_upload",
            )

            pdf_result, pdf_error = _parse_uploaded_pdf(uploaded_file)
            pdf_text_is_valid = bool(
                pdf_result and has_valid_resume_text(pdf_result.text)
            )

            if uploaded_file is None:
                st.session_state["resume_fallback"] = ""
            elif pdf_text_is_valid:
                st.session_state["resume_fallback"] = ""
                st.success(
                    f"PDF 解析成功：共 {pdf_result.page_count} 页，将优先使用 PDF 文本。"
                )
            else:
                if pdf_result is not None:
                    pdf_error = "PDF 中提取到的有效文字少于 50 个字符。"
                st.warning(
                    f"{pdf_error or 'PDF 无法提取有效文字。'} "
                    "请粘贴简历内容，或重新上传文本型 PDF。"
                )
                st.text_area(
                    "粘贴简历内容（备用入口）",
                    key="resume_fallback",
                    height=220,
                    placeholder="请粘贴至少 50 个非空白字符……",
                    help=(
                        "仅当 PDF 无法提取有效文字时使用。"
                        "内容仅用于当前会话，不会被长期保存。"
                    ),
                )
                st.caption(
                    "粘贴内容不会被长期保存；重新上传有效 PDF 后将自动改用 PDF 文本。"
                )

            st.info("隐私保护已开启：原始文件不落盘，预览与模型输入会先脱敏。")
            with st.expander("查看隐私与模型说明"):
                st.write(
                    "原始 PDF、提取文字和粘贴内容仅用于当前会话，不会被本应用长期保存。"
                    "电话、邮箱和详细地址会被自动脱敏。开始分析后，脱敏简历文本和岗位信息"
                    "将发送给 OpenAI，并设置 store=False。"
                )

    with jobs_column:
        with st.container(key="job_input_panel"):
            st.subheader("添加岗位")
            st.caption("支持一次分析多份 JD，后续仍可继续补充岗位进行比较。")
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
                _render_job_input_fields("initial", index)
                for index in range(job_count)
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
        icon=":material/analytics:",
        icon_position="right",
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
