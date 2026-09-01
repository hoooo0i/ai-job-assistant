from __future__ import annotations

import hashlib
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from src.ai_parser import (
    AiParserError,
    create_openai_client,
    get_model_name,
    has_api_key,
    match_requirements,
    parse_job,
    parse_resume,
)
from src.matching import apply_user_confirmations, calculate_scores
from src.pdf_parser import (
    EncryptedPdfError,
    InvalidPdfError,
    PdfExtractionResult,
    PdfReadError,
    extract_pdf_text,
)
from src.privacy import redact_sensitive_info
from src.schemas import JobProfile, MatchAnalysis, MatchStatus, ResumeProfile, ScoreResult
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


def _format_job_errors(exc: ValidationError) -> list[str]:
    labels = {
        "company": "公司名称",
        "job_title": "岗位名称",
        "jd_text": "岗位 JD",
        "location": "工作地点",
        "job_type": "岗位类型",
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
            st.markdown("**简历证据**")
            if match.resume_evidence:
                for evidence in match.resume_evidence:
                    st.write(f"> {evidence}")
            else:
                st.caption("没有可验证的简历原文证据。")
            if match.status is MatchStatus.unknown:
                st.info(f"待确认：你是否满足“{requirement.original_text}”？")


def _render_action_plan(
    analysis: MatchAnalysis,
    base_analysis: MatchAnalysis,
    job_profile: JobProfile,
    fingerprint: str,
    current_answers: dict[str, str],
) -> Optional[dict[str, str]]:
    requirements_by_id = {
        requirement.id: requirement for requirement in job_profile.requirements
    }

    st.markdown("### 待确认事项")
    unknown_matches = [
        match for match in base_analysis.matches if match.status is MatchStatus.unknown
    ]
    if not unknown_matches:
        st.caption("当前没有需要补充确认的岗位条件。")
    else:
        option_labels = ["尚未回答", "符合", "不符合"]
        option_values = {
            "尚未回答": "unknown",
            "符合": "matched",
            "不符合": "missing",
        }
        reverse_labels = {value: label for label, value in option_values.items()}
        selected_answers: dict[str, str] = {}
        with st.form(f"confirmation_form_{fingerprint}"):
            for match in unknown_matches:
                requirement = requirements_by_id[match.requirement_id]
                current_label = reverse_labels.get(
                    current_answers.get(match.requirement_id, "unknown"),
                    "尚未回答",
                )
                selected_label = st.selectbox(
                    f"你是否满足“{requirement.original_text}”？",
                    options=option_labels,
                    index=option_labels.index(current_label),
                    key=f"confirmation_{fingerprint}_{match.requirement_id}",
                )
                selected_answers[match.requirement_id] = option_values[selected_label]
            submitted = st.form_submit_button("更新匹配度与信息完整度", type="primary")
        if submitted:
            return selected_answers

    st.markdown("### 简历优化建议")
    st.caption("建议只能重组或强化已有事实；程序会过滤原文不存在或新增数字的改写。")
    if not analysis.resume_suggestions:
        st.caption("本次没有通过证据校验的简历改写建议。")
    for index, suggestion in enumerate(analysis.resume_suggestions, start=1):
        related_names = [
            requirements_by_id[identifier].normalized_name
            for identifier in suggestion.requirement_ids
        ]
        with st.expander(f"建议 {index} · {'、'.join(related_names)}", expanded=index == 1):
            st.markdown("**简历原文**")
            st.write(f"> {suggestion.original_text}")
            st.markdown("**建议版本**")
            st.write(suggestion.suggested_text)
            st.markdown("**修改理由**")
            st.write(suggestion.reason)
            if suggestion.follow_up_question:
                st.info(f"需要你补充确认：{suggestion.follow_up_question}")

    st.markdown("### 面试准备")
    category_labels = {
        "job_knowledge": "岗位知识",
        "behavioral": "行为面试",
        "project_deep_dive": "项目深挖",
        "capability_gap": "能力缺口",
    }
    if not analysis.interview_questions:
        st.caption("本次没有生成通过校验的面试问题。")
    for index, question in enumerate(analysis.interview_questions, start=1):
        related_names = [
            requirements_by_id[identifier].normalized_name
            for identifier in question.requirement_ids
        ]
        category = category_labels[question.category.value]
        with st.expander(f"{category} {index} · {question.question}"):
            st.markdown("**为什么可能被问**")
            st.write(question.why_asked)
            st.markdown("**答题准备思路**")
            for item in question.answer_outline:
                st.write(f"- {item}")
            st.caption("关联要求：" + "、".join(related_names))
    return None


def _render_analysis_bundle(bundle: dict) -> None:
    resume_profile = ResumeProfile.model_validate(bundle["resume_profile"])
    job_profile = JobProfile.model_validate(bundle["job_profile"])
    base_analysis = MatchAnalysis.model_validate(bundle["base_analysis"])
    match_analysis = MatchAnalysis.model_validate(bundle["match_analysis"])
    score = calculate_scores(job_profile, match_analysis)

    st.subheader("分析结果")
    match_tab, action_tab, resume_tab, job_tab = st.tabs(
        ["证据匹配", "行动建议", "简历结构", "JD 结构"]
    )
    with match_tab:
        _render_match_analysis(match_analysis, score, job_profile)
    with action_tab:
        updated_answers = _render_action_plan(
            match_analysis,
            base_analysis,
            job_profile,
            bundle["fingerprint"],
            bundle.get("confirmation_answers", {}),
        )
    with resume_tab:
        _render_resume_profile(resume_profile)
    with job_tab:
        _render_job_profile(job_profile)
    st.caption("本阶段不预测录取概率，不生成或补写简历经历，也不使用 RAG。")

    if updated_answers is not None:
        updated_analysis = apply_user_confirmations(base_analysis, updated_answers)
        bundle["match_analysis"] = updated_analysis.model_dump(mode="json")
        bundle["confirmation_answers"] = updated_answers
        st.session_state["analysis_bundle"] = bundle
        st.rerun()


def main() -> None:
    load_dotenv(override=True)
    st.set_page_config(page_title="AI 求职助手", page_icon="📄", layout="centered")

    st.title("AI 求职助手")
    st.caption("第五阶段 MVP：行动建议、补充确认与即时重新评分")
    st.info(
        "隐私说明：原始 PDF、提取文字和粘贴内容仅用于当前会话，不会被本应用长期保存。"
        "预览中的电话、邮箱和详细地址会被自动脱敏。开始分析后，脱敏简历文本和岗位信息"
        "将发送给 OpenAI 进行结构化解析，并在 API 请求中设置 store=False。"
    )

    st.subheader("1. 上传简历")
    uploaded_file = st.file_uploader(
        "上传一份 PDF 简历（必填，最大 10 MB）",
        type=["pdf"],
        accept_multiple_files=False,
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

    st.subheader("2. 填写岗位信息")
    company = st.text_input("公司名称（必填）")
    job_title = st.text_input("岗位名称（必填）")
    location = st.text_input("工作地点（选填）")
    job_type = st.selectbox(
        "岗位类型（选填）",
        options=["", "全职", "实习", "兼职", "合同", "其他"],
        format_func=lambda value: "请选择" if value == "" else value,
    )
    jd_text = st.text_area(
        "岗位 JD（必填）",
        height=260,
        placeholder="请粘贴完整岗位描述，至少 50 个非空白字符……",
    )

    fallback_text = st.session_state.get("resume_fallback", "") if not pdf_text_is_valid else ""
    fingerprint = _analysis_fingerprint(
        uploaded_file,
        company,
        job_title,
        location,
        job_type,
        jd_text,
        fallback_text,
    )
    existing_bundle = st.session_state.get("analysis_bundle")
    if existing_bundle and existing_bundle.get("fingerprint") != fingerprint:
        st.session_state.pop("analysis_bundle", None)
        existing_bundle = None

    st.caption(f"语义解析与匹配模型：{get_model_name()}（低推理强度）；最终分数由 Python 计算")
    analysis_requested = st.button("开始分析", type="primary", use_container_width=True)
    if not analysis_requested:
        if existing_bundle:
            _render_analysis_bundle(existing_bundle)
        return

    errors: list[str] = []
    job: Optional[JobInput] = None

    if uploaded_file is None:
        errors.append("请先上传一份 PDF 简历。")
    else:
        try:
            validate_pdf_upload(uploaded_file.name, uploaded_file.size)
        except InputValidationError as exc:
            errors.append(str(exc))

    try:
        job = JobInput(
            company=company,
            job_title=job_title,
            location=location,
            job_type=job_type,
            jd_text=jd_text,
        )
    except ValidationError as exc:
        errors.extend(_format_job_errors(exc))

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

    assert job is not None and resume_text is not None and resume_source is not None

    st.success("输入校验通过，简历内容已准备完成。")
    st.subheader("分析输入预览")

    col1, col2, col3 = st.columns(3)
    col1.metric("PDF 文件名", uploaded_file.name)
    col2.metric("页数", str(pdf_result.page_count) if pdf_result else "无法读取")
    col3.metric("简历来源", "PDF 文本" if resume_source == "pdf" else "粘贴文本")

    st.markdown(f"**目标岗位：** {job.company} · {job.job_title}")
    if job.location or job.job_type:
        st.caption(f"工作地点：{job.location or '未填写'} ｜ 岗位类型：{job.job_type or '未填写'}")

    st.markdown("**简历文本预览（已脱敏）**")
    st.text_area(
        "简历文本预览",
        value=_preview(redact_sensitive_info(resume_text), RESUME_PREVIEW_LIMIT),
        height=260,
        disabled=True,
        label_visibility="collapsed",
    )

    st.markdown("**JD 文本预览**")
    st.text_area(
        "JD 文本预览",
        value=_preview(job.jd_text, JD_PREVIEW_LIMIT),
        height=220,
        disabled=True,
        label_visibility="collapsed",
    )
    if not has_api_key():
        st.warning(
            "输入预览已完成，但尚未配置 OPENAI_API_KEY。请复制 .env.example 为 .env，"
            "填入 API 密钥后重新点击“开始分析”。"
        )
        return

    try:
        client = create_openai_client()
        with st.status("正在使用 GPT-5.5 解析并匹配证据……", expanded=True) as status:
            st.write("正在提取简历结构（发送前已脱敏）……")
            resume_profile = parse_resume(resume_text, client=client)
            st.write("正在提取岗位职责与要求……")
            job_profile = parse_job(job, client=client)
            st.write("正在逐项查找可验证的简历证据……")
            match_analysis = match_requirements(
                resume_profile,
                job_profile,
                resume_text,
                client=client,
            )
            status.update(label="分析与评分完成", state="complete", expanded=False)
    except AiParserError as exc:
        st.error(str(exc))
        st.info("已保留当前页面输入，请检查配置后直接重试。")
        return

    bundle = {
        "fingerprint": fingerprint,
        "resume_profile": resume_profile.model_dump(mode="json"),
        "job_profile": job_profile.model_dump(mode="json"),
        "base_analysis": match_analysis.model_dump(mode="json"),
        "match_analysis": match_analysis.model_dump(mode="json"),
        "confirmation_answers": {},
    }
    st.session_state["analysis_bundle"] = bundle
    _render_analysis_bundle(bundle)


if __name__ == "__main__":
    main()
