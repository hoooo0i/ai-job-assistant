from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.schemas import (
    ClarificationQuestion,
    JobProfile,
    JobRequirement,
    MatchAnalysis,
    MatchStatus,
    PreliminaryAnalysis,
    RequirementCategory,
    RequirementImportance,
    RequirementMatch,
    ResumeProfile,
)


def _clarification_state() -> tuple[dict, dict]:
    resume = ResumeProfile(
        summary="Synthetic candidate",
        education=[],
        experience=[],
        projects=[],
        skills=[],
        languages=[],
        evidence_chunks=[],
    )
    requirement = JobRequirement(
        id="req_001",
        original_text="需要 Python 项目经验",
        normalized_name="Python 项目经验",
        category=RequirementCategory.experience,
        importance=RequirementImportance.must_have,
        is_hard_condition=False,
    )
    job = JobProfile(
        company="示例公司",
        title="实习生",
        location=None,
        job_type=None,
        responsibilities=[],
        requirements=[requirement],
        domain_background=[],
    )
    match = RequirementMatch(
        requirement_id=requirement.id,
        status=MatchStatus.missing,
        resume_evidence=[],
        explanation="简历中未找到证据。",
        confidence=0.8,
    )
    question = ClarificationQuestion(
        id="cq_req_001",
        requirement_id=requirement.id,
        prompt="你是否有简历中未写明的 Python 项目经验？",
    )
    preliminary = PreliminaryAnalysis(matches=[match], clarification_questions=[question])
    candidate = {
        "resume_id": "resume_1",
        "resume_profile": resume.model_dump(mode="json"),
        "resume_text": "Synthetic resume content with no Python project evidence.",
        "resume_source": "pdf",
        "filename": "synthetic.pdf",
        "page_count": 1,
        "facts": [],
    }
    bundle = {
        "job_id": "job_1",
        "fingerprint": "job_1",
        "job_profile": job.model_dump(mode="json"),
        "preliminary_analysis": preliminary.model_dump(mode="json"),
        "clarification_questions": [question.model_dump(mode="json")],
        "clarification_answers": [],
        "stage": "clarification",
    }
    return candidate, bundle


def _button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def test_theme_toggle_persists_dark_mode() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30)

    app.run()
    app.toggle[0].set_value(True)
    app.run()

    assert not app.exception
    assert app.session_state["ui_theme"] == "dark"
    assert app.session_state["ui_theme_dark"] is True


def test_clarification_step_uses_choices_only_and_finishes_without_model_call() -> None:
    candidate, bundle = _clarification_state()
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30)
    app.session_state["candidate_profile"] = candidate
    app.session_state["job_analyses"] = {"job_1": bundle}
    app.session_state["active_job_id"] = "job_1"
    app.session_state["model_call_count"] = 3

    app.run()

    assert not app.exception
    assert any(item.value == "第 3 步：补充真实信息" for item in app.subheader)
    for label in ["岗位", "简历", "投递", "面试", "更多"]:
        assert _button(app, label).disabled is True
    assert not app.download_button
    assert not app.text_area

    app.radio[0].set_value("具备")
    submit = next(
        item for item in app.button if item.label == "完成选择并查看结果"
    )
    submit.click()
    app.run()

    final_bundle = app.session_state["job_analyses"]["job_1"]
    assert final_bundle["stage"] == "final"
    assert MatchAnalysis.model_validate(final_bundle["final_analysis"]).matches[0].status \
        is MatchStatus.partial
    assert app.session_state["model_call_count"] == 3
    assert not app.exception


def test_final_step_exposes_results_and_downloads() -> None:
    candidate, bundle = _clarification_state()
    preliminary = PreliminaryAnalysis.model_validate(bundle["preliminary_analysis"])
    bundle.update(
        {
            "stage": "final",
            "final_analysis": MatchAnalysis(matches=preliminary.matches).model_dump(mode="json"),
            "resume_edits": {},
            "cover_letters": {},
            "interview_preparations": {},
            "interview_feedback": {},
        }
    )
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30)
    app.session_state["candidate_profile"] = candidate
    app.session_state["job_analyses"] = {"job_1": bundle}
    app.session_state["active_job_id"] = "job_1"

    app.run()

    assert not app.exception
    for label in ["岗位", "简历", "投递", "面试", "更多", "岗位匹配", "关键词缺口"]:
        assert any(item.label == label for item in app.button)

    _button(app, "投递").click()
    app.run()
    assert any(item.label == "保存投递记录" for item in app.button)
    for label in ["投递管理", "材料包", "求职信", "报告下载"]:
        assert any(item.label == label for item in app.button)
    _button(app, "报告下载").click()
    app.run()
    assert {item.label for item in app.download_button} == {
        "下载 Word 报告",
        "下载 PDF 报告",
    }
    _button(app, "材料包").click()
    app.run()
    assert any(
        item.label == "下载当前投递材料包 ZIP"
        for item in app.download_button
    )
    _button(app, "面试").click()
    app.run()
    assert any(item.value == "实时面试辅助" for item in app.subheader)
    assert any(
        item.label == "我确认已获得录音与实时转写所需的同意"
        for item in app.checkbox
    )


def test_cancel_clarification_edit_returns_to_existing_final_without_regeneration() -> None:
    candidate, bundle = _clarification_state()
    preliminary = PreliminaryAnalysis.model_validate(bundle["preliminary_analysis"])
    original_final = MatchAnalysis(matches=preliminary.matches).model_dump(mode="json")
    bundle.update(
        {
            "stage": "final",
            "final_analysis": original_final,
            "resume_edits": {},
            "cover_letters": {},
            "interview_preparations": {},
            "interview_feedback": {},
        }
    )
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30)
    app.session_state["candidate_profile"] = candidate
    app.session_state["job_analyses"] = {"job_1": bundle}
    app.session_state["active_job_id"] = "job_1"

    app.run()
    next(item for item in app.button if item.label == "修改补充信息").click()
    app.run()

    editing_bundle = app.session_state["job_analyses"]["job_1"]
    assert editing_bundle["stage"] == "clarification"
    assert editing_bundle["final_analysis"] == original_final
    cancel = next(item for item in app.button if item.label == "取消修改，返回原结果")

    cancel.click()
    app.run()

    restored_bundle = app.session_state["job_analyses"]["job_1"]
    assert restored_bundle["stage"] == "final"
    assert restored_bundle["final_analysis"] == original_final
    assert "editing_clarifications" not in restored_bundle
    assert any("已返回上一次生成的结果" in item.value for item in app.success)


def test_workspace_compares_jobs_and_opens_detail_only_after_selection() -> None:
    candidate, first = _clarification_state()
    second_job = JobProfile.model_validate(first["job_profile"]).model_copy(
        update={"company": "第二家公司", "title": "数据分析师"}
    )
    second = {
        **first,
        "job_id": "job_2",
        "fingerprint": "job_2",
        "job_profile": second_job.model_dump(mode="json"),
    }
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30)
    app.session_state["candidate_profile"] = candidate
    app.session_state["job_analyses"] = {"job_1": first, "job_2": second}
    app.session_state["active_job_id"] = None

    app.run()

    assert not app.exception
    assert any(item.value == "选择最值得投入的岗位" for item in app.title)
    assert len([item for item in app.button if item.label == "预览"]) == 1
    assert any(item.label == "已选择" for item in app.button)
    assert {item.label for item in app.download_button} == {
        "下载对比 Word",
        "下载对比 PDF",
        "导出脱敏档案",
    }
    next(item for item in app.button if item.label == "预览").click()
    app.run()

    assert app.session_state["active_job_id"] is None
    assert app.session_state["comparison_selected_job_id"] == "job_2"
    next(item for item in app.button if item.label == "进入补充确认").click()
    app.run()

    assert any(item.value == "第 3 步：补充真实信息" for item in app.subheader)
    assert len(app.session_state["job_analyses"]) == 2


def test_workspace_add_job_dialog_opens_without_changing_active_job() -> None:
    candidate, bundle = _clarification_state()
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30)
    app.session_state["candidate_profile"] = candidate
    app.session_state["job_analyses"] = {"job_1": bundle}
    app.session_state["active_job_id"] = None

    app.run()
    _button(app, "添加岗位 JD").click()
    app.run()

    assert not app.exception
    assert app.session_state["active_job_id"] is None
    assert any(
        item.label == "本次添加岗位数量" for item in app.number_input
    )
    assert any(item.label == "分析并加入对比" for item in app.button)


def test_final_detail_can_return_to_multi_job_workspace() -> None:
    candidate, bundle = _clarification_state()
    preliminary = PreliminaryAnalysis.model_validate(bundle["preliminary_analysis"])
    bundle.update(
        {
            "stage": "final",
            "final_analysis": MatchAnalysis(matches=preliminary.matches).model_dump(mode="json"),
        }
    )
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30)
    app.session_state["candidate_profile"] = candidate
    app.session_state["job_analyses"] = {"job_1": bundle}
    app.session_state["active_job_id"] = "job_1"
    app.session_state["ui_theme"] = "dark"
    app.session_state["ui_theme_dark"] = True

    app.run()
    next(item for item in app.button if item.label == "返回岗位对比").click()
    app.run()

    assert not app.exception
    assert any(item.value == "选择最值得投入的岗位" for item in app.title)
    assert app.session_state["active_job_id"] is None
    assert app.session_state["ui_theme"] == "dark"
    assert app.session_state["ui_theme_dark"] is True


def test_restart_clears_analysis_but_preserves_dark_theme() -> None:
    candidate, bundle = _clarification_state()
    preliminary = PreliminaryAnalysis.model_validate(bundle["preliminary_analysis"])
    bundle.update(
        {
            "stage": "final",
            "final_analysis": MatchAnalysis(
                matches=preliminary.matches
            ).model_dump(mode="json"),
        }
    )
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30)
    app.session_state["candidate_profile"] = candidate
    app.session_state["job_analyses"] = {"job_1": bundle}
    app.session_state["active_job_id"] = "job_1"
    app.session_state["ui_theme"] = "dark"
    app.session_state["ui_theme_dark"] = True

    app.run()
    _button(app, "重新开始").click()
    app.run()

    assert not app.exception
    assert "candidate_profile" not in app.session_state
    assert app.session_state["ui_theme"] == "dark"
    assert app.session_state["ui_theme_dark"] is True


def test_new_resume_clears_workspace_but_preserves_dark_theme() -> None:
    candidate, bundle = _clarification_state()
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30)
    app.session_state["candidate_profile"] = candidate
    app.session_state["job_analyses"] = {"job_1": bundle}
    app.session_state["active_job_id"] = None
    app.session_state["ui_theme"] = "dark"
    app.session_state["ui_theme_dark"] = True

    app.run()
    _button(app, "上传新简历并清空当前工作台").click()
    app.run()

    assert not app.exception
    assert "candidate_profile" not in app.session_state
    assert app.session_state["ui_theme"] == "dark"
    assert app.session_state["ui_theme_dark"] is True


def test_final_navigation_stays_on_current_section_after_button_error() -> None:
    candidate, bundle = _clarification_state()
    preliminary = PreliminaryAnalysis.model_validate(bundle["preliminary_analysis"])
    partial_match = preliminary.matches[0].model_copy(update={"status": MatchStatus.partial})
    bundle.update(
        {
            "stage": "final",
            "clarification_answers": [
                {
                    "question_id": "cq_req_001",
                    "requirement_id": "req_001",
                    "status": "have",
                    "evidence_text": "",
                    "metrics": None,
                }
            ],
            "final_analysis": MatchAnalysis(matches=[partial_match]).model_dump(mode="json"),
        }
    )
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30)
    app.session_state["candidate_profile"] = candidate
    app.session_state["job_analyses"] = {"job_1": bundle}
    app.session_state["active_job_id"] = "job_1"

    app.run()
    _button(app, "简历").click()
    app.run()
    next(item for item in app.button if item.label == "AI 批量优化已填写内容").click()
    app.run()

    assert not app.exception
    assert app.session_state["workspace_section"] == "简历"
    assert app.session_state["detail_section_job_1_简历"] == "简历优化"
    assert any("请至少填写一项真实经历" in item.value for item in app.error)
