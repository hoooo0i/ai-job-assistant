from src.ui import (
    count_pending_confirmations,
    default_detail_section,
    detail_sections,
    normalise_selected_job_id,
    theme_tokens,
)


def test_theme_tokens_expose_distinct_accessible_surfaces() -> None:
    light = theme_tokens("light")
    dark = theme_tokens("dark")

    assert light["page"] == "#EAEBED"
    assert light["accent"] == "#F2C84B"
    assert dark["page"] == "#101216"
    assert dark["accent"] == "#E0B84E"
    assert light["text"] != dark["text"]


def test_navigation_groups_preserve_every_detailed_feature() -> None:
    assert detail_sections("岗位") == ("岗位匹配", "关键词缺口")
    assert detail_sections("简历")[0] == "简历优化"
    assert "报告下载" in detail_sections("投递")
    assert detail_sections("面试") == ("面试辅助",)
    assert default_detail_section("更多") == "JD 结构"


def test_selected_job_falls_back_to_first_ranked_item() -> None:
    assert normalise_selected_job_id(["job_high", "job_low"], None) == "job_high"
    assert normalise_selected_job_id(["job_high", "job_low"], "job_low") == "job_low"
    assert normalise_selected_job_id([], "missing") is None


def test_pending_confirmation_count_only_includes_unanswered_or_unsure() -> None:
    analyses = [
        {
            "clarification_questions": [
                {"id": "q1"},
                {"id": "q2"},
                {"id": "q3"},
            ],
            "clarification_answers": [
                {"question_id": "q1", "status": "have"},
                {"question_id": "q2", "status": "unsure"},
            ],
        },
        {
            "clarification_questions": [{"id": "q4"}],
            "clarification_answers": [
                {"question_id": "q4", "status": "not_have"}
            ],
        },
    ]

    assert count_pending_confirmations(analyses) == 2
