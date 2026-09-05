import pytest

from src.resume_versions import (
    ResumeVersionError,
    add_resume_version,
    create_resume_version,
    restore_resume_decisions,
)


def test_resume_version_can_be_saved_and_restored() -> None:
    decisions = {
        "0": {"decision": "accepted", "text": "Improved evidence statement."},
        "1": {"decision": "ignored", "text": "Unused."},
    }
    version = create_resume_version(
        "job_1",
        "投递版",
        decisions,
        [("Original", "Improved evidence statement.")],
        created_at="2026-09-05T00:00:00+00:00",
    )

    versions = add_resume_version([], version)

    assert versions[0]["label"] == "投递版"
    assert restore_resume_decisions(versions[0]) == decisions


def test_resume_version_rejects_placeholders_and_enforces_limit() -> None:
    decisions = {"0": {"decision": "accepted", "text": "完成了[请填写数据]"}}
    with pytest.raises(ResumeVersionError, match="占位符"):
        create_resume_version(
            "job_1",
            "不完整版本",
            decisions,
            [("Original", "完成了[请填写数据]")],
        )

    valid = create_resume_version(
        "job_1",
        "版本一",
        {"0": {"decision": "accepted", "text": "Verified fact."}},
        [("Original", "Verified fact.")],
        created_at="2026-09-05T00:00:00+00:00",
    )
    with pytest.raises(ResumeVersionError, match="最多保留"):
        add_resume_version([valid.model_dump(mode="json")], valid, maximum=1)
