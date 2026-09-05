from datetime import date

import pytest

from src.application_tracker import (
    ApplicationTrackerError,
    build_application_metrics,
    build_application_record,
    upcoming_application_actions,
)


def _bundle(status: str, **dates: str) -> dict:
    return {
        "job_profile": {"company": "示例公司", "title": "数据岗位"},
        "application_tracking": {
            "status": status,
            "applied_on": dates.get("applied_on"),
            "deadline": dates.get("deadline"),
            "interview_on": dates.get("interview_on"),
            "follow_up_on": dates.get("follow_up_on"),
            "job_url": None,
            "notes": "",
            "resume_version_id": None,
        },
    }


def test_application_record_validates_url_dates_and_resume_version() -> None:
    record = build_application_record(
        {
            "status": "applied",
            "applied_on": "2026-09-06",
            "job_url": "https://example.test/jobs/1",
            "resume_version_id": "version_1",
        },
        available_resume_version_ids={"version_1"},
    )
    assert record.status.value == "applied"
    assert record.applied_on == "2026-09-06"

    with pytest.raises(ApplicationTrackerError, match="http"):
        build_application_record({"job_url": "example.test/jobs/1"})
    with pytest.raises(ApplicationTrackerError, match="不存在"):
        build_application_record(
            {"resume_version_id": "missing"},
            available_resume_version_ids={"version_1"},
        )
    with pytest.raises(ApplicationTrackerError, match="有效日期"):
        build_application_record({"deadline": "06/09/2026"})


def test_application_metrics_use_submitted_jobs_as_rate_denominator() -> None:
    jobs = {
        "one": _bundle("preparing"),
        "two": _bundle("applied"),
        "three": _bundle("interview"),
        "four": _bundle("offer"),
    }

    metrics = build_application_metrics(jobs)

    assert metrics.total_jobs == 4
    assert metrics.submitted == 3
    assert metrics.responses == 2
    assert metrics.interviews == 2
    assert metrics.offers == 1
    assert metrics.response_rate == 66.7
    assert metrics.offer_rate == 33.3


def test_upcoming_actions_include_overdue_and_near_term_dates() -> None:
    jobs = {
        "one": _bundle(
            "applied",
            deadline="2026-09-05",
            follow_up_on="2026-09-10",
            interview_on="2026-11-01",
        )
    }

    actions = upcoming_application_actions(
        jobs,
        today=date(2026, 9, 6),
        within_days=30,
    )

    assert [(item["kind"], item["timing"]) for item in actions] == [
        ("申请截止", "已逾期"),
        ("跟进", "即将到期"),
    ]
