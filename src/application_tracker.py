from __future__ import annotations

from datetime import date, timedelta

from src.schemas import ApplicationMetrics, ApplicationRecord, ApplicationStatus


STATUS_LABELS = {
    ApplicationStatus.not_started: "尚未开始",
    ApplicationStatus.preparing: "准备材料",
    ApplicationStatus.applied: "已投递",
    ApplicationStatus.assessment: "笔试或测评",
    ApplicationStatus.interview: "面试中",
    ApplicationStatus.rejected: "未通过",
    ApplicationStatus.offer: "已获 Offer",
    ApplicationStatus.withdrawn: "已放弃",
}


class ApplicationTrackerError(ValueError):
    """Raised when an application tracking record is invalid."""


def _normalise_date(value: str | None, label: str) -> str | None:
    if value is None or not str(value).strip():
        return None
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise ApplicationTrackerError(f"{label}不是有效日期。") from exc


def build_application_record(
    payload: dict,
    *,
    available_resume_version_ids: set[str] | None = None,
) -> ApplicationRecord:
    """Validate a locally edited tracking record and its resume-version link."""
    try:
        status = ApplicationStatus(payload.get("status", "not_started"))
    except ValueError as exc:
        raise ApplicationTrackerError("投递状态无效。") from exc
    job_url = str(payload.get("job_url") or "").strip() or None
    if job_url and not job_url.startswith(("https://", "http://")):
        raise ApplicationTrackerError("岗位链接必须以 http:// 或 https:// 开头。")
    if job_url and len(job_url) > 2_000:
        raise ApplicationTrackerError("岗位链接过长。")
    notes = str(payload.get("notes") or "").strip()
    if len(notes) > 2_000:
        raise ApplicationTrackerError("投递备注不能超过 2000 个字符。")
    resume_version_id = str(payload.get("resume_version_id") or "").strip() or None
    if (
        resume_version_id
        and available_resume_version_ids is not None
        and resume_version_id not in available_resume_version_ids
    ):
        raise ApplicationTrackerError("绑定的简历版本不存在，请重新选择。")
    return ApplicationRecord(
        status=status,
        applied_on=_normalise_date(payload.get("applied_on"), "投递日期"),
        deadline=_normalise_date(payload.get("deadline"), "申请截止日期"),
        interview_on=_normalise_date(payload.get("interview_on"), "面试日期"),
        follow_up_on=_normalise_date(payload.get("follow_up_on"), "跟进日期"),
        job_url=job_url,
        notes=notes,
        resume_version_id=resume_version_id,
    )


def get_application_record(bundle: dict) -> ApplicationRecord:
    return ApplicationRecord.model_validate(bundle.get("application_tracking", {}))


def build_application_metrics(job_analyses: dict[str, dict]) -> ApplicationMetrics:
    records = [get_application_record(bundle) for bundle in job_analyses.values()]
    submitted_statuses = {
        ApplicationStatus.applied,
        ApplicationStatus.assessment,
        ApplicationStatus.interview,
        ApplicationStatus.rejected,
        ApplicationStatus.offer,
    }
    response_statuses = {
        ApplicationStatus.assessment,
        ApplicationStatus.interview,
        ApplicationStatus.rejected,
        ApplicationStatus.offer,
    }
    interview_statuses = {ApplicationStatus.interview, ApplicationStatus.offer}
    submitted = sum(record.status in submitted_statuses for record in records)
    responses = sum(record.status in response_statuses for record in records)
    interviews = sum(record.status in interview_statuses for record in records)
    offers = sum(record.status is ApplicationStatus.offer for record in records)

    def rate(count: int) -> float:
        return round(count / submitted * 100, 1) if submitted else 0.0

    return ApplicationMetrics(
        total_jobs=len(records),
        submitted=submitted,
        responses=responses,
        interviews=interviews,
        offers=offers,
        response_rate=rate(responses),
        interview_rate=rate(interviews),
        offer_rate=rate(offers),
    )


def upcoming_application_actions(
    job_analyses: dict[str, dict],
    *,
    today: date | None = None,
    within_days: int = 30,
) -> list[dict[str, str]]:
    """Return overdue and near-term deadlines, interviews and follow-ups."""
    current_date = today or date.today()
    latest_date = current_date + timedelta(days=within_days)
    labels = {
        "deadline": "申请截止",
        "interview_on": "面试",
        "follow_up_on": "跟进",
    }
    actions: list[dict[str, str]] = []
    for job_id, bundle in job_analyses.items():
        record = get_application_record(bundle)
        job = bundle.get("job_profile", {})
        for field, label in labels.items():
            value = getattr(record, field)
            if not value:
                continue
            action_date = date.fromisoformat(value)
            if action_date > latest_date:
                continue
            actions.append(
                {
                    "job_id": job_id,
                    "company": str(job.get("company", "未命名公司")),
                    "title": str(job.get("title", "未命名岗位")),
                    "kind": label,
                    "date": value,
                    "timing": "已逾期" if action_date < current_date else "即将到期",
                }
            )
    return sorted(actions, key=lambda item: (item["date"], item["company"], item["title"]))
