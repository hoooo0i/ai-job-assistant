from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from src.privacy import redact_sensitive_info
from src.schemas import (
    ApplicationRecord,
    CandidateFact,
    ClarificationAnswer,
    ClarificationQuestion,
    JobProfile,
    MatchAnalysis,
    PdfLayoutSignals,
    PreliminaryAnalysis,
    ResumeProfile,
    ResumeVersion,
)


ARCHIVE_VERSION = "1.0"
MAX_ARCHIVE_BYTES = 5 * 1024 * 1024
MAX_ARCHIVE_JOBS = 5
_RUNTIME_ONLY_KEYS = {
    "docx",
    "pdf",
    "report_files",
    "tailored_resume_file",
    "comparison_report_files",
    "application_package",
}


class WorkspaceArchiveError(ValueError):
    """Raised when an exported workspace archive is unsafe or invalid."""


def _sanitise_json(value: Any, *, parent_key: str = "") -> Any:
    if parent_key in _RUNTIME_ONLY_KEYS:
        return None
    if isinstance(value, bytes):
        return None
    if isinstance(value, str):
        return redact_sensitive_info(value)
    if isinstance(value, list):
        return [
            _sanitise_json(item)
            for item in value
            if not isinstance(item, bytes)
        ]
    if isinstance(value, tuple):
        return [_sanitise_json(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _sanitise_json(item, parent_key=str(key))
            for key, item in value.items()
            if str(key) not in _RUNTIME_ONLY_KEYS and not isinstance(item, bytes)
        }
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise WorkspaceArchiveError("档案中包含不支持的数据类型。")


def _validate_candidate(candidate: dict) -> dict:
    required = {"resume_id", "resume_profile", "resume_text", "resume_source"}
    if not required.issubset(candidate):
        raise WorkspaceArchiveError("档案缺少候选人基础信息。")
    candidate["resume_profile"] = ResumeProfile.model_validate(
        candidate["resume_profile"]
    ).model_dump(mode="json")
    candidate["facts"] = [
        CandidateFact.model_validate(item).model_dump(mode="json")
        for item in candidate.get("facts", [])
    ]
    if candidate.get("pdf_layout") is not None:
        candidate["pdf_layout"] = PdfLayoutSignals.model_validate(
            candidate["pdf_layout"]
        ).model_dump(mode="json")
    candidate["filename"] = "已导入档案简历.pdf"
    candidate["resume_text"] = redact_sensitive_info(str(candidate["resume_text"]))
    return candidate


def _validate_job_bundle(job_id: str, bundle: dict) -> dict:
    if bundle.get("job_id") != job_id:
        raise WorkspaceArchiveError("档案中的岗位标识不一致。")
    bundle["job_profile"] = JobProfile.model_validate(
        bundle["job_profile"]
    ).model_dump(mode="json")
    bundle["preliminary_analysis"] = PreliminaryAnalysis.model_validate(
        bundle["preliminary_analysis"]
    ).model_dump(mode="json")
    bundle["clarification_questions"] = [
        ClarificationQuestion.model_validate(item).model_dump(mode="json")
        for item in bundle.get("clarification_questions", [])
    ]
    bundle["clarification_answers"] = [
        ClarificationAnswer.model_validate(item).model_dump(mode="json")
        for item in bundle.get("clarification_answers", [])
    ]
    if bundle.get("final_analysis") is not None:
        bundle["final_analysis"] = MatchAnalysis.model_validate(
            bundle["final_analysis"]
        ).model_dump(mode="json")
    if bundle.get("resume_versions") is not None:
        bundle["resume_versions"] = [
            ResumeVersion.model_validate(item).model_dump(mode="json")
            for item in bundle["resume_versions"]
        ]
    if bundle.get("application_tracking") is not None:
        bundle["application_tracking"] = ApplicationRecord.model_validate(
            bundle["application_tracking"]
        ).model_dump(mode="json")
    if bundle.get("stage") not in {"clarification", "final"}:
        raise WorkspaceArchiveError("档案中的岗位阶段无效。")
    return bundle


def _normalise_payload(payload: dict) -> tuple[dict, dict[str, dict]]:
    if payload.get("schema_version") != ARCHIVE_VERSION:
        raise WorkspaceArchiveError("档案版本不受支持，请使用当前版本重新导出。")
    candidate = payload.get("candidate_profile")
    jobs = payload.get("job_analyses")
    if not isinstance(candidate, dict) or not isinstance(jobs, dict):
        raise WorkspaceArchiveError("档案结构不完整。")
    if len(jobs) > MAX_ARCHIVE_JOBS:
        raise WorkspaceArchiveError(f"档案最多包含 {MAX_ARCHIVE_JOBS} 个岗位。")
    cleaned_candidate = _validate_candidate(_sanitise_json(candidate))
    cleaned_jobs = {
        str(job_id): _validate_job_bundle(
            str(job_id),
            _sanitise_json(bundle),
        )
        for job_id, bundle in jobs.items()
        if isinstance(bundle, dict)
    }
    if len(cleaned_jobs) != len(jobs):
        raise WorkspaceArchiveError("档案包含无效的岗位记录。")
    return cleaned_candidate, cleaned_jobs


def build_workspace_archive(
    candidate_profile: dict,
    job_analyses: dict[str, dict],
    *,
    exported_at: str | None = None,
) -> bytes:
    """Export only redacted, JSON-safe session data; never include PDF/file bytes."""
    payload = {
        "schema_version": ARCHIVE_VERSION,
        "exported_at": exported_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "candidate_profile": candidate_profile,
        "job_analyses": job_analyses,
    }
    try:
        candidate, jobs = _normalise_payload(_sanitise_json(payload))
    except (KeyError, TypeError, ValidationError) as exc:
        raise WorkspaceArchiveError("当前会话数据不完整，暂时无法导出档案。") from exc
    payload["candidate_profile"] = candidate
    payload["job_analyses"] = jobs
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def load_workspace_archive(data: bytes) -> tuple[dict, dict[str, dict]]:
    """Validate and load a previously exported redacted workspace archive."""
    if not data:
        raise WorkspaceArchiveError("请选择一个求职档案 JSON 文件。")
    if len(data) > MAX_ARCHIVE_BYTES:
        raise WorkspaceArchiveError("求职档案不能超过 5 MB。")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkspaceArchiveError("无法读取该文件，请上传有效的求职档案 JSON。") from exc
    if not isinstance(payload, dict):
        raise WorkspaceArchiveError("求职档案必须是 JSON 对象。")
    try:
        return _normalise_payload(payload)
    except (KeyError, TypeError, ValidationError) as exc:
        raise WorkspaceArchiveError("求职档案内容不完整或已损坏。") from exc
