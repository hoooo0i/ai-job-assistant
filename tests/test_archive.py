import json

import pytest

from src.archive import (
    WorkspaceArchiveError,
    build_workspace_archive,
    load_workspace_archive,
)
from src.schemas import (
    JobProfile,
    MatchAnalysis,
    PreliminaryAnalysis,
    ResumeProfile,
)


def _workspace() -> tuple[dict, dict[str, dict]]:
    resume = ResumeProfile(
        summary="Candidate candidate@example.test +61 412 345 678",
        education=[],
        experience=[],
        projects=[],
        skills=[],
        languages=[],
        evidence_chunks=[],
    )
    job = JobProfile(
        company="示例公司",
        title="数据岗位",
        location="墨尔本",
        job_type="全职",
        responsibilities=[],
        requirements=[],
        domain_background=[],
    )
    preliminary = PreliminaryAnalysis(matches=[], clarification_questions=[])
    final = MatchAnalysis(matches=[])
    candidate = {
        "resume_id": "resume_1",
        "resume_profile": resume.model_dump(mode="json"),
        "resume_text": (
            "Contact candidate@example.test\nPhone +61 412 345 678\n"
            "Address: 12 Smith Street Melbourne"
        ),
        "resume_source": "pdf",
        "filename": "candidate@example.test.pdf",
        "page_count": 1,
        "facts": [],
    }
    jobs = {
        "job_1": {
            "job_id": "job_1",
            "fingerprint": "job_1",
            "job_profile": job.model_dump(mode="json"),
            "preliminary_analysis": preliminary.model_dump(mode="json"),
            "clarification_questions": [],
            "clarification_answers": [],
            "stage": "final",
            "final_analysis": final.model_dump(mode="json"),
            "report_files": {"docx": b"private", "pdf": b"private"},
            "tailored_resume_file": {"docx": b"private"},
            "application_tracking": {
                "status": "applied",
                "applied_on": "2026-09-05",
                "deadline": None,
                "interview_on": None,
                "follow_up_on": "2026-09-10",
                "job_url": "https://example.test/job/1",
                "notes": "Follow up next week",
                "resume_version_id": None,
            },
        }
    }
    return candidate, jobs


def test_archive_is_redacted_json_and_excludes_binary_caches() -> None:
    candidate, jobs = _workspace()

    archive = build_workspace_archive(
        candidate,
        jobs,
        exported_at="2026-09-05T00:00:00+00:00",
    )
    text = archive.decode("utf-8")
    payload = json.loads(text)

    assert payload["schema_version"] == "1.0"
    assert "candidate@example.test" not in text
    assert "+61 412 345 678" not in text
    assert "12 Smith Street" not in text
    assert "report_files" not in text
    assert "tailored_resume_file" not in text
    assert payload["candidate_profile"]["filename"] == "已导入档案简历.pdf"


def test_archive_round_trip_restores_candidate_and_jobs() -> None:
    candidate, jobs = _workspace()
    archive = build_workspace_archive(candidate, jobs)

    restored_candidate, restored_jobs = load_workspace_archive(archive)

    assert restored_candidate["resume_id"] == "resume_1"
    assert list(restored_jobs) == ["job_1"]
    assert restored_jobs["job_1"]["stage"] == "final"
    assert restored_jobs["job_1"]["application_tracking"]["status"] == "applied"


def test_archive_rejects_invalid_version_and_mismatched_job_id() -> None:
    candidate, jobs = _workspace()
    payload = json.loads(build_workspace_archive(candidate, jobs))
    payload["schema_version"] = "9.9"
    with pytest.raises(WorkspaceArchiveError, match="版本"):
        load_workspace_archive(json.dumps(payload).encode("utf-8"))

    payload["schema_version"] = "1.0"
    payload["job_analyses"]["job_1"]["job_id"] = "other"
    with pytest.raises(WorkspaceArchiveError, match="标识"):
        load_workspace_archive(json.dumps(payload).encode("utf-8"))
