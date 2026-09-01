from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class StrictSchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EducationItem(StrictSchemaModel):
    institution: str
    degree: Optional[str]
    field_of_study: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    highlights: list[str]


class ExperienceItem(StrictSchemaModel):
    organization: str
    title: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    location: Optional[str]
    bullets: list[str]


class ProjectItem(StrictSchemaModel):
    name: str
    role: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    bullets: list[str]
    technologies: list[str]


class SkillGroup(StrictSchemaModel):
    category: str
    skills: list[str]


class EvidenceChunk(StrictSchemaModel):
    source_section: str
    text: str = Field(description="A short verbatim excerpt from the resume text.")


class ResumeProfile(StrictSchemaModel):
    summary: Optional[str]
    education: list[EducationItem]
    experience: list[ExperienceItem]
    projects: list[ProjectItem]
    skills: list[SkillGroup]
    languages: list[str]
    evidence_chunks: list[EvidenceChunk]


class RequirementCategory(str, Enum):
    skill = "skill"
    experience = "experience"
    education = "education"
    domain = "domain"
    soft_skill = "soft_skill"
    availability = "availability"
    location = "location"
    other = "other"


class RequirementImportance(str, Enum):
    must_have = "must_have"
    preferred = "preferred"
    other = "other"


class JobRequirement(StrictSchemaModel):
    id: str = Field(description="A stable identifier such as req_001.")
    original_text: str = Field(description="A verbatim requirement excerpt from the JD.")
    normalized_name: str
    category: RequirementCategory
    importance: RequirementImportance
    is_hard_condition: bool


class JobProfile(StrictSchemaModel):
    company: str
    title: str
    location: Optional[str]
    job_type: Optional[str]
    responsibilities: list[str]
    requirements: list[JobRequirement]
    domain_background: list[str]


class MatchStatus(str, Enum):
    matched = "matched"
    partial = "partial"
    missing = "missing"
    unknown = "unknown"


class RequirementMatch(StrictSchemaModel):
    requirement_id: str
    status: MatchStatus
    resume_evidence: list[str] = Field(
        description="Short verbatim excerpts copied from the supplied resume text."
    )
    explanation: str
    confidence: float = Field(ge=0, le=1)


class ResumeSuggestion(StrictSchemaModel):
    original_text: str = Field(description="A verbatim excerpt from the supplied resume.")
    suggested_text: str
    requirement_ids: list[str]
    reason: str
    follow_up_question: Optional[str]


class InterviewCategory(str, Enum):
    job_knowledge = "job_knowledge"
    behavioral = "behavioral"
    project_deep_dive = "project_deep_dive"
    capability_gap = "capability_gap"


class InterviewQuestion(StrictSchemaModel):
    category: InterviewCategory
    question: str
    why_asked: str
    answer_outline: list[str]
    requirement_ids: list[str]


class MatchAnalysis(StrictSchemaModel):
    matches: list[RequirementMatch]
    resume_suggestions: list[ResumeSuggestion] = Field(default_factory=list)
    interview_questions: list[InterviewQuestion] = Field(default_factory=list)


class ScoreResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    match_score: Optional[float]
    information_completeness: float
    known_weight: int
    total_weight: int
    calculation_version: str
