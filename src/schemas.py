from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

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


class EvidenceSource(str, Enum):
    resume = "resume"
    user_confirmed = "user_confirmed"


class MatchEvidence(StrictSchemaModel):
    source: EvidenceSource
    text: str
    fact_id: Optional[str] = None


class RequirementMatch(StrictSchemaModel):
    requirement_id: str
    status: MatchStatus
    resume_evidence: list[str] = Field(
        description="Short verbatim excerpts copied from the supplied resume text."
    )
    explanation: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[MatchEvidence] = Field(default_factory=list)


class ClarificationQuestion(StrictSchemaModel):
    id: str
    requirement_id: str
    prompt: str


class PreliminaryAnalysis(StrictSchemaModel):
    matches: list[RequirementMatch]
    clarification_questions: list[ClarificationQuestion]
    resume_suggestions: list[ResumeSuggestion] = Field(default_factory=list)
    interview_questions: list[InterviewQuestion] = Field(default_factory=list)


class CandidateFact(StrictSchemaModel):
    id: str
    category: RequirementCategory
    statement: str
    metrics: Optional[str]
    source_job_id: str
    source_requirement_text: str
    user_confirmed: bool = True


class ClarificationAnswer(StrictSchemaModel):
    question_id: str
    requirement_id: str
    status: Literal["unanswered", "have", "not_have", "unsure"]
    evidence_text: str = ""
    metrics: Optional[str] = None


class SupplementDetail(StrictSchemaModel):
    requirement_id: str
    situation: str = ""
    action: str = ""
    result: str = ""
    metrics: Optional[str] = None


class ResumeSuggestion(StrictSchemaModel):
    original_text: str = Field(description="A verbatim excerpt from the supplied resume.")
    suggested_text: str
    requirement_ids: list[str]
    reason: str
    follow_up_question: Optional[str]


class SupplementRewriteResult(StrictSchemaModel):
    suggestions: list[ResumeSuggestion]


class PdfLayoutSignals(StrictSchemaModel):
    page_count: int = 0
    text_block_count: int = 0
    column_page_count: int = 0
    table_count: int = 0
    image_count: int = 0
    drawing_count: int = 0
    minimum_font_size: Optional[float] = None
    has_contact_details: Optional[bool] = None
    readable: bool = True


class AtsCheck(StrictSchemaModel):
    code: str
    severity: Literal["critical", "warning", "passed"]
    title: str
    detail: str
    recommendation: Optional[str] = None


class AtsReport(StrictSchemaModel):
    score: int = Field(ge=0, le=100)
    checks: list[AtsCheck]
    keyword_coverage: float = Field(ge=0, le=100)
    layout: PdfLayoutSignals


class SubmissionCheckItem(StrictSchemaModel):
    code: str
    passed: bool
    blocking: bool
    label: str
    detail: str


class SubmissionChecklist(StrictSchemaModel):
    ready: bool
    items: list[SubmissionCheckItem]


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


class StarOutline(StrictSchemaModel):
    situation: Optional[str]
    task: Optional[str]
    action: Optional[str]
    result: Optional[str]


class InterviewPreparation(StrictSchemaModel):
    personalized_answer: Optional[str]
    key_points: list[str]
    star_outline: StarOutline
    evidence_ids: list[str]
    missing_information: list[str]
    caution_notes: list[str]


class InterviewFeedback(StrictSchemaModel):
    completeness_score: int = Field(ge=0, le=5)
    star_score: int = Field(ge=0, le=5)
    relevance_score: int = Field(ge=0, le=5)
    clarity_score: int = Field(ge=0, le=5)
    strengths: list[str]
    improvements: list[str]
    unsupported_claims: list[str]
    improved_structure: list[str]
    follow_up_question: Optional[str]


class CoverLetterParagraph(StrictSchemaModel):
    text: str
    evidence_ids: list[str]


class CoverLetterDraft(StrictSchemaModel):
    language: Literal["zh", "en"]
    salutation: str
    paragraphs: list[CoverLetterParagraph]
    closing: str
    caution_notes: list[str]


class ScoreResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    match_score: Optional[float]
    information_completeness: float
    known_weight: int
    total_weight: int
    calculation_version: str


class JobComparisonItem(StrictSchemaModel):
    job_id: str
    company: str
    title: str
    stage: Literal["clarification", "final"]
    match_score: Optional[float]
    information_completeness: float = Field(ge=0, le=100)
    ats_score: int = Field(ge=0, le=100)
    hard_risks: int = Field(ge=0)
    must_have_gaps: int = Field(ge=0)
    recommendation_score: float = Field(ge=0)
    application_status: str = "not_started"


class ResumeEditDecision(StrictSchemaModel):
    decision: Literal["pending", "accepted", "ignored"]
    text: str


class ResumeVersion(StrictSchemaModel):
    id: str
    label: str = Field(min_length=1, max_length=60)
    created_at: str
    decisions: dict[str, ResumeEditDecision]
    accepted_suggestions: list[tuple[str, str]]


class ApplicationStatus(str, Enum):
    not_started = "not_started"
    preparing = "preparing"
    applied = "applied"
    assessment = "assessment"
    interview = "interview"
    rejected = "rejected"
    offer = "offer"
    withdrawn = "withdrawn"


class ApplicationRecord(StrictSchemaModel):
    status: ApplicationStatus = ApplicationStatus.not_started
    applied_on: Optional[str] = None
    deadline: Optional[str] = None
    interview_on: Optional[str] = None
    follow_up_on: Optional[str] = None
    job_url: Optional[str] = None
    notes: str = Field(default="", max_length=2_000)
    resume_version_id: Optional[str] = None


class ApplicationMetrics(StrictSchemaModel):
    total_jobs: int = Field(ge=0)
    submitted: int = Field(ge=0)
    responses: int = Field(ge=0)
    interviews: int = Field(ge=0)
    offers: int = Field(ge=0)
    response_rate: float = Field(ge=0, le=100)
    interview_rate: float = Field(ge=0, le=100)
    offer_rate: float = Field(ge=0, le=100)


class JobLinkResult(StrictSchemaModel):
    source_url: str
    company: str = ""
    title: str = ""
    location: str = ""
    job_type: str = ""
    description: str
