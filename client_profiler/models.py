from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class DocumentInput:
    source_path: Path
    source_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentClassification:
    document_kind: str
    is_client_related: bool
    confidence: float
    rationale: str


@dataclass
class ExtractedEvent:
    date: str | None
    title: str
    details: str


@dataclass
class ExtractedInsight:
    key_findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    contacts: list[str] = field(default_factory=list)
    report_type: str | None = None
    authors: list[str] = field(default_factory=list)
    project_areas: list[str] = field(default_factory=list)


@dataclass
class ProjectContext:
    project_name: str | None = None
    project_code: str | None = None
    quote_number: str | None = None
    purchase_order_number: str | None = None
    access_reference: str | None = None
    related_references: list[str] = field(default_factory=list)


@dataclass
class ExtractedProfileData:
    client_name: str | None
    classification: DocumentClassification
    events: list[ExtractedEvent] = field(default_factory=list)
    insight: ExtractedInsight = field(default_factory=ExtractedInsight)
    project_context: ProjectContext = field(default_factory=ProjectContext)
    hierarchy_paths: list[str] = field(default_factory=list)
    additional_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimelineItem:
    client_name: str
    date: str | None
    summary: str
    source_document: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProfileNode:
    path: str
    facts: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=datetime.utcnow)
