"""Data model dataclasses for the AD Knowledge Graph pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any


@dataclass
class Paper:
    pmid: str
    title: str
    abstract: str
    pub_date: str  # ISO date string or partial date
    authors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Paper":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Trial:
    nct_id: str
    title: str
    status: str
    phase: str
    conditions: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trial":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class EntityMention:
    text: str
    label: str
    start: int
    end: int
    paper_id: str
    canonical_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EntityMention":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class GWASHit:
    snp_id: str
    gene: str
    trait: str
    p_value: float
    odds_ratio: float
    study_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GWASHit":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class FAERSReport:
    drug_name: str
    reaction: str
    ror: float
    ci_lower: float
    ci_upper: float
    report_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FAERSReport":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
