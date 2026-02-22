"""Data models for Indexatron."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class LocationInfo(BaseModel):
    """Location information from photo analysis."""

    setting: str = Field(description="General setting (beach, park, home, etc.)")
    type: str = Field(description="Indoor or outdoor")
    specific: Optional[str] = Field(default=None, description="Specific location if identifiable")


class PersonInfo(BaseModel):
    """Information about a person in the photo."""

    description: str = Field(description="Description of the person")
    estimated_age: Optional[str] = Field(default=None, description="Estimated age or age range")
    position: Optional[str] = Field(default=None, description="Position in frame")


class EraEstimate(BaseModel):
    """Estimated era/decade of the photo."""

    decade: str = Field(description="Estimated decade (1980s, 1990s, etc.)")
    confidence: str = Field(description="Confidence level: low, medium, high")
    reasoning: Optional[str] = Field(default=None, description="Why this era was estimated")


class PhotoAnalysis(BaseModel):
    """Complete analysis of a photo."""

    filename: str
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str = "llava:7b"

    description: str = Field(description="Detailed description of the photo")
    location: Optional[LocationInfo] = None
    people: list[PersonInfo] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    era: Optional[EraEstimate] = None
    mood: Optional[str] = Field(default=None, description="Mood/emotion of the scene")
    colors: list[str] = Field(default_factory=list, description="Notable colors")
    objects: list[str] = Field(default_factory=list, description="Objects visible")

    raw_response: Optional[str] = Field(default=None, description="Raw LLM response")


class EmbeddingResult(BaseModel):
    """Embedding result for a photo."""

    filename: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str = "nomic-embed-text"
    dimensions: int = 768
    embedding: list[float]
    source_text: str = Field(description="Text that was embedded")
