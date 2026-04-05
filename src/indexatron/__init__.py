"""Indexatron - AI-powered family photo analysis using local LLMs."""

from .analyzer import PhotoAnalyzer
from .api_client import ApiError, McculloghsClient
from .config import Settings, get_settings
from .embedder import TextEmbedder
from .service import IndexatronService

__all__ = [
    "PhotoAnalyzer",
    "TextEmbedder",
    "IndexatronService",
    "McculloghsClient",
    "ApiError",
    "Settings",
    "get_settings",
]

__version__ = "0.2.0"
