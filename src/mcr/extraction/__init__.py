"""Multimodal candidate memory extraction."""

from mcr.extraction.errors import ExtractionError, ExtractionResponseError
from mcr.extraction.extractor import MemoryExtractor
from mcr.extraction.models import (
    CandidateMemory,
    CandidateMetadata,
    ConversationMessage,
    ExtractionRequest,
    ExtractionResult,
    ExtractionTrace,
    ExistingMemoryContext,
    ExtractedSourceReference,
)
from mcr.extraction.candidate_discovery import (
    build_candidate_prompt as build_extraction_prompt,
)

__all__ = [
    "CandidateMemory",
    "CandidateMetadata",
    "ConversationMessage",
    "ExtractionError",
    "ExtractionRequest",
    "ExtractionResponseError",
    "ExtractionResult",
    "ExtractionTrace",
    "ExistingMemoryContext",
    "ExtractedSourceReference",
    "MemoryExtractor",
    "build_extraction_prompt",
]

