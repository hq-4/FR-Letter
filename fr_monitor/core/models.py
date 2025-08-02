"""
Data models for the Federal Register monitoring system.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class DocumentType(str, Enum):
    """Federal Register document types."""
    EXECUTIVE_ORDER = "executive_order"
    FINAL_RULE = "final_rule"
    PROPOSED_RULE = "proposed_rule"
    NOTICE = "notice"
    PRESIDENTIAL_DOCUMENT = "presidential_document"


class Agency(BaseModel):
    """Federal agency information."""
    name: str
    abbreviation: str
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)


class FederalRegisterDocument(BaseModel):
    """Federal Register document model."""
    document_id: str
    title: str
    abstract: Optional[str] = None
    full_text_url: Optional[str] = None
    html_url: Optional[str] = None
    pdf_url: Optional[str] = None
    
    document_type: DocumentType
    agencies: List[Agency]
    publication_date: datetime
    effective_date: Optional[datetime] = None
    
    page_length: Optional[int] = None
    is_final_rule: bool = False
    is_major_rule: bool = False
    
    raw_data: Dict[str, Any] = Field(default_factory=dict)


class ImpactScore(BaseModel):
    """Document impact scoring."""
    document_id: str
    total_score: float = Field(ge=0.0, le=1.0)
    
    agency_score: float = Field(ge=0.0, le=1.0)
    length_score: float = Field(ge=0.0, le=1.0)
    type_score: float = Field(ge=0.0, le=1.0)
    
    scoring_timestamp: datetime = Field(default_factory=datetime.utcnow)


class DocumentEmbedding(BaseModel):
    """Document embedding storage."""
    document_id: str
    embedding: List[float]
    embedding_model: str
    text_source: str  # "title", "abstract", or "full_text"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    """Document chunk for summarization."""
    document_id: str
    chunk_id: str
    content: str
    chunk_index: int
    token_count: Optional[int] = None


class ChunkSummary(BaseModel):
    """Summary of a document chunk."""
    document_id: str
    chunk_id: str
    summary: str
    model_used: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ConsolidatedSummary(BaseModel):
    """Consolidated summary from all chunks."""
    document_id: str
    summary: str
    chunk_count: int
    total_tokens: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FinalSummary(BaseModel):
    """Final LLM-generated summary."""
    document_id: str
    headline: str
    bullets: List[str]
    model_used: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PipelineRun(BaseModel):
    """Pipeline execution tracking."""
    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    
    documents_processed: int = 0
    documents_summarized: int = 0
    openrouter_calls_made: int = 0
    
    error_message: Optional[str] = None
    logs: List[str] = Field(default_factory=list)


class PublishingResult(BaseModel):
    """Result of publishing to external channels."""
    document_id: str
    channel: str  # "substack", "telegram"
    success: bool
    published_at: Optional[datetime] = None
    error_message: Optional[str] = None
    external_id: Optional[str] = None  # ID from external service
