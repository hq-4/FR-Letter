"""
Document summarization module with chunking and local/remote processing.
"""

from .document_chunker import DocumentChunker
from .local_summarizer import LocalSummarizer
from .openrouter_summarizer import OpenRouterSummarizer

__all__ = ["DocumentChunker", "LocalSummarizer", "OpenRouterSummarizer"]
