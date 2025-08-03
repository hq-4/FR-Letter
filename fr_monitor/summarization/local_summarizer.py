"""
Local summarization using Ollama for document chunks.
"""

import requests
from typing import List, Optional
import structlog
from datetime import datetime

from ..core.models import DocumentChunk, ChunkSummary, ConsolidatedSummary
from ..core.config import settings

logger = structlog.get_logger(__name__)


class LocalSummarizer:
    """Local document summarization using Ollama."""
    
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None):
        self.host = host or settings.ollama_host
        self.model = model or settings.summary_model
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
        
        # Summarization prompt template
        self.chunk_prompt_template = """
You are a policy analyst specializing in federal regulations. Summarize the following section from a Federal Register document in 50-100 words. Focus on:

1. Key regulatory changes or requirements
2. Affected parties or industries
3. Implementation dates or deadlines
4. Economic or operational impacts

Be concise and factual. Avoid speculation.

Document section:
{content}

Summary:"""
    
    def summarize_chunks(self, chunks: List[DocumentChunk]) -> List[ChunkSummary]:
        """
        Summarize multiple document chunks.
        
        Args:
            chunks: List of document chunks to summarize
            
        Returns:
            List of chunk summaries
        """
        summaries = []
        
        logger.info("Starting chunk summarization", 
                   chunk_count=len(chunks),
                   model=self.model)
        
        for chunk in chunks:
            try:
                summary_text = self._summarize_single_chunk(chunk.content)
                if summary_text:
                    summary = ChunkSummary(
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        summary=summary_text,
                        model_used=self.model
                    )
                    summaries.append(summary)
                    
                    logger.debug("Summarized chunk", 
                               chunk_id=chunk.chunk_id,
                               original_length=len(chunk.content),
                               summary_length=len(summary_text))
                
            except Exception as e:
                logger.error("Failed to summarize chunk", 
                           chunk_id=chunk.chunk_id,
                           error=str(e))
        
        logger.info("Completed chunk summarization", 
                   successful=len(summaries),
                   total=len(chunks))
        
        return summaries
    
    def consolidate_summaries(self, summaries: List[ChunkSummary]) -> List[ConsolidatedSummary]:
        """
        Consolidate chunk summaries by document.
        
        Args:
            summaries: List of chunk summaries
            
        Returns:
            List of consolidated document summaries
        """
        # Group summaries by document
        doc_summaries = {}
        for summary in summaries:
            if summary.document_id not in doc_summaries:
                doc_summaries[summary.document_id] = []
            doc_summaries[summary.document_id].append(summary)
        
        consolidated = []
        
        for document_id, doc_chunk_summaries in doc_summaries.items():
            try:
                # Sort summaries by chunk order (if available)
                # Note: ChunkSummary has chunk_id, but ConsolidatedSummary doesn't
                # We're working with ChunkSummary objects here, not ConsolidatedSummary
                doc_chunk_summaries.sort(key=lambda x: getattr(x, 'chunk_id', 0))
                
                # Combine all chunk summaries
                combined_text = "\n\n".join([s.summary for s in doc_chunk_summaries])
                
                # Create consolidated summary
                consolidated_summary = ConsolidatedSummary(
                    document_id=document_id,
                    summary=combined_text,
                    chunk_count=len(doc_chunk_summaries),
                    total_tokens=self._estimate_tokens(combined_text)
                )
                
                consolidated.append(consolidated_summary)
                
                logger.info(f"Consolidated document summaries for {document_id} with {len(doc_chunk_summaries)} chunks, total length {len(combined_text)}")
                
            except Exception as e:
                logger.error(f"Failed to consolidate summaries for {document_id}: {e}")
        
        logger.info("Completed summary consolidation", 
                   documents=len(consolidated))
        
        return consolidated
    
    def _summarize_single_chunk(self, content: str) -> Optional[str]:
        """Summarize a single chunk of content."""
        if not content.strip():
            return None
        
        try:
            # Prepare prompt
            prompt = self.chunk_prompt_template.format(content=content)
            
            # Call Ollama API
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more focused summaries
                    "top_p": 0.9,
                    "max_tokens": 150,   # Limit summary length
                    "stop": ["\n\n", "Document section:", "Summary:"]
                }
            }
            
            response = self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=120  # Increased timeout for summarization
            )
            response.raise_for_status()
            
            data = response.json()
            summary = data.get("response", "").strip()
            
            if summary:
                # Clean up the summary
                summary = self._clean_summary(summary)
                return summary
            else:
                logger.warning("Empty summary response from Ollama")
                return None
                
        except requests.RequestException as e:
            logger.error("Failed to call Ollama for summarization", 
                        error=str(e),
                        host=self.host,
                        model=self.model)
            return None
    
    def _clean_summary(self, summary: str) -> str:
        """Clean and format summary text."""
        # Remove common artifacts
        summary = summary.replace("Summary:", "").strip()
        summary = summary.replace("Here is a summary:", "").strip()
        summary = summary.replace("This section", "The section").strip()
        
        # Ensure it ends with proper punctuation
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        return summary
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimation."""
        return len(text) // 4
    
    def health_check(self) -> bool:
        """Check if Ollama service is available for summarization."""
        try:
            # Test with a simple prompt
            test_payload = {
                "model": self.model,
                "prompt": "Test prompt for health check.",
                "stream": False,
                "options": {"max_tokens": 10}
            }
            
            response = self.session.post(
                f"{self.host}/api/generate",
                json=test_payload,
                timeout=30  # Increased timeout for health check
            )
            response.raise_for_status()
            
            data = response.json()
            if "response" in data:
                logger.info("Local summarizer health check passed", 
                           host=self.host,
                           model=self.model)
                return True
            else:
                logger.warning("Unexpected response format from Ollama")
                return False
                
        except Exception as e:
            logger.error("Local summarizer health check failed", 
                        host=self.host,
                        model=self.model,
                        error=str(e))
            return False
