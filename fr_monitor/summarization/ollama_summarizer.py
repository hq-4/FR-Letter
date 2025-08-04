"""
Final summarization using Ollama instead of OpenRouter.
"""

import os
import ollama
from typing import List, Optional
import structlog
from pathlib import Path

from ..core.models import ConsolidatedSummary, FinalSummary
from ..core.config import settings

logger = structlog.get_logger(__name__)


class OllamaSummarizer:
    """Final summarization using Ollama Python library."""
    
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None):
        self.host = host or settings.ollama_host
        self.model = model or settings.summary_model
        
        # Configure Ollama client
        self.client = ollama.Client(host=self.host)
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        prompt_path = Path(settings.prompt_dir) / "system_prompt.md"
        try:
            with open(prompt_path, "r") as f:
                return f.read()
        except Exception as e:
            logger.error("Failed to load system prompt", error=str(e))
            return """
            You are a policy analyst specializing in federal regulations. 
            Create a concise 2-3 sentence summary highlighting the key regulatory changes, 
            affected parties, and implementation timeline.
            """
    
    def generate_final_summaries(
        self, 
        consolidated_summaries: List[ConsolidatedSummary]
    ) -> List[FinalSummary]:
        """
        Generate final summaries using Ollama Python library.
        
        Args:
            consolidated_summaries: List of consolidated document summaries
            
        Returns:
            List of final summaries with headlines and bullets
        """
        final_summaries = []
        
        logger.info("Starting final summarization", 
                   documents=len(consolidated_summaries),
                   model=self.model)
        
        for summary in consolidated_summaries:
            try:
                final_summary = self._generate_single_final_summary(summary)
                if final_summary:
                    final_summaries.append(final_summary)
                    logger.info("Generated final summary", 
                               document_id=summary.document_id)
                
            except Exception as e:
                logger.error("Failed to generate final summary", 
                           document_id=summary.document_id,
                           error=str(e))
        
        logger.info("Completed final summarization", 
                   successful=len(final_summaries),
                   total_requested=len(consolidated_summaries))
        
        return final_summaries
    
    def _generate_single_final_summary(self, summary: ConsolidatedSummary) -> Optional[FinalSummary]:
        """Generate a single final summary using Ollama Python library."""
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": summary.summary}
            ]
            
            # Call Ollama API
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.7}
            )
            
            # Extract content
            full_response = response["message"]["content"]
            
            # Create final summary
            return FinalSummary(
                document_id=summary.document_id,
                headline=self._extract_headline(full_response),
                bullets=self._extract_bullets(full_response)
            )
            
        except Exception as e:
            logger.error("Ollama summarization failed", error=str(e))
            return None
    
    def _extract_headline(self, text: str) -> str:
        """Extract headline from summary text."""
        # First sentence is headline
        return text.split('.')[0] + '.'
    
    def _extract_bullets(self, text: str) -> List[str]:
        """Extract bullet points from summary text."""
        # Split by new lines that start with hyphen
        return [line.strip(" -\n") for line in text.split("\n") 
                if line.strip().startswith("-")]
