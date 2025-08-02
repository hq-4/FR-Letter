"""
OpenRouter API integration for final LLM-driven summaries.
"""

import requests
from typing import List, Optional, Dict, Any
import structlog
from datetime import datetime
import re
from pathlib import Path

from ..core.models import ConsolidatedSummary, FinalSummary
from ..core.config import settings

logger = structlog.get_logger(__name__)


class OpenRouterSummarizer:
    """Final summarization using OpenRouter API with usage tracking."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.openrouter_model
        self.base_url = "https://openrouter.ai/api/v1"
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/federal-register-monitor",
            "X-Title": "Federal Register Monitor"
        })
        
        # Track daily usage
        self.daily_calls_made = 0
        self.max_daily_calls = settings.max_daily_openrouter_calls
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
    
    def generate_final_summaries(
        self, 
        consolidated_summaries: List[ConsolidatedSummary]
    ) -> List[FinalSummary]:
        """
        Generate final summaries using OpenRouter API.
        
        Args:
            consolidated_summaries: List of consolidated document summaries
            
        Returns:
            List of final summaries with headlines and bullets
        """
        if self.daily_calls_made >= self.max_daily_calls:
            logger.warning("Daily OpenRouter call limit reached", 
                          calls_made=self.daily_calls_made,
                          max_calls=self.max_daily_calls)
            return []
        
        final_summaries = []
        
        logger.info("Starting final summarization", 
                   documents=len(consolidated_summaries),
                   model=self.model,
                   calls_remaining=self.max_daily_calls - self.daily_calls_made)
        
        for summary in consolidated_summaries:
            if self.daily_calls_made >= self.max_daily_calls:
                logger.warning("Reached daily call limit during processing")
                break
            
            try:
                final_summary = self._generate_single_final_summary(summary)
                if final_summary:
                    final_summaries.append(final_summary)
                    self.daily_calls_made += 1
                    
                    logger.info("Generated final summary", 
                               document_id=summary.document_id,
                               calls_made=self.daily_calls_made)
                
            except Exception as e:
                logger.error("Failed to generate final summary", 
                           document_id=summary.document_id,
                           error=str(e))
        
        logger.info("Completed final summarization", 
                   successful=len(final_summaries),
                   total_requested=len(consolidated_summaries),
                   api_calls_made=self.daily_calls_made)
        
        return final_summaries
    
    def _generate_single_final_summary(self, summary: ConsolidatedSummary) -> Optional[FinalSummary]:
        """Generate a single final summary using OpenRouter."""
        try:
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": f"Document Summary:\n\n{summary.summary}"
                }
            ]
            
            # API payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 300,
                "top_p": 0.9
            }
            
            # Make API call
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "choices" not in data or not data["choices"]:
                logger.error("Invalid response format from OpenRouter", response_data=data)
                return None
            
            content = data["choices"][0]["message"]["content"]
            
            # Parse the structured response
            headline, bullets = self._parse_response(content)
            
            if headline and bullets:
                return FinalSummary(
                    document_id=summary.document_id,
                    headline=headline,
                    bullets=bullets,
                    model_used=self.model
                )
            else:
                logger.warning("Failed to parse structured response", 
                             document_id=summary.document_id,
                             content=content[:200])
                return None
                
        except requests.RequestException as e:
            logger.error("OpenRouter API call failed", 
                        document_id=summary.document_id,
                        error=str(e))
            return None
    
    def _parse_response(self, content: str) -> tuple[Optional[str], List[str]]:
        """Parse the structured response from the LLM."""
        try:
            # Look for headline pattern
            headline_match = re.search(r'headline:\s*["\']?([^"\'\n]+)["\']?', content, re.IGNORECASE)
            headline = headline_match.group(1).strip() if headline_match else None
            
            # Look for bullets section
            bullets = []
            bullets_section = re.search(r'bullets:\s*\n(.*?)(?:\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
            
            if bullets_section:
                bullets_text = bullets_section.group(1)
                # Extract bullet points
                bullet_matches = re.findall(r'^\s*[-•*]\s*(.+)$', bullets_text, re.MULTILINE)
                bullets = [bullet.strip() for bullet in bullet_matches if bullet.strip()]
            
            # Fallback: try to extract bullets from anywhere in the response
            if not bullets:
                bullet_matches = re.findall(r'^\s*[-•*]\s*(.+)$', content, re.MULTILINE)
                bullets = [bullet.strip() for bullet in bullet_matches if bullet.strip()]
            
            # If no structured format found, try to extract from plain text
            if not headline and not bullets:
                lines = content.strip().split('\n')
                if lines:
                    headline = lines[0].strip()
                    bullets = [line.strip() for line in lines[1:] if line.strip()]
            
            return headline, bullets[:5]  # Limit to 5 bullets max
            
        except Exception as e:
            logger.error("Failed to parse LLM response", error=str(e))
            return None, []
    
    def _load_system_prompt(self) -> str:
        """Load the Politico-style system prompt from file."""
        try:
            prompt_path = settings.config_dir / "prompts" / "politico_style.txt"
            
            if prompt_path.exists():
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            else:
                # Return default prompt if file doesn't exist
                return self._get_default_system_prompt()
                
        except Exception as e:
            logger.warning("Failed to load system prompt from file", error=str(e))
            return self._get_default_system_prompt()
    
    def _get_default_system_prompt(self) -> str:
        """Get default Politico-style system prompt."""
        return """You are a seasoned policy reporter writing for a sophisticated audience of government affairs professionals, lobbyists, and policy experts.

Your task is to transform regulatory summaries into punchy, insider-focused news items in the style of Politico's policy newsletters.

Requirements:
1. Write a compelling headline that captures the regulatory impact and political implications
2. Create 3-5 bullet points that highlight:
   - Key stakeholders affected
   - Implementation timeline and deadlines  
   - Political/economic implications
   - Industry reactions or expected pushback
   - Connection to broader policy trends

Style guidelines:
- Use active voice and present tense
- Include specific dollar amounts, dates, and affected parties when available
- Adopt a slightly irreverent, insider tone
- Focus on "what this means" rather than just "what happened"
- Keep bullets concise but substantive (15-25 words each)

Format your response exactly as:
headline: "<your headline>"
bullets:
  - <bullet point 1>
  - <bullet point 2>
  - <bullet point 3>
  - <bullet point 4>
  - <bullet point 5>"""
    
    def reset_daily_usage(self) -> None:
        """Reset daily usage counter (typically called at start of new day)."""
        self.daily_calls_made = 0
        logger.info("Reset daily OpenRouter usage counter")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            "daily_calls_made": self.daily_calls_made,
            "max_daily_calls": self.max_daily_calls,
            "calls_remaining": self.max_daily_calls - self.daily_calls_made,
            "model": self.model
        }
    
    def health_check(self) -> bool:
        """Check OpenRouter API connectivity and authentication."""
        try:
            # Make a minimal test call
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5
            }
            
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("OpenRouter health check passed", model=self.model)
                return True
            else:
                logger.error("OpenRouter health check failed", 
                           status_code=response.status_code,
                           response=response.text[:200])
                return False
                
        except Exception as e:
            logger.error("OpenRouter health check failed", error=str(e))
            return False
