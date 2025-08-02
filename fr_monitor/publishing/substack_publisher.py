"""
Substack API integration for publishing Federal Register summaries.
"""

import requests
from typing import List, Optional, Dict, Any
import structlog
from datetime import datetime
import html

from ..core.models import FinalSummary, PublishingResult
from ..core.config import settings

logger = structlog.get_logger(__name__)


class SubstackPublisher:
    """Publisher for Substack newsletter platform."""
    
    def __init__(self, api_key: Optional[str] = None, publication_id: Optional[str] = None):
        self.api_key = api_key or settings.substack_api_key
        self.publication_id = publication_id or settings.substack_publication_id
        self.base_url = "https://substack.com/api/v1"
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
    
    def publish_daily_digest(self, summaries: List[FinalSummary]) -> PublishingResult:
        """
        Publish daily digest of Federal Register summaries to Substack.
        
        Args:
            summaries: List of final summaries to publish
            
        Returns:
            PublishingResult with success status and details
        """
        if not summaries:
            logger.warning("No summaries provided for Substack publishing")
            return PublishingResult(
                document_id="daily_digest",
                channel="substack",
                success=False,
                error_message="No summaries to publish"
            )
        
        try:
            # Generate HTML content
            html_content = self._generate_html_content(summaries)
            
            # Create post title
            today = datetime.now().strftime("%B %d, %Y")
            title = f"Federal Register Daily Brief - {today}"
            
            # Publish to Substack
            result = self._publish_post(title, html_content)
            
            if result["success"]:
                logger.info("Successfully published to Substack", 
                           post_id=result.get("post_id"),
                           summaries_count=len(summaries))
                
                return PublishingResult(
                    document_id="daily_digest",
                    channel="substack",
                    success=True,
                    published_at=datetime.utcnow(),
                    external_id=result.get("post_id")
                )
            else:
                return PublishingResult(
                    document_id="daily_digest",
                    channel="substack",
                    success=False,
                    error_message=result.get("error", "Unknown error")
                )
                
        except Exception as e:
            logger.error("Failed to publish to Substack", error=str(e))
            return PublishingResult(
                document_id="daily_digest",
                channel="substack",
                success=False,
                error_message=str(e)
            )
    
    def _generate_html_content(self, summaries: List[FinalSummary]) -> str:
        """Generate HTML content for the newsletter."""
        today = datetime.now().strftime("%B %d, %Y")
        
        html_parts = [
            f"<h1>Federal Register Daily Brief</h1>",
            f"<p><em>{today}</em></p>",
            f"<p>Today's top regulatory developments from the Federal Register, analyzed for policy professionals.</p>",
            "<hr>"
        ]
        
        for i, summary in enumerate(summaries, 1):
            # Escape HTML in headline and bullets
            headline = html.escape(summary.headline)
            
            html_parts.extend([
                f"<h2>{i}. {headline}</h2>",
                "<ul>"
            ])
            
            for bullet in summary.bullets:
                escaped_bullet = html.escape(bullet)
                html_parts.append(f"<li>{escaped_bullet}</li>")
            
            html_parts.extend([
                "</ul>",
                "<br>"
            ])
        
        # Add footer
        html_parts.extend([
            "<hr>",
            "<p><small>This digest is generated using AI analysis of Federal Register documents. ",
            "For full regulatory text and official details, please consult the ",
            '<a href="https://www.federalregister.gov">Federal Register</a>.</small></p>',
            "<p><small>Questions or feedback? Reply to this email.</small></p>"
        ])
        
        return "\n".join(html_parts)
    
    def _publish_post(self, title: str, content: str) -> Dict[str, Any]:
        """Publish a post to Substack."""
        try:
            payload = {
                "title": title,
                "subtitle": "AI-powered analysis of today's regulatory developments",
                "body": content,
                "type": "newsletter",
                "audience": "everyone",
                "publication_id": self.publication_id,
                "draft": False,  # Set to True for testing
                "send_email": True,
                "send_push": False
            }
            
            response = self.session.post(
                f"{self.base_url}/posts",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 201:
                data = response.json()
                return {
                    "success": True,
                    "post_id": data.get("id"),
                    "url": data.get("canonical_url")
                }
            else:
                logger.error("Substack API error", 
                           status_code=response.status_code,
                           response=response.text[:500])
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}"
                }
                
        except requests.RequestException as e:
            logger.error("Failed to call Substack API", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_draft_post(self, summaries: List[FinalSummary]) -> Optional[str]:
        """Create a draft post for review before publishing."""
        try:
            html_content = self._generate_html_content(summaries)
            today = datetime.now().strftime("%B %d, %Y")
            title = f"Federal Register Daily Brief - {today} (DRAFT)"
            
            payload = {
                "title": title,
                "subtitle": "AI-powered analysis of today's regulatory developments",
                "body": html_content,
                "type": "newsletter",
                "publication_id": self.publication_id,
                "draft": True,
                "send_email": False
            }
            
            response = self.session.post(
                f"{self.base_url}/posts",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 201:
                data = response.json()
                draft_url = data.get("canonical_url")
                logger.info("Created Substack draft", url=draft_url)
                return draft_url
            else:
                logger.error("Failed to create Substack draft", 
                           status_code=response.status_code)
                return None
                
        except Exception as e:
            logger.error("Failed to create Substack draft", error=str(e))
            return None
    
    def health_check(self) -> bool:
        """Check Substack API connectivity and authentication."""
        try:
            # Test API access by getting publication info
            response = self.session.get(
                f"{self.base_url}/publications/{self.publication_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Substack health check passed")
                return True
            else:
                logger.error("Substack health check failed", 
                           status_code=response.status_code)
                return False
                
        except Exception as e:
            logger.error("Substack health check failed", error=str(e))
            return False
