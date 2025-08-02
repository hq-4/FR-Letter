"""Markdown file publisher for Federal Register articles."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

from ..core.models import ProcessedArticle

logger = structlog.get_logger(__name__)


class MarkdownPublisher:
    """Publisher that outputs articles as markdown files."""
    
    def __init__(self, output_dir: str = "posts"):
        """Initialize the markdown publisher.
        
        Args:
            output_dir: Directory to save markdown files (default: "posts")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info("Markdown publisher initialized", output_dir=str(self.output_dir))
    
    def health_check(self) -> bool:
        """Check if the output directory is writable."""
        try:
            # Test write access
            test_file = self.output_dir / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
            logger.info("Markdown publisher health check passed")
            return True
        except Exception as e:
            logger.error("Markdown publisher health check failed", error=str(e))
            return False
    
    def publish(self, article: ProcessedArticle) -> bool:
        """Publish article as a markdown file.
        
        Args:
            article: The processed article to publish
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate filename with current date
            today = datetime.now().strftime("%Y-%m-%d")
            filename = f"{today}.md"
            filepath = self.output_dir / filename
            
            # Create article content
            content = self._format_article(article)
            
            # Write to file
            filepath.write_text(content, encoding='utf-8')
            
            logger.info(
                "Article published successfully",
                filename=filename,
                filepath=str(filepath),
                article_title=article.title[:50] + "..." if len(article.title) > 50 else article.title
            )
            return True
            
        except Exception as e:
            logger.error("Failed to publish article", error=str(e))
            return False
    
    def _format_article(self, article: ProcessedArticle) -> str:
        """Format the article as plain text content.
        
        Args:
            article: The processed article
            
        Returns:
            Formatted article content
        """
        # Simple plain text format without markdown formatting
        content_parts = [
            f"Title: {article.title}",
            f"Date: {article.date}",
            f"Summary: {article.summary}",
            "",
            "Content:",
            article.content,
            "",
            f"Source: {article.source_url}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        return "\n".join(content_parts)
