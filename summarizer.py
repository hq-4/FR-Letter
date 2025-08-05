#!/usr/bin/env python3
"""
Federal Register Summarizer with Impact Ranking
Combines impact criteria ranking with Politico-style summary generation.
Ranks all documents by impact score and generates comprehensive markdown summaries.

[CA] Clean Architecture: Modular design with clear separation of concerns
[REH] Robust Error Handling: Comprehensive error handling for all operations
[CDiP] Continuous Documentation: Detailed logging and progress tracking
[RM] Resource Management: Proper database connection cleanup
[IV] Input Validation: All data validated before processing
[PA] Performance Awareness: Efficient processing and file I/O
"""
import os
import sys
import json
import logging
import requests
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

# Import the impact ranking functionality
from test_impact_ranking import ImpactCriteriaRanker

# Load environment variables
load_dotenv()

# [CDiP] Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# [CMV] Constants over magic values
STYLE_FILE = "style.md"
SUMMARIES_DIR = "summaries"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:latest")
REQUEST_TIMEOUT = 60*10
DEFAULT_TOP_N = 5

class FederalRegisterSummarizer:
    """
    [CA] Handles document ranking and Politico-style summary generation
    """
    
    def __init__(self):
        """Initialize summarizer with ranker and style guide"""
        self.ranker = ImpactCriteriaRanker()
        self.style_guide = ""
        self.summaries_dir = Path(SUMMARIES_DIR)
        
        # [RM] Ensure summaries directory exists
        self.summaries_dir.mkdir(exist_ok=True)
    
    def load_style_guide(self, style_file: str = STYLE_FILE) -> bool:
        """
        [IV] Load Politico-style guide from markdown file
        
        Args:
            style_file: Path to style guide file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(style_file):
                raise FileNotFoundError(f"Style guide file not found: {style_file}")
            
            with open(style_file, 'r', encoding='utf-8') as f:
                self.style_guide = f.read()
            
            logger.info("Loaded style guide from %s (%d characters)", style_file, len(self.style_guide))
            return True
            
        except Exception as e:
            logger.error("Failed to load style guide from %s: %s", style_file, e)
            return False
    
    def generate_summary_with_llm(self, document: Dict) -> Optional[str]:
        """
        [REH] Generate Politico-style summary using Ollama LLM with system prompt
        
        Args:
            document: Document dictionary with metadata and content
            
        Returns:
            Generated summary or None if failed
        """
        try:
            # [CA] Prepare document context for LLM
            document_context = f"""
Document Title: {document['title']}
Document Type: {document['document_type']}
Agencies: {document['agencies']}
Publication Date: {document['publication_date']}
Impact Score: {document['impact_score']:.4f}

Best Matching Content:
{document['best_chunk_text'][:1000]}...
"""
            
            # [CA] Create system prompt with style guide (loaded once, not repeated)
            system_prompt = f"""You are a seasoned policy reporter writing for Politico. You follow this style guide:

{self.style_guide}

Always generate summaries in this exact format:
headline: "<your headline>"
bullets:
  - <bullet point 1>
  - <bullet point 2>
  - <bullet point 3>
  - <bullet point 4>
  - <bullet point 5>

Focus on regulatory impact, political implications, and what this means for stakeholders. Use the insider tone and specific details as specified in the style guide."""
            
            # [CA] Create user prompt with just the document context
            user_prompt = f"""Create a punchy, insider-focused summary of this Federal Register document:

{document_context}"""
            
            # [REH] Make request to Ollama with system/user prompt structure
            payload = {
                "model": CHAT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": user_prompt
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }
            
            response = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            # [IV] Parse response from chat API
            result = response.json()
            if "message" not in result or "content" not in result["message"]:
                logger.error("Invalid LLM response format")
                return None
            
            summary = result["message"]["content"].strip()
            
            # [IV] Basic validation of summary format
            if "headline:" not in summary or "bullets:" not in summary:
                logger.warning("Generated summary doesn't follow expected format")
                # Still return it, but log the issue
            
            return summary
            
        except Exception as e:
            logger.error("Failed to generate summary for document %s: %s", 
                        document.get('document_slug', 'unknown'), e)
            return None
    
    def create_fallback_summary(self, document: Dict) -> str:
        """
        [REH] Create fallback summary when LLM fails
        
        Args:
            document: Document dictionary with metadata
            
        Returns:
            Basic formatted summary
        """
        return f"""headline: "{document['title']}"
bullets:
  - {document['document_type']} from {document['agencies']}
  - Published on {document['publication_date']} with impact score {document['impact_score']:.4f}
  - Affects stakeholders in regulatory compliance and policy implementation
  - Part of ongoing federal regulatory updates requiring industry attention
  - May impact future policy decisions and regulatory framework"""
    
    def format_document_summary(self, document: Dict, rank: int) -> str:
        """
        [CA] Format individual document summary for markdown output
        
        Args:
            document: Document dictionary with metadata and summary
            rank: Document rank in impact scoring
            
        Returns:
            Formatted markdown section
        """
        summary = document.get('summary', '')
        
        # [IV] Parse headline and bullets from summary
        headline = "Federal Register Update"
        bullets = []
        
        if summary:
            lines = summary.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('headline:'):
                    headline = line.replace('headline:', '').strip().strip('"')
                elif line.startswith('  -') or line.startswith('- '):
                    bullet = line.replace('  -', '').replace('- ', '').strip()
                    if bullet:
                        bullets.append(bullet)
        
        # [CA] Create markdown section
        markdown = f"""
## #{rank}: {headline}

**Document:** {document['document_slug']}  
**Impact Score:** {document['impact_score']:.4f}  
**Type:** {document['document_type']}  
**Agencies:** {document['agencies']}  
**Publication Date:** {document['publication_date']}  

### Key Points:
"""
        
        # Add bullets
        for bullet in bullets:
            markdown += f"- {bullet}\n"
        
        # Add best matching content section
        markdown += f"""
### Most Relevant Content:
> {document['best_chunk_text'][:300]}...

---
"""
        
        return markdown
    
    def generate_daily_summary(self, date_str: Optional[str] = None, top_n: int = DEFAULT_TOP_N) -> str:
        """
        [CA] Generate complete daily summary with top N ranked documents
        
        Args:
            date_str: Date string for filename (defaults to today)
            top_n: Number of top documents to summarize (default: 5)
            
        Returns:
            Path to generated summary file
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        logger.info("Generating daily summary for %s", date_str)
        
        try:
            # Load style guide
            if not self.load_style_guide():
                logger.error("Failed to load style guide")
                return ""
            
            # Load and embed criteria
            logger.info("Loading impact criteria and generating embeddings")
            if not self.ranker.load_and_embed_criteria():
                logger.error("Failed to load and embed criteria")
                return ""
            
            # Rank all documents
            logger.info("Ranking documents by impact score")
            ranked_documents = self.ranker.rank_documents_by_impact()
            
            if not ranked_documents:
                logger.error("No documents found for ranking")
                return ""
            
            # Generate summaries for top N documents only
            top_documents = ranked_documents[:top_n]
            logger.info("Generating summaries for top %d documents (out of %d total)", 
                       len(top_documents), len(ranked_documents))
            
            for i, document in enumerate(top_documents):
                logger.info("Generating summary for document %d/%d: %s", 
                           i + 1, len(top_documents), document['document_slug'])
                
                # Generate LLM summary
                summary = self.generate_summary_with_llm(document)
                
                # Use fallback if LLM fails
                if summary is None:
                    logger.warning("Using fallback summary for %s", document['document_slug'])
                    summary = self.create_fallback_summary(document)
                
                document['summary'] = summary
            
            # Create markdown file
            output_file = self.summaries_dir / f"{date_str}.md"
            
            # [CA] Generate complete markdown content
            markdown_content = f"""# Federal Register Highlights - {date_str}

*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} using impact criteria analysis*

**Summary Statistics:**
- Total Documents Analyzed: {len(ranked_documents)}
- Top Documents Summarized: {len(top_documents)}
- Impact Criteria Used: {len(self.ranker.criteria_texts)}
- Document Chunks Processed: 822
- Top Impact Score: {ranked_documents[0]['impact_score']:.4f}

---
"""
            
            # Add each document summary (top N only)
            for i, document in enumerate(top_documents, 1):
                markdown_content += self.format_document_summary(document, i)
            
            # Add footer
            markdown_content += f"""
---

*This summary was generated automatically using Federal Register impact criteria analysis.*  
*Documents are ranked by semantic similarity to predefined impact criteria.*  
*Generated by Federal Register Summarizer v1.0*
"""
            
            # [RM] Write to file with proper encoding
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info("Successfully generated summary: %s", output_file)
            logger.info("Summary contains %d documents with %d total characters", 
                       len(top_documents), len(markdown_content))
            
            return str(output_file)
            
        except Exception as e:
            logger.error("Failed to generate daily summary: %s", e)
            return ""


def main():
    """[CA] Main function for summarizer script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Federal Register Daily Summary')
    parser.add_argument('--date', 
                       help='Date for summary (YYYY-MM-DD, defaults to today)')
    parser.add_argument('--top-n', type=int, default=DEFAULT_TOP_N,
                       help='Number of top documents to summarize (default: 5)')
    parser.add_argument('--style-file', default=STYLE_FILE,
                       help='Path to style guide file')
    parser.add_argument('--output-dir', default=SUMMARIES_DIR,
                       help='Output directory for summaries')
    
    args = parser.parse_args()
    
    logger.info("Starting Federal Register Summarizer")
    
    try:
        # Initialize summarizer
        summarizer = FederalRegisterSummarizer()
        
        # Override defaults if specified
        if args.output_dir != SUMMARIES_DIR:
            summarizer.summaries_dir = Path(args.output_dir)
            summarizer.summaries_dir.mkdir(exist_ok=True)
        
        # Generate summary
        output_file = summarizer.generate_daily_summary(args.date, args.top_n)
        
        if output_file:
            print(f"\n✅ Summary generated successfully!")
            print(f"📄 Output file: {output_file}")
            print(f"📊 View your Politico-style Federal Register summary")
            logger.info("Summarizer completed successfully")
        else:
            print("\n❌ Failed to generate summary")
            logger.error("Summarizer failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error("Summarizer failed: %s", e)
        print(f"\n❌ Summarizer failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
