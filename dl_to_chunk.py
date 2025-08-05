#!/usr/bin/env python3
"""
Federal Register Document Download-to-Chunk Orchestration Pipeline
Complete workflow: RSS parsing → JSON fetching → title ingestion → .txt content fetching → chunking → embedding

[CA] Clean Architecture: Modular design with clear separation of concerns
[REH] Robust Error Handling: Comprehensive error handling with rollback and status updates
[CDiP] Continuous Documentation: Detailed logging after key operations
[RM] Resource Management: Proper connection cleanup and resource handling
[IV] Input Validation: All external data validated before processing
[SFT] Security-First Thinking: Safe URL handling and input sanitization
[PA] Performance Awareness: Rate limiting and batch processing
"""
import os
import sys
import json
import logging
import requests
import psycopg2
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv

# Import existing chunking processor
from chunking_embeddings import ChunkingEmbeddingProcessor

# Load environment variables
load_dotenv()

# [CDiP] Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dl_to_chunk.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# [CMV] Constants over magic values
RSS_ENDPOINT = "https://www.federalregister.gov/api/v1/documents.rss"
JSON_API_BASE = "https://www.federalregister.gov/api/v1/documents/"
REQUEST_TIMEOUT = 30
RATE_LIMIT_DELAY = 0.1
MAX_RETRIES = 3

class FederalRegisterOrchestrator:
    """
    [CA] Complete orchestration class for Federal Register document processing
    Handles RSS parsing, JSON fetching, ingestion, .txt content retrieval, chunking, and embedding
    """
    
    def __init__(self):
        """Initialize orchestrator with database connection and chunking processor"""
        self.chunking_processor = ChunkingEmbeddingProcessor()
        
    def get_db_connection(self) -> psycopg2.extensions.connection:
        """[RM] Establish connection to PostgreSQL database with proper error handling"""
        try:
            conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                database=os.getenv("POSTGRES_DB", "federalregister"),
                user=os.getenv("POSTGRES_USER", "user"),
                password=os.getenv("POSTGRES_PASSWORD", "")
            )
            return conn
        except Exception as e:
            logger.error("Failed to connect to database: %s", e)
            raise
    
    def parse_rss_feed(self, rss_url: str = RSS_ENDPOINT) -> List[Dict[str, str]]:
        """
        [IV] Parse RSS feed and extract document metadata with input validation
        
        Args:
            rss_url: RSS feed URL to parse
            
        Returns:
            List of document dictionaries with slug, title, link, and json_url
        """
        logger.info("Parsing RSS feed from %s", rss_url)
        
        try:
            # [SFT] Validate URL before making request
            parsed_url = urlparse(rss_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid RSS URL: {rss_url}")
            
            # [REH] Fetch RSS with timeout and error handling
            response = requests.get(rss_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # [IV] Parse XML with error handling
            root = ET.fromstring(response.content)
            
            documents = []
            items = root.findall('.//item')
            
            for item in items:
                try:
                    # [IV] Extract and validate required fields
                    title_elem = item.find('title')
                    link_elem = item.find('link')
                    guid_elem = item.find('guid')
                    
                    if title_elem is None or link_elem is None or guid_elem is None:
                        logger.warning("Skipping item with missing required fields")
                        continue
                    
                    title = title_elem.text.strip() if title_elem.text else ""
                    link = link_elem.text.strip() if link_elem.text else ""
                    guid = guid_elem.text.strip() if guid_elem.text else ""
                    
                    # [IV] Validate that we have actual content
                    if not title or not link:
                        logger.warning("Skipping item with empty title or link")
                        continue
                    
                    # [IV] Extract slug from link (e.g., 2025-14789 from URL)
                    slug = self._extract_slug_from_url(link)
                    if not slug:
                        logger.warning("Could not extract slug from link: %s", link)
                        continue
                    
                    # [CA] Construct JSON API URL
                    json_url = f"{JSON_API_BASE}{slug}.json"
                    
                    documents.append({
                        "slug": slug,
                        "title": title,
                        "link": link,
                        "guid": guid,
                        "json_url": json_url
                    })
                    
                    # [CDiP] Debug logging for each document
                    logger.debug("Parsed document: slug=%s, title=%s", slug, title[:50])
                    
                except Exception as e:
                    logger.warning("Failed to parse RSS item: %s", e)
                    continue
            
            logger.info("Successfully parsed %d documents from RSS feed", len(documents))
            return documents
            
        except Exception as e:
            logger.error("Failed to parse RSS feed: %s", e)
            raise
    
    def _extract_slug_from_url(self, url: str) -> Optional[str]:
        """[IV] Extract document slug from Federal Register URL"""
        try:
            # URL format: https://www.federalregister.gov/documents/2025/08/04/2025-14789/title
            parts = url.split('/')
            for part in parts:
                if part.startswith('2025-') and len(part) == 10:  # Format: 2025-XXXXX
                    return part
            return None
        except Exception:
            return None
    
    def fetch_document_json(self, json_url: str) -> Optional[Dict]:
        """
        [REH] Fetch document JSON from API with retry logic
        
        Args:
            json_url: JSON API URL to fetch
            
        Returns:
            Document JSON data or None if failed
        """
        for attempt in range(MAX_RETRIES):
            try:
                # [SFT] Validate URL
                parsed_url = urlparse(json_url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    logger.error("Invalid JSON URL: %s", json_url)
                    return None
                
                response = requests.get(json_url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                
                # [IV] Validate JSON response
                doc_json = response.json()
                if not isinstance(doc_json, dict):
                    logger.error("Invalid JSON response format from %s", json_url)
                    return None
                
                logger.debug("Successfully fetched JSON for %s", json_url)
                return doc_json
                
            except requests.exceptions.RequestException as e:
                logger.warning("Attempt %d failed for %s: %s", attempt + 1, json_url, e)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RATE_LIMIT_DELAY * (attempt + 1))  # [PA] Exponential backoff
                    continue
                else:
                    logger.error("All attempts failed for %s", json_url)
                    return None
            except Exception as e:
                logger.error("Unexpected error fetching %s: %s", json_url, e)
                return None
    
    def ingest_document(self, slug: str, document_json: Dict) -> bool:
        """
        [CA] Ingest document JSON into database with deduplication
        
        Args:
            slug: Document slug (unique identifier)
            document_json: Full JSON document data
            
        Returns:
            True if successfully ingested, False otherwise
        """
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            
            # [IV] Extract and validate key fields
            title = document_json.get("title", "").strip()
            agencies = []
            if "agencies" in document_json and isinstance(document_json["agencies"], list):
                agencies = [ag.get("name", "Unnamed Agency") for ag in document_json["agencies"]]
            agencies_str = ", ".join(agencies)
            
            document_type = document_json.get("type", "").strip()
            publication_date = document_json.get("publication_date")
            
            # [IV] Parse publication date
            pub_date = None
            if publication_date:
                try:
                    pub_date = datetime.strptime(publication_date, "%Y-%m-%d").date()
                except ValueError:
                    logger.warning("Invalid publication date format for %s: %s", slug, publication_date)
            
            # [IV] Extract .txt content URLs
            raw_text_url = document_json.get("raw_text_url", "").strip()
            body_html_url = document_json.get("body_html_url", "").strip()
            
            # [REH] Insert with conflict handling (deduplication)
            cursor.execute(
                """
                INSERT INTO processed_documents 
                (slug, publication_date, raw_json, title, agencies, document_type, 
                 raw_text_url, body_html_url, embedding_status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (slug) DO UPDATE SET
                    title = EXCLUDED.title,
                    agencies = EXCLUDED.agencies,
                    document_type = EXCLUDED.document_type,
                    raw_text_url = EXCLUDED.raw_text_url,
                    body_html_url = EXCLUDED.body_html_url,
                    raw_json = EXCLUDED.raw_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (slug, pub_date, json.dumps(document_json), title, agencies_str, 
                 document_type, raw_text_url, body_html_url)
            )
            
            # [CDiP] Log ingestion result
            if cursor.rowcount > 0:
                logger.debug("Successfully ingested document %s: %s", slug, title[:50])
                conn.commit()
                return True
            else:
                logger.debug("Document %s already exists, updated metadata", slug)
                conn.commit()
                return True
                
        except Exception as e:
            logger.error("Failed to ingest document %s: %s", slug, e)
            conn.rollback()
            return False
        finally:
            # [RM] Ensure connection cleanup
            conn.close()
    
    def run_complete_pipeline(self) -> Dict[str, int]:
        """
        [CA] Execute complete pipeline: RSS → JSON → Ingest → Chunk → Embed
        
        Returns:
            Statistics dictionary with counts for each stage
        """
        stats = {
            "rss_documents": 0,
            "json_fetched": 0,
            "ingested": 0,
            "chunked": 0,
            "failed": 0
        }
        
        logger.info("Starting complete Federal Register processing pipeline")
        
        try:
            # Stage 1: Parse RSS feed
            logger.info("Stage 1: Parsing RSS feed")
            documents = self.parse_rss_feed()
            stats["rss_documents"] = len(documents)
            
            if not documents:
                logger.warning("No documents found in RSS feed")
                return stats
            
            # Stage 2: Fetch JSON and ingest documents
            logger.info("Stage 2: Fetching JSON documents and ingesting")
            for i, doc in enumerate(documents, 1):
                slug = doc["slug"]
                json_url = doc["json_url"]
                
                logger.info("[%d/%d] Processing document %s", i, len(documents), slug)
                
                # [PA] Rate limiting
                if i > 1:
                    time.sleep(RATE_LIMIT_DELAY)
                
                # Fetch JSON
                document_json = self.fetch_document_json(json_url)
                if document_json is None:
                    stats["failed"] += 1
                    continue
                
                stats["json_fetched"] += 1
                
                # Ingest document
                if self.ingest_document(slug, document_json):
                    stats["ingested"] += 1
                else:
                    stats["failed"] += 1
            
            # Stage 3: Chunk and embed all pending documents
            logger.info("Stage 3: Chunking and embedding documents")
            chunking_stats = self.chunking_processor.process_all_pending_documents()
            stats["chunked"] = chunking_stats["successful"]
            stats["failed"] += chunking_stats["failed"]
            
            # [CDiP] Log final statistics
            logger.info("Pipeline completed successfully")
            logger.info("RSS documents: %d", stats["rss_documents"])
            logger.info("JSON fetched: %d", stats["json_fetched"])
            logger.info("Documents ingested: %d", stats["ingested"])
            logger.info("Documents chunked: %d", stats["chunked"])
            logger.info("Total failures: %d", stats["failed"])
            
            return stats
            
        except Exception as e:
            logger.error("Pipeline failed: %s", e)
            stats["failed"] += 1
            return stats
    
    def verify_database_state(self) -> Dict[str, int]:
        """
        [CDiP] Verify database state with SQL queries for validation
        
        Returns:
            Dictionary with database counts and statistics
        """
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Count processed documents
            cursor.execute("SELECT COUNT(*) FROM processed_documents")
            doc_count = cursor.fetchone()[0]
            
            # Count document chunks
            cursor.execute("SELECT COUNT(*) FROM document_chunks")
            chunk_count = cursor.fetchone()[0]
            
            # Count by embedding status
            cursor.execute("""
                SELECT embedding_status, COUNT(*) 
                FROM processed_documents 
                GROUP BY embedding_status
            """)
            status_counts = dict(cursor.fetchall())
            
            # Average chunks per document
            cursor.execute("""
                SELECT AVG(chunk_count) 
                FROM processed_documents 
                WHERE chunk_count > 0
            """)
            avg_chunks = cursor.fetchone()[0] or 0
            
            # Documents with .txt content
            cursor.execute("""
                SELECT COUNT(*) 
                FROM processed_documents 
                WHERE full_text_content IS NOT NULL AND LENGTH(full_text_content) > 0
            """)
            txt_content_count = cursor.fetchone()[0]
            
            verification = {
                "total_documents": doc_count,
                "total_chunks": chunk_count,
                "avg_chunks_per_doc": float(avg_chunks),
                "documents_with_txt_content": txt_content_count,
                "status_counts": status_counts
            }
            
            # [CDiP] Log verification results
            logger.info("Database verification results:")
            logger.info("Total documents: %d", doc_count)
            logger.info("Total chunks: %d", chunk_count)
            logger.info("Average chunks per document: %.2f", avg_chunks)
            logger.info("Documents with .txt content: %d", txt_content_count)
            logger.info("Status breakdown: %s", status_counts)
            
            return verification
            
        except Exception as e:
            logger.error("Database verification failed: %s", e)
            return {}
        finally:
            # [RM] Ensure connection cleanup
            conn.close()


def main():
    """[CA] Main orchestration function with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Federal Register Download-to-Chunk Pipeline')
    parser.add_argument('--verify-only', action='store_true', 
                       help='Only verify database state without processing')
    parser.add_argument('--rss-url', default=RSS_ENDPOINT,
                       help='RSS feed URL to process')
    
    args = parser.parse_args()
    
    orchestrator = FederalRegisterOrchestrator()
    
    if args.verify_only:
        # [CDiP] Verification mode
        logger.info("Running database verification only")
        verification = orchestrator.verify_database_state()
        
        # Expected values validation
        expected_docs = 200
        if verification.get("total_documents", 0) >= expected_docs:
            logger.info("✓ Document count meets expectations (%d >= %d)", 
                       verification["total_documents"], expected_docs)
        else:
            logger.warning("⚠ Document count below expectations (%d < %d)", 
                          verification.get("total_documents", 0), expected_docs)
        
        if verification.get("total_chunks", 0) > verification.get("total_documents", 0):
            logger.info("✓ Chunk count exceeds document count (multi-chunk documents confirmed)")
        else:
            logger.warning("⚠ Chunk count suspicious - may indicate chunking issues")
        
        return
    
    # [CA] Run complete pipeline
    logger.info("Starting Federal Register Download-to-Chunk Pipeline")
    stats = orchestrator.run_complete_pipeline()
    
    # [CDiP] Verify final state
    logger.info("Verifying final database state")
    verification = orchestrator.verify_database_state()
    
    # [REH] Exit with appropriate code
    if stats["failed"] > 0:
        logger.warning("Pipeline completed with %d failures", stats["failed"])
        sys.exit(1)
    else:
        logger.info("Pipeline completed successfully with no failures")
        sys.exit(0)


if __name__ == "__main__":
    main()
