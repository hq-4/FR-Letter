#!/usr/bin/env python3
"""
Federal Register Document Chunking and Embedding Module
Handles 512-token chunking and embedding generation with robust error handling
"""
import os
import json
import logging
import psycopg2
import requests
import time
import re
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChunkingEmbeddingProcessor:
    """Handles document chunking and embedding generation"""
    
    def __init__(self, max_tokens_per_chunk: int = 512):
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-large")
        
    def get_db_connection(self):
        """Establish connection to PostgreSQL database"""
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
            logger.error(f"Failed to connect to database: {e}")
            raise

    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """
        Split text into ~max_tokens_per_chunk token chunks with position tracking.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunk dictionaries with text, token count, and character positions
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        # Simple whitespace tokenization (for production, consider tiktoken)
        tokens = text.split()
        chunks = []
        
        # Track character positions in original text
        current_pos = 0
        
        for i in range(0, len(tokens), self.max_tokens_per_chunk):
            chunk_tokens = tokens[i:i + self.max_tokens_per_chunk]
            chunk_text = " ".join(chunk_tokens)
            
            # [CA] Calculate character positions in original text
            start_pos = text.find(chunk_tokens[0], current_pos) if chunk_tokens else current_pos
            end_pos = start_pos + len(chunk_text) if chunk_tokens else start_pos
            current_pos = end_pos
            
            chunks.append({
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "chunk_index": i // self.max_tokens_per_chunk,  # [CSD] Proper index calculation
                "start_position": start_pos,
                "end_position": end_pos
            })
            
            # [CDiP] Debug logging for chunk creation with positions
            logger.debug("Created chunk %d: %d tokens (char pos %d-%d, token range %d-%d)", 
                        i // self.max_tokens_per_chunk, len(chunk_tokens), 
                        start_pos, end_pos, i, i + len(chunk_tokens) - 1)
        
        logger.info(f"Created {len(chunks)} chunks from {len(tokens)} tokens")
        return chunks

    def generate_embedding(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """
        Generate embedding using Ollama BGE-large model with retry logic.
        
        Args:
            text: Text to embed
            max_retries: Maximum number of retry attempts
            
        Returns:
            Embedding vector or None if failed
        """
        if not text or not text.strip():
            logger.warning("Skipping embedding for empty text")
            return None

        retry_delay = 1
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_host}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    },
                    timeout=30
                )
                response.raise_for_status()
                
                embedding = response.json().get('embedding', [])
                if embedding and len(embedding) == 1024:  # BGE-large dimension check
                    logger.debug(f"Generated embedding (length: {len(embedding)})")
                    return embedding
                else:
                    logger.warning(f"Invalid embedding dimensions: {len(embedding) if embedding else 0}")
                    
            except Exception as e:
                logger.warning(f"Embedding request failed on attempt {attempt+1}: {str(e)}")
                
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        logger.error(f"Failed to generate embedding after {max_retries} retries")
        return None

    def calculate_similarity_score(self, embedding: List[float], criteria_embeddings: List[List[float]]) -> float:
        """
        Calculate maximum cosine similarity between chunk embedding and criteria embeddings.
        
        Args:
            embedding: Chunk embedding vector
            criteria_embeddings: List of criteria embedding vectors
            
        Returns:
            Maximum similarity score (0.0 to 1.0)
        """
        if not embedding or not criteria_embeddings:
            return 0.0
        
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            chunk_vec = np.array(embedding).reshape(1, -1)
            max_similarity = 0.0
            
            for criteria_vec in criteria_embeddings:
                if criteria_vec:
                    criteria_array = np.array(criteria_vec).reshape(1, -1)
                    similarity = cosine_similarity(chunk_vec, criteria_array)[0][0]
                    max_similarity = max(max_similarity, similarity)
            
            return float(max_similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0

    def load_impact_criteria_embeddings(self) -> List[List[float]]:
        """Load pre-computed impact criteria embeddings from database"""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM impact_criteria WHERE embedding IS NOT NULL")
            rows = cursor.fetchall()
            return [row[0] for row in rows if row[0]]
        except Exception as e:
            logger.error(f"Failed to load criteria embeddings: {e}")
            return []
        finally:
            conn.close()

    def process_document_chunks(self, document_slug: str) -> bool:
        """Process a single document: fetch .txt content, chunk text, embed each chunk, store results."""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            
            # [CA] Fetch document JSON and check if .txt content already exists
            cursor.execute(
                "SELECT raw_json, full_text_content, title, raw_text_url FROM processed_documents WHERE slug = %s",
                (document_slug,)
            )
            row = cursor.fetchone()
            if not row:
                logger.error("Document %s not found in database", document_slug)
                return False
            
            document_json, existing_content, existing_title, existing_url = row
            logger.debug("Processing document %s with keys: %s", document_slug, list(document_json.keys()))
            
            # [CA] Extract/fetch full .txt content with proper storage
            full_text = existing_content  # Use cached content if available
            title = existing_title or document_json.get('title', '')
            
            # If no cached content, fetch from .txt URL
            if not full_text:
                raw_text_url = document_json.get('raw_text_url')
                body_html_url = document_json.get('body_html_url')
                
                if raw_text_url:
                    try:
                        logger.debug("Fetching .txt content from: %s", raw_text_url)
                        response = requests.get(raw_text_url, timeout=30)
                        response.raise_for_status()
                        full_text = response.text.strip()
                        
                        # [CDiP] Clean up HTML tags if present in .txt file
                        if '<html>' in full_text or '<body>' in full_text:
                            full_text = re.sub(r'<[^>]+>', ' ', full_text)
                            full_text = re.sub(r'\s+', ' ', full_text).strip()
                        
                        # [CA] Store fetched content in database
                        word_count = len(full_text.split())
                        cursor.execute(
                            """
                            UPDATE processed_documents 
                            SET full_text_content = %s, 
                                title = %s,
                                raw_text_url = %s,
                                body_html_url = %s,
                                word_count = %s,
                                page_length = %s,
                                content_fetched_at = CURRENT_TIMESTAMP,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE slug = %s
                            """,
                            (full_text, title, raw_text_url, body_html_url, 
                             word_count, document_json.get('page_length', 0), document_slug)
                        )
                        conn.commit()
                        logger.info("Fetched and stored %d words of .txt content for %s", word_count, document_slug)
                        
                    except Exception as e:
                        logger.error("Failed to fetch .txt content for %s: %s", document_slug, e)
                        # Fallback to JSON metadata
                        full_text = document_json.get('abstract') or document_json.get('title', '')
                        logger.warning("Using JSON fallback content (%d chars) for %s", len(full_text), document_slug)
                else:
                    logger.warning("No raw_text_url found for %s, using JSON metadata", document_slug)
                    full_text = document_json.get('abstract') or document_json.get('title', '')
            
            # [IV] Validate extracted text
            if not full_text or not full_text.strip():
                logger.warning("No textual content found for document %s", document_slug)
                return False
            
            # [CDiP] Update processing status
            cursor.execute(
                "UPDATE processed_documents SET embedding_status = 'processing' WHERE slug = %s",
                (document_slug,)
            )
            conn.commit()
            logger.debug("Marked document %s as processing", document_slug)
            
            # [CA] Generate text chunks
            chunks = self.chunk_text(full_text)
            if not chunks:
                logger.warning("No chunks generated for document %s", document_slug)
                return False
            
            logger.info("Generated %d chunks for document %s", len(chunks), document_slug)
            
            # [RM] Load impact criteria embeddings once for efficiency
            criteria_embeddings = self.load_impact_criteria_embeddings()
            logger.debug("Loaded %d criteria embeddings", len(criteria_embeddings))
            
            # [PA] Process chunks with progress tracking
            max_similarity = 0.0
            successful_embeddings = 0
            
            for chunk in chunks:
                chunk_index = chunk["chunk_index"]
                chunk_text = chunk["text"]
                token_count = chunk["token_count"]
                start_pos = chunk.get("start_position", 0)
                end_pos = chunk.get("end_position", 0)
                
                logger.debug("Processing chunk %d/%d (tokens: %d, pos: %d-%d)", 
                           chunk_index + 1, len(chunks), token_count, start_pos, end_pos)
                
                # [REH] Generate embedding with error handling
                embedding = self.generate_embedding(chunk_text)
                similarity_score = 0.0
                
                if embedding is not None:
                    similarity_score = self.calculate_similarity_score(embedding, criteria_embeddings)
                    max_similarity = max(max_similarity, similarity_score)
                    successful_embeddings += 1
                    logger.debug("Chunk %d embedding successful, similarity: %.4f", 
                               chunk_index, similarity_score)
                else:
                    logger.warning("Failed to generate embedding for chunk %d of %s", 
                                 chunk_index, document_slug)
                
                # [CA] Store chunk with position tracking
                cursor.execute(
                    """
                    INSERT INTO document_chunks 
                        (document_slug, chunk_index, chunk_text, token_count, start_position, end_position, embedding, similarity_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_slug, chunk_index) DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        token_count = EXCLUDED.token_count,
                        start_position = EXCLUDED.start_position,
                        end_position = EXCLUDED.end_position,
                        embedding = COALESCE(EXCLUDED.embedding, document_chunks.embedding),
                        similarity_score = COALESCE(EXCLUDED.similarity_score, document_chunks.similarity_score)
                    """,
                    (document_slug, chunk_index, chunk_text, token_count, start_pos, end_pos, embedding, similarity_score)
                )
            
            # [CA] Update document-level statistics
            cursor.execute(
                """
                UPDATE processed_documents
                SET chunk_count = %s,
                    impact_score = %s,
                    embedding_status = 'completed'
                WHERE slug = %s
                """,
                (len(chunks), max_similarity, document_slug)
            )
            
            # [RM] Commit all changes
            conn.commit()
            
            logger.info("Successfully processed %d chunks for document %s (embeddings: %d/%d, max_similarity: %.4f)",
                       len(chunks), document_slug, successful_embeddings, len(chunks), max_similarity)
            return True
            
        except Exception as e:
            # [REH] Comprehensive error handling
            logger.error("Failed to process document %s: %s", document_slug, str(e))
            conn.rollback()
            
            # [REH] Mark document as failed
            try:
                cursor.execute(
                    "UPDATE processed_documents SET embedding_status = 'failed' WHERE slug = %s",
                    (document_slug,)
                )
                conn.commit()
                logger.debug("Marked document %s as failed", document_slug)
            except Exception as update_error:
                logger.error("Failed to update status for %s: %s", document_slug, update_error)
            
            return False
            
        finally:
            # [RM] Ensure connection cleanup
            conn.close()

    def process_all_pending_documents(self) -> Dict[str, int]:
        """Process all documents with pending or failed embedding status."""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            
            # [CA] Fetch all pending documents
            cursor.execute(
                "SELECT slug FROM processed_documents WHERE embedding_status IN ('pending', 'failed') ORDER BY publication_date DESC"
            )
            pending_slugs = [row[0] for row in cursor.fetchall()]
            
            # [CDiP] Initialize processing statistics
            stats = {
                "total": len(pending_slugs),
                "successful": 0,
                "failed": 0
            }
            
            logger.info("Processing %d pending documents", stats['total'])
            
            # [PA] Process each document with progress tracking
            for i, slug in enumerate(pending_slugs, 1):
                logger.info("[%d/%d] Processing document %s", i, stats['total'], slug)
                
                # [REH] Process with error handling
                if self.process_document_chunks(slug):
                    stats["successful"] += 1
                    logger.debug("Successfully processed document %s", slug)
                else:
                    stats["failed"] += 1
                    logger.warning("Failed to process document %s", slug)
                
                # [PA] Rate limiting to avoid overwhelming the system
                time.sleep(0.1)
            
            # [CDiP] Log final statistics
            logger.info("Processing completed: %d successful, %d failed out of %d total",
                       stats['successful'], stats['failed'], stats['total'])
            return stats
            
        except Exception as e:
            # [REH] Handle batch processing errors
            logger.error("Failed during batch processing: %s", str(e))
            return {"total": 0, "successful": 0, "failed": 0}
            
        finally:
            # [RM] Ensure connection cleanup
            conn.close()


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process document chunks and embeddings')
    parser.add_argument('--slug', help='Process specific document slug')
    parser.add_argument('--all', action='store_true', help='Process all pending documents')
    parser.add_argument('--max-tokens', type=int, default=512, help='Maximum tokens per chunk')
    
    args = parser.parse_args()
    
    processor = ChunkingEmbeddingProcessor(max_tokens_per_chunk=args.max_tokens)
    
    if args.slug:
        success = processor.process_document_chunks(args.slug)
        exit(0 if success else 1)
    elif args.all:
        stats = processor.process_all_pending_documents()
        exit(0 if stats["failed"] == 0 else 1)
    else:
        print("Use --slug <slug> to process specific document or --all to process all pending")
        exit(1)


if __name__ == "__main__":
    main()
