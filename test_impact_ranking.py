#!/usr/bin/env python3
"""
Impact Criteria Ranking Test Script
Loads impact criteria from criteria.md, computes on-the-fly similarity rankings 
for all document chunks, and displays top 5 most relevant notices to stdout.

[CA] Clean Architecture: Modular design with clear separation of concerns
[REH] Robust Error Handling: Comprehensive error handling for all operations
[CDiP] Continuous Documentation: Detailed logging and progress tracking
[RM] Resource Management: Proper database connection cleanup
[IV] Input Validation: All data validated before processing
[PA] Performance Awareness: Efficient batch processing and similarity computation
"""
import os
import sys
import logging
import psycopg2
import requests
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# [CDiP] Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# [CMV] Constants over magic values
CRITERIA_FILE = "criteria.md"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-large")
REQUEST_TIMEOUT = 30
TOP_N_RESULTS = 5

class ImpactCriteriaRanker:
    """
    [CA] Handles impact criteria loading, embedding generation, and similarity ranking
    """
    
    def __init__(self):
        """Initialize ranker with database connection"""
        self.criteria_embeddings = []
        self.criteria_texts = []
    
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
    
    def load_criteria_from_file(self, criteria_file: str = CRITERIA_FILE) -> List[str]:
        """
        [IV] Load and parse impact criteria from markdown file
        
        Args:
            criteria_file: Path to criteria markdown file
            
        Returns:
            List of criteria text strings
        """
        try:
            if not os.path.exists(criteria_file):
                raise FileNotFoundError(f"Criteria file not found: {criteria_file}")
            
            with open(criteria_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # [IV] Parse markdown content to extract criteria
            criteria_texts = []
            lines = content.split('\n')
            
            current_section = ""
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and headers
                if not line or line.startswith('#'):
                    if line.startswith('##'):
                        current_section = line.replace('##', '').strip().lower()
                    continue
                
                # Extract bullet points and convert to criteria text
                if line.startswith('-'):
                    criterion = line[1:].strip()
                    if criterion:
                        # [CA] Create contextual criteria text based on section
                        if current_section == "keywords":
                            criteria_texts.append(f"Federal Register document about {criterion}")
                        elif current_section == "agencies":
                            criteria_texts.append(f"Federal Register document from {criterion}")
                        elif current_section == "document types":
                            criteria_texts.append(f"Federal Register {criterion}")
                        else:
                            criteria_texts.append(criterion)
            
            logger.info("Loaded %d criteria from %s", len(criteria_texts), criteria_file)
            
            # [CDiP] Log loaded criteria for debugging
            for i, criterion in enumerate(criteria_texts):
                logger.debug("Criterion %d: %s", i + 1, criterion)
            
            return criteria_texts
            
        except Exception as e:
            logger.error("Failed to load criteria from %s: %s", criteria_file, e)
            raise
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        [REH] Generate embedding using Ollama BGE-large model with error handling
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        if not text or not text.strip():
            logger.warning("Skipping embedding for empty text")
            return None
        
        try:
            # [SFT] Prepare request payload
            payload = {
                "model": EMBEDDING_MODEL,
                "prompt": text.strip()
            }
            
            response = requests.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            # [IV] Validate response format
            result = response.json()
            if "embedding" not in result:
                logger.error("Invalid embedding response format")
                return None
            
            embedding = result["embedding"]
            
            # [IV] Validate embedding dimensions
            if not isinstance(embedding, list) or len(embedding) != 1024:
                logger.error("Invalid embedding dimensions: expected 1024, got %d", len(embedding))
                return None
            
            return embedding
            
        except Exception as e:
            logger.error("Failed to generate embedding for text: %s", e)
            return None
    
    def load_and_embed_criteria(self) -> bool:
        """
        [CA] Load criteria from file and generate embeddings
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load criteria texts
            self.criteria_texts = self.load_criteria_from_file()
            
            if not self.criteria_texts:
                logger.error("No criteria loaded from file")
                return False
            
            # Generate embeddings for each criterion
            logger.info("Generating embeddings for %d criteria", len(self.criteria_texts))
            self.criteria_embeddings = []
            
            for i, criterion_text in enumerate(self.criteria_texts):
                logger.info("Embedding criterion %d/%d: %s", i + 1, len(self.criteria_texts), criterion_text[:50])
                
                embedding = self.generate_embedding(criterion_text)
                if embedding is None:
                    logger.warning("Failed to generate embedding for criterion: %s", criterion_text)
                    continue
                
                self.criteria_embeddings.append(embedding)
            
            logger.info("Successfully generated %d criteria embeddings", len(self.criteria_embeddings))
            return len(self.criteria_embeddings) > 0
            
        except Exception as e:
            logger.error("Failed to load and embed criteria: %s", e)
            return False
    
    def calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        [PA] Calculate cosine similarity between two vectors efficiently
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            # [IV] Validate input vectors
            if not vec1 or not vec2 or len(vec1) != len(vec2):
                return 0.0
            
            # Calculate dot product and magnitudes
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            
            # [REH] Handle zero magnitude vectors
            if magnitude1 == 0.0 or magnitude2 == 0.0:
                return 0.0
            
            # Calculate cosine similarity and normalize to 0-1 range
            similarity = dot_product / (magnitude1 * magnitude2)
            return max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            
        except Exception as e:
            logger.error("Failed to calculate cosine similarity: %s", e)
            return 0.0
    
    def calculate_max_similarity(self, chunk_embedding: List[float]) -> float:
        """
        [PA] Calculate maximum similarity between chunk and all criteria embeddings
        
        Args:
            chunk_embedding: Embedding vector for document chunk
            
        Returns:
            Maximum similarity score across all criteria
        """
        if not chunk_embedding or not self.criteria_embeddings:
            return 0.0
        
        max_similarity = 0.0
        for criteria_embedding in self.criteria_embeddings:
            similarity = self.calculate_cosine_similarity(chunk_embedding, criteria_embedding)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def load_document_chunks(self) -> List[Dict]:
        """
        [RM] Load all document chunks with embeddings from database
        
        Returns:
            List of chunk dictionaries with metadata and embeddings
        """
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            
            # [CA] Query chunks with document metadata
            cursor.execute("""
                SELECT 
                    dc.document_slug,
                    dc.chunk_index,
                    dc.chunk_text,
                    dc.embedding,
                    pd.title,
                    pd.agencies,
                    pd.document_type,
                    pd.publication_date
                FROM document_chunks dc
                JOIN processed_documents pd ON dc.document_slug = pd.slug
                WHERE dc.embedding IS NOT NULL
                ORDER BY pd.publication_date DESC, dc.chunk_index ASC
            """)
            
            chunks = []
            for row in cursor.fetchall():
                slug, chunk_index, chunk_text, embedding_json, title, agencies, doc_type, pub_date = row
                
                # [IV] Parse embedding JSON
                try:
                    embedding = json.loads(embedding_json) if embedding_json else None
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Invalid embedding JSON for chunk %s:%d", slug, chunk_index)
                    continue
                
                if not embedding:
                    continue
                
                chunks.append({
                    'document_slug': slug,
                    'chunk_index': chunk_index,
                    'chunk_text': chunk_text,
                    'embedding': embedding,
                    'title': title,
                    'agencies': agencies,
                    'document_type': doc_type,
                    'publication_date': pub_date
                })
            
            logger.info("Loaded %d chunks with embeddings", len(chunks))
            return chunks
            
        except Exception as e:
            logger.error("Failed to load document chunks: %s", e)
            return []
        finally:
            # [RM] Ensure connection cleanup
            conn.close()
    
    def rank_documents_by_impact(self) -> List[Dict]:
        """
        [CA] Rank all documents by maximum impact score across their chunks
        
        Returns:
            List of documents ranked by impact score (highest first)
        """
        try:
            # Load document chunks
            chunks = self.load_document_chunks()
            if not chunks:
                logger.error("No chunks loaded for ranking")
                return []
            
            # [PA] Calculate similarity scores for all chunks
            logger.info("Calculating impact scores for %d chunks", len(chunks))
            document_scores = {}
            
            for i, chunk in enumerate(chunks):
                if (i + 1) % 100 == 0:
                    logger.info("Processed %d/%d chunks", i + 1, len(chunks))
                
                # Calculate max similarity for this chunk
                similarity_score = self.calculate_max_similarity(chunk['embedding'])
                
                slug = chunk['document_slug']
                
                # [CA] Track maximum similarity score per document
                if slug not in document_scores:
                    document_scores[slug] = {
                        'max_similarity': similarity_score,
                        'title': chunk['title'],
                        'agencies': chunk['agencies'],
                        'document_type': chunk['document_type'],
                        'publication_date': chunk['publication_date'],
                        'best_chunk_text': chunk['chunk_text'],
                        'best_chunk_index': chunk['chunk_index']
                    }
                else:
                    # Update if this chunk has higher similarity
                    if similarity_score > document_scores[slug]['max_similarity']:
                        document_scores[slug]['max_similarity'] = similarity_score
                        document_scores[slug]['best_chunk_text'] = chunk['chunk_text']
                        document_scores[slug]['best_chunk_index'] = chunk['chunk_index']
            
            # [CA] Convert to ranked list
            ranked_documents = []
            for slug, doc_data in document_scores.items():
                ranked_documents.append({
                    'document_slug': slug,
                    'impact_score': doc_data['max_similarity'],
                    'title': doc_data['title'],
                    'agencies': doc_data['agencies'],
                    'document_type': doc_data['document_type'],
                    'publication_date': doc_data['publication_date'],
                    'best_chunk_text': doc_data['best_chunk_text'],
                    'best_chunk_index': doc_data['best_chunk_index']
                })
            
            # [PA] Sort by impact score (highest first)
            ranked_documents.sort(key=lambda x: x['impact_score'], reverse=True)
            
            logger.info("Ranked %d documents by impact score", len(ranked_documents))
            return ranked_documents
            
        except Exception as e:
            logger.error("Failed to rank documents by impact: %s", e)
            return []
    
    def display_top_results(self, ranked_documents: List[Dict], top_n: int = TOP_N_RESULTS):
        """
        [CDiP] Display top N results to stdout with formatted output
        
        Args:
            ranked_documents: List of documents ranked by impact score
            top_n: Number of top results to display
        """
        if not ranked_documents:
            print("No documents found for ranking.")
            return
        
        print(f"\n{'='*80}")
        print(f"TOP {min(top_n, len(ranked_documents))} MOST IMPACTFUL FEDERAL REGISTER NOTICES")
        print(f"{'='*80}")
        print(f"Ranked from {len(ranked_documents)} total documents")
        print(f"Criteria loaded: {len(self.criteria_texts)} impact criteria")
        print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        for i, doc in enumerate(ranked_documents[:top_n], 1):
            print(f"RANK #{i}")
            print(f"Impact Score: {doc['impact_score']:.4f}")
            print(f"Document: {doc['document_slug']}")
            print(f"Title: {doc['title']}")
            print(f"Agencies: {doc['agencies']}")
            print(f"Type: {doc['document_type']}")
            print(f"Publication Date: {doc['publication_date']}")
            print(f"Best Matching Chunk (#{doc['best_chunk_index']}):")
            
            # [CDiP] Display chunk text with word wrapping
            chunk_text = doc['best_chunk_text'][:500]  # Limit to 500 chars
            if len(doc['best_chunk_text']) > 500:
                chunk_text += "..."
            
            # Simple word wrapping
            words = chunk_text.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + " " + word) <= 75:
                    current_line += (" " if current_line else "") + word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            for line in lines:
                print(f"  {line}")
            
            print(f"{'-'*80}\n")


def main():
    """[CA] Main function for impact criteria ranking test"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Impact Criteria Ranking')
    parser.add_argument('--criteria-file', default=CRITERIA_FILE,
                       help='Path to criteria markdown file')
    parser.add_argument('--top-n', type=int, default=TOP_N_RESULTS,
                       help='Number of top results to display')
    
    args = parser.parse_args()
    
    logger.info("Starting Impact Criteria Ranking Test")
    
    try:
        # Initialize ranker
        ranker = ImpactCriteriaRanker()
        
        # Load and embed criteria
        logger.info("Loading impact criteria from %s", args.criteria_file)
        if not ranker.load_and_embed_criteria():
            logger.error("Failed to load and embed criteria")
            sys.exit(1)
        
        # Rank documents
        logger.info("Ranking documents by impact score")
        ranked_documents = ranker.rank_documents_by_impact()
        
        if not ranked_documents:
            logger.error("No documents found for ranking")
            sys.exit(1)
        
        # Display results
        ranker.display_top_results(ranked_documents, args.top_n)
        
        logger.info("Impact criteria ranking test completed successfully")
        
    except Exception as e:
        logger.error("Impact criteria ranking test failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
