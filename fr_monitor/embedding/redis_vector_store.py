"""
Redis-based vector storage with RediSearch integration.
"""

import redis
try:
    # Try new import structure (redis-py 4.0+)
    from redis.commands.search.field import VectorField, TextField, NumericField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    REDISEARCH_AVAILABLE = True
except ImportError:
    try:
        # Fallback for different redis-py versions
        from redis.search.field import VectorField, TextField, NumericField
        from redis.search.indexDefinition import IndexDefinition, IndexType
        from redis.search.query import Query
        REDISEARCH_AVAILABLE = True
    except ImportError:
        # RediSearch not available - we'll use basic Redis operations
        REDISEARCH_AVAILABLE = False
        logger.warning("RediSearch not available, using basic Redis operations")
        
        # Define dummy classes for compatibility
        class VectorField:
            def __init__(self, name, algorithm, attributes):
                pass
        
        class TextField:
            def __init__(self, name):
                pass
        
        class NumericField:
            def __init__(self, name):
                pass
        
        class IndexDefinition:
            def __init__(self, index_type=None, prefix=None):
                pass
        
        class IndexType:
            HASH = "HASH"
        
        class Query:
            def __init__(self, query_string):
                pass
import json
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import structlog
from datetime import datetime

from ..core.models import DocumentEmbedding, FederalRegisterDocument
from ..core.config import settings

logger = structlog.get_logger(__name__)


class RedisVectorStore:
    """Redis-based vector storage with search capabilities."""
    
    def __init__(self, 
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 password: Optional[str] = None,
                 db: Optional[int] = None):
        
        self.host = host or settings.redis_host
        self.port = port or settings.redis_port
        self.password = password or settings.redis_password
        self.db = db or settings.redis_db
        
        self.redis_client = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password,
            db=self.db,
            decode_responses=True
        )
        
        self.index_name = "document_embeddings"
        self.key_prefix = "doc:"
        
        # Check if RediSearch is available
        self.search_available = REDISEARCH_AVAILABLE
        
        if self.search_available:
            # Initialize search index if RediSearch is available
            self._create_search_index()
        else:
            logger.warning("RediSearch not available, using basic Redis operations")
    
    def _create_search_index(self) -> None:
        """Create RediSearch index for document embeddings."""
        if not self.search_available:
            logger.warning("RediSearch not available, skipping index creation")
            return
            
        try:
            # Check if index already exists
            self.redis_client.ft(self.index_name).info()
            logger.info(f"Index {self.index_name} already exists")
            return
        except redis.ResponseError:
            # Index doesn't exist, create it
            pass
        
        # Define schema for document embeddings
        schema = (
            VectorField("embedding", "FLAT", {
                "TYPE": "FLOAT32",
                "DIM": 384,  # Dimension of qwen2:1.5b embeddings
                "DISTANCE_METRIC": "COSINE"
            }),
            TextField("document_id"),
            TextField("title"),
            TextField("summary"),
            TextField("agency"),
            NumericField("impact_score"),
            NumericField("published_date"),
            TextField("document_type")
        )
        
        definition = IndexDefinition(index_type=IndexType.HASH, prefix=[self.key_prefix])
        
        try:
            self.redis_client.ft(self.index_name).create_index(
                fields=schema,
                definition=definition
            )
            
            logger.info("Created search index", index=self.index_name)
            
        except Exception as e:
            logger.error("Failed to create search index", error=str(e))
            raise
    
    def store_embeddings(self, 
                        embeddings: List[DocumentEmbedding],
                        documents: Optional[List[FederalRegisterDocument]] = None) -> None:
        """
        Store document embeddings in Redis.
        
        Args:
            embeddings: List of document embeddings
            documents: Optional list of corresponding documents for metadata
        """
        doc_map = {}
        if documents:
            doc_map = {doc.document_id: doc for doc in documents}
        
        pipeline = self.redis_client.pipeline()
        
        for embedding in embeddings:
            key = f"{self.key_prefix}{embedding.document_id}"
            
            # Convert embedding to bytes for storage
            embedding_bytes = np.array(embedding.embedding, dtype=np.float32).tobytes()
            
            # Prepare document data
            doc_data = {
                "document_id": embedding.document_id,
                "embedding": embedding_bytes,
                "embedding_model": embedding.embedding_model,
                "text_source": embedding.text_source,
                "created_at": embedding.created_at.isoformat()
            }
            
            # Add document metadata if available
            if embedding.document_id in doc_map:
                doc = doc_map[embedding.document_id]
                doc_data.update({
                    "title": doc.title or "",
                    "abstract": doc.abstract or "",
                    "agency": ",".join([agency.abbreviation for agency in doc.agencies]),
                    "publication_date": int(doc.publication_date.timestamp()),
                    "document_type": doc.document_type.value
                })
            
            pipeline.hset(key, mapping=doc_data)
        
        pipeline.execute()
        
        logger.info("Stored embeddings in Redis", count=len(embeddings))
    
    def get_embedding(self, document_id: str) -> Optional[DocumentEmbedding]:
        """Retrieve embedding for a specific document."""
        key = f"{self.key_prefix}{document_id}"
        
        try:
            data = self.redis_client.hgetall(key)
            if not data:
                return None
            
            # Convert bytes back to embedding vector
            embedding_bytes = self.redis_client.hget(key, "embedding")
            if not embedding_bytes:
                return None
            
            embedding_vector = np.frombuffer(
                embedding_bytes, 
                dtype=np.float32
            ).tolist()
            
            return DocumentEmbedding(
                document_id=data["document_id"],
                embedding=embedding_vector,
                embedding_model=data["embedding_model"],
                text_source=data["text_source"],
                created_at=datetime.fromisoformat(data["created_at"])
            )
            
        except Exception as e:
            logger.error("Failed to retrieve embedding", 
                        document_id=document_id,
                        error=str(e))
            return None
    
    def search_similar_documents(self, 
                                query_embedding: List[float],
                                top_k: int = 10,
                                filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of similar documents to return
            filters: Optional filters (e.g., agency, date range)
            
        Returns:
            List of tuples (document_id, similarity_score)
        """
        try:
            # Convert query embedding to bytes
            query_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            
            # Build search query
            base_query = f"*=>[KNN {top_k} @embedding $query_vec AS score]"
            
            # Add filters if provided
            if filters:
                filter_parts = []
                for field, value in filters.items():
                    if field == "agency" and value:
                        filter_parts.append(f"@agency:{value}")
                    elif field == "min_date" and value:
                        filter_parts.append(f"@publication_date:[{int(value.timestamp())} +inf]")
                    elif field == "max_date" and value:
                        filter_parts.append(f"@publication_date:[-inf {int(value.timestamp())}]")
                
                if filter_parts:
                    base_query = f"({' '.join(filter_parts)})=>[KNN {top_k} @embedding $query_vec AS score]"
            
            query = Query(base_query).return_fields("document_id", "score").sort_by("score").paging(0, top_k)
            
            # Execute search
            results = self.redis_client.ft(self.index_name).search(
                query,
                query_params={"query_vec": query_bytes}
            )
            
            # Extract results
            similar_docs = []
            for doc in results.docs:
                doc_id = doc.document_id
                score = float(doc.score)
                similar_docs.append((doc_id, score))
            
            logger.info("Vector search completed", 
                       query_results=len(similar_docs),
                       top_k=top_k)
            
            return similar_docs
            
        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            return []
    
    def calculate_recent_impact_centroid(self, 
                                       days_back: int = 30,
                                       min_impact_score: float = 0.7) -> Optional[List[float]]:
        """
        Calculate centroid of recent high-impact documents.
        
        Args:
            days_back: Number of days to look back
            min_impact_score: Minimum impact score threshold
            
        Returns:
            Centroid embedding vector or None
        """
        try:
            # Calculate timestamp threshold
            from datetime import datetime, timedelta
            threshold_date = datetime.now() - timedelta(days=days_back)
            threshold_timestamp = int(threshold_date.timestamp())
            
            # Search for recent high-impact documents
            query = Query(f"@publication_date:[{threshold_timestamp} +inf] @impact_score:[{min_impact_score} +inf]")
            results = self.redis_client.ft(self.index_name).search(query)
            
            if not results.docs:
                logger.warning("No recent high-impact documents found for centroid calculation")
                return None
            
            # Collect embeddings
            embeddings = []
            for doc in results.docs:
                doc_id = doc.document_id
                embedding_data = self.get_embedding(doc_id)
                if embedding_data:
                    embeddings.append(embedding_data.embedding)
            
            if not embeddings:
                return None
            
            # Calculate centroid
            embedding_matrix = np.array(embeddings)
            centroid = np.mean(embedding_matrix, axis=0)
            
            logger.info("Calculated recent impact centroid", 
                       documents_used=len(embeddings),
                       days_back=days_back,
                       min_impact_score=min_impact_score)
            
            return centroid.tolist()
            
        except Exception as e:
            logger.error("Failed to calculate recent impact centroid", error=str(e))
            return None
    
    def update_impact_scores(self, scores: Dict[str, float]) -> None:
        """Update impact scores for stored documents."""
        pipeline = self.redis_client.pipeline()
        
        for document_id, score in scores.items():
            key = f"{self.key_prefix}{document_id}"
            pipeline.hset(key, "impact_score", score)
        
        pipeline.execute()
        
        logger.info("Updated impact scores", count=len(scores))
    
    def cleanup_old_documents(self, days_to_keep: int = 90) -> int:
        """Remove old document embeddings to save space."""
        try:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_timestamp = int(cutoff_date.timestamp())
            
            # Find old documents
            query = Query(f"@publication_date:[-inf {cutoff_timestamp}]")
            results = self.redis_client.ft(self.index_name).search(query)
            
            # Delete old documents
            pipeline = self.redis_client.pipeline()
            for doc in results.docs:
                key = f"{self.key_prefix}{doc.document_id}"
                pipeline.delete(key)
            
            pipeline.execute()
            
            deleted_count = len(results.docs)
            logger.info("Cleaned up old documents", 
                       deleted_count=deleted_count,
                       days_to_keep=days_to_keep)
            
            return deleted_count
            
        except Exception as e:
            logger.error("Failed to cleanup old documents", error=str(e))
            return 0
    
    def health_check(self) -> bool:
        """Check Redis connection and index health."""
        try:
            # Test Redis connection
            self.redis_client.ping()
            
            # Check index exists
            self.redis_client.ft(self.index_name).info()
            
            logger.info("Redis vector store health check passed")
            return True
            
        except Exception as e:
            logger.error("Redis vector store health check failed", error=str(e))
            return False
