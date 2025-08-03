"""
BGE-large embedding integration with RedisSearch for Federal Register documents.
"""
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import redis
import requests

# Try to import RedisSearch components, fallback if not available
try:
    from redis.commands.search.field import VectorField, TextField, NumericField
    from redis.commands.search.indexdefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    REDISEARCH_AVAILABLE = True
except ImportError:
    REDISEARCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("RedisSearch not available - falling back to basic Redis operations")

from ..storage.database import FederalRegisterDB

logger = logging.getLogger(__name__)


class BGEEmbeddingClient:
    """Client for BGE-large embeddings using Ollama."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = "bge-large"
        self.embedding_dim = 1024  # BGE-large dimension
        
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        embeddings = []
        
        for text in texts:
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text
                    },
                    timeout=60
                )
                response.raise_for_status()
                
                result = response.json()
                embedding = result.get("embedding", [])
                
                if len(embedding) != self.embedding_dim:
                    logger.warning(f"Unexpected embedding dimension: {len(embedding)}")
                
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Failed to get embedding for text: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * self.embedding_dim)
        
        return embeddings
    
    def get_single_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        embeddings = self.get_embeddings([text])
        return embeddings[0] if embeddings else [0.0] * self.embedding_dim


class RedisVectorStore:
    """Redis-based vector store with search capabilities."""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, 
                 redis_db: int = 0, index_name: str = "federal_register_chunks"):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.index_name = index_name
        self.embedding_dim = 1024  # BGE-large dimension
        self.use_search = REDISEARCH_AVAILABLE
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        
        if self.use_search:
            self._create_index()
        else:
            logger.warning("RedisSearch not available - using basic Redis storage without vector search")
    
    def _create_index(self):
        """Create RedisSearch index for vector similarity search."""
        if not REDISEARCH_AVAILABLE:
            logger.warning("Cannot create search index - RedisSearch not available")
            return
            
        try:
            # Check if index already exists
            try:
                self.redis_client.ft(self.index_name).info()
                logger.info(f"Index {self.index_name} already exists")
                return
            except:
                pass  # Index doesn't exist, create it
            
            # Define schema
            schema = [
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.embedding_dim,
                        "DISTANCE_METRIC": "COSINE"
                    }
                ),
                TextField("document_number"),
                TextField("title"),
                TextField("agency"),
                TextField("chunk_type"),
                NumericField("chunk_level"),
                TextField("content"),
                TextField("xml_path"),
                NumericField("token_count"),
                NumericField("document_id"),
                NumericField("chunk_id")
            ]
            
            # Create index
            self.redis_client.ft(self.index_name).create_index(
                schema,
                definition=IndexDefinition(prefix=[f"{self.index_name}:"], index_type=IndexType.HASH)
            )
            
            logger.info(f"Created RedisSearch index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to create RedisSearch index: {e}")
            raise
    
    def store_chunk_embeddings(self, document_data: Dict[str, Any], 
                              chunks: List[Dict[str, Any]], 
                              embeddings: List[List[float]]) -> int:
        """Store document chunks with their embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        stored_count = 0
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            try:
                # Create Redis key
                key = f"{self.index_name}:{document_data['document_number']}:{chunk['id']}"
                
                # Prepare document for storage
                doc = {
                    "embedding": np.array(embedding, dtype=np.float32).tobytes(),
                    "document_number": document_data['document_number'],
                    "title": document_data.get('title', ''),
                    "agency": document_data.get('agency', ''),
                    "chunk_type": chunk['chunk_type'],
                    "chunk_level": chunk['chunk_level'],
                    "content": chunk['content'][:1000],  # Truncate for storage
                    "xml_path": chunk.get('xml_path', ''),
                    "token_count": chunk['token_count'],
                    "document_id": document_data['id'],
                    "chunk_id": chunk['id']
                }
                
                # Store in Redis
                self.redis_client.hset(key, mapping=doc)
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Failed to store chunk {chunk.get('id', i)}: {e}")
                continue
        
        logger.info(f"Stored {stored_count}/{len(chunks)} chunk embeddings for document {document_data['document_number']}")
        return stored_count
    
    def search_similar_chunks(self, query_embedding: List[float], 
                             limit: int = 10, 
                             filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity."""
        if not self.use_search:
            logger.warning("Vector search not available - RedisSearch required")
            return self._fallback_text_search("environmental regulation", limit)
            
        try:
            # Convert query embedding to bytes
            query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
            
            # Build query
            base_query = f"*=>[KNN {limit} @embedding $query_vector AS score]"
            
            # Add filters if provided
            if filters:
                filter_parts = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_parts.append(f"@{key}:{value}")
                    elif isinstance(value, list):
                        filter_parts.append(f"@{key}:({' | '.join(map(str, value))})")
                
                if filter_parts:
                    base_query = f"({' '.join(filter_parts)})=>[KNN {limit} @embedding $query_vector AS score]"
            
            # Execute search
            query = Query(base_query).return_fields(
                "document_number", "title", "agency", "chunk_type", 
                "chunk_level", "content", "xml_path", "token_count",
                "document_id", "chunk_id", "score"
            ).sort_by("score").paging(0, limit).dialect(2)
            
            results = self.redis_client.ft(self.index_name).search(
                query, {"query_vector": query_vector}
            )
            
            # Process results
            chunks = []
            for doc in results.docs:
                chunk = {
                    "document_number": doc.document_number,
                    "title": doc.title,
                    "agency": doc.agency,
                    "chunk_type": doc.chunk_type,
                    "chunk_level": int(doc.chunk_level),
                    "content": doc.content,
                    "xml_path": doc.xml_path,
                    "token_count": int(doc.token_count),
                    "document_id": int(doc.document_id),
                    "chunk_id": int(doc.chunk_id),
                    "similarity_score": float(doc.score)
                }
                chunks.append(chunk)
            
            logger.info(f"Found {len(chunks)} similar chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def search_by_text(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search chunks by text content."""
        if not self.use_search:
            return self._fallback_text_search(query_text, limit)
            
        try:
            # Simple text search
            query = Query(query_text).return_fields(
                "document_number", "title", "agency", "chunk_type",
                "content", "document_id", "chunk_id"
            ).paging(0, limit)
            
            results = self.redis_client.ft(self.index_name).search(query)
            
            chunks = []
            for doc in results.docs:
                chunk = {
                    "document_number": doc.document_number,
                    "title": doc.title,
                    "agency": doc.agency,
                    "chunk_type": doc.chunk_type,
                    "content": doc.content,
                    "document_id": int(doc.document_id),
                    "chunk_id": int(doc.chunk_id)
                }
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    def _fallback_text_search(self, query_text: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback text search when RedisSearch is not available."""
        try:
            # Get all keys matching our pattern
            pattern = f"{self.index_name}:*"
            keys = self.redis_client.keys(pattern)
            
            chunks = []
            query_lower = query_text.lower()
            
            for key in keys[:limit * 3]:  # Get more keys to filter
                try:
                    doc = self.redis_client.hgetall(key)
                    if not doc:
                        continue
                    
                    # Convert bytes to strings
                    doc_str = {k.decode() if isinstance(k, bytes) else k: 
                              v.decode() if isinstance(v, bytes) else v 
                              for k, v in doc.items()}
                    
                    # Simple text matching
                    content = doc_str.get('content', '').lower()
                    title = doc_str.get('title', '').lower()
                    
                    if query_lower in content or query_lower in title:
                        chunk = {
                            "document_number": doc_str.get('document_number', ''),
                            "title": doc_str.get('title', ''),
                            "agency": doc_str.get('agency', ''),
                            "chunk_type": doc_str.get('chunk_type', ''),
                            "content": doc_str.get('content', ''),
                            "document_id": int(doc_str.get('document_id', 0)),
                            "chunk_id": int(doc_str.get('chunk_id', 0)),
                            "similarity_score": 0.5  # Placeholder score
                        }
                        chunks.append(chunk)
                        
                        if len(chunks) >= limit:
                            break
                            
                except Exception as e:
                    logger.warning(f"Error processing key {key}: {e}")
                    continue
            
            logger.info(f"Fallback search found {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index."""
        if not self.use_search:
            # Fallback stats for basic Redis
            try:
                pattern = f"{self.index_name}:*"
                keys = self.redis_client.keys(pattern)
                return {
                    "total_documents": len(keys),
                    "redisearch_available": False,
                    "storage_type": "basic_redis"
                }
            except Exception as e:
                logger.error(f"Failed to get basic stats: {e}")
                return {"redisearch_available": False}
        
        try:
            info = self.redis_client.ft(self.index_name).info()
            
            stats = {
                "total_documents": info.get("num_docs", 0),
                "index_size_mb": info.get("inverted_sz_mb", 0),
                "vector_index_size_mb": info.get("vector_index_sz_mb", 0),
                "total_inverted_index_blocks": info.get("num_records", 0),
                "redisearch_available": True
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"redisearch_available": True, "error": str(e)}


class DocumentEmbeddingProcessor:
    """Main processor for embedding Federal Register documents."""
    
    def __init__(self, db_path: str = "federal_register.db"):
        self.db = FederalRegisterDB(db_path)
        self.embedding_client = BGEEmbeddingClient()
        self.vector_store = RedisVectorStore()
    
    def process_unembedded_documents(self, limit: Optional[int] = None) -> int:
        """Process documents that haven't been embedded yet."""
        # Get processed documents (chunked but not embedded)
        processed_docs = self.db.get_unprocessed_documents(limit)
        
        if not processed_docs:
            logger.info("No documents need embedding processing")
            return 0
        
        logger.info(f"Processing embeddings for {len(processed_docs)} documents")
        
        processed_count = 0
        for doc in processed_docs:
            try:
                # Get document chunks
                chunks = self.db.get_document_chunks(doc['id'])
                
                if not chunks:
                    logger.warning(f"No chunks found for document {doc['document_number']}")
                    continue
                
                # Extract content for embedding
                chunk_texts = [chunk['content'] for chunk in chunks]
                
                # Get embeddings
                logger.info(f"Getting embeddings for {len(chunks)} chunks in document {doc['document_number']}")
                embeddings = self.embedding_client.get_embeddings(chunk_texts)
                
                # Store in vector database
                stored_count = self.vector_store.store_chunk_embeddings(doc, chunks, embeddings)
                
                if stored_count > 0:
                    processed_count += 1
                    logger.info(f"Successfully processed document {doc['document_number']}")
                
            except Exception as e:
                logger.error(f"Failed to process document {doc['document_number']}: {e}")
                continue
        
        logger.info(f"Successfully processed embeddings for {processed_count}/{len(processed_docs)} documents")
        return processed_count
    
    def search_documents(self, query: str, limit: int = 10, 
                        filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search documents using semantic similarity."""
        # Get query embedding
        query_embedding = self.embedding_client.get_single_embedding(query)
        
        # Search similar chunks
        results = self.vector_store.search_similar_chunks(
            query_embedding, limit=limit, filters=filters
        )
        
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        db_stats = self.db.get_stats()
        vector_stats = self.vector_store.get_index_stats()
        
        return {
            "database": db_stats,
            "vector_store": vector_stats,
            "embedding_model": {
                "model_name": self.embedding_client.model_name,
                "embedding_dimension": self.embedding_client.embedding_dim
            }
        }
