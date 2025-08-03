"""
BGE-large embedding integration with RedisSearch for Federal Register documents.
"""
import json
import redis
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import requests
from ..storage.database import FederalRegisterDB

# RedisSearch imports
try:
    from redis.commands.search import Search
    from redis.commands.search.field import VectorField, TextField, NumericField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    REDISEARCH_IMPORTS_AVAILABLE = True
except ImportError:
    REDISEARCH_IMPORTS_AVAILABLE = False

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
                # Truncate text if too long (BGE models have token limits)
                truncated_text = text[:8000] if len(text) > 8000 else text
                
                response = requests.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": truncated_text
                    },
                    timeout=60
                )
                response.raise_for_status()
                
                result = response.json()
                embedding = result.get("embedding", [])
                
                if not embedding:
                    logger.error(f"Empty embedding returned for text: {truncated_text[:100]}...")
                    embeddings.append([0.0] * self.embedding_dim)
                    continue
                
                if len(embedding) != self.embedding_dim:
                    logger.warning(f"Unexpected embedding dimension: {len(embedding)}, expected {self.embedding_dim}")
                    # Pad or truncate to expected dimension
                    if len(embedding) < self.embedding_dim:
                        embedding.extend([0.0] * (self.embedding_dim - len(embedding)))
                    else:
                        embedding = embedding[:self.embedding_dim]
                
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Failed to get embedding for text: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * self.embedding_dim)
        
        return embeddings
    
    def generate_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Backward-compatibility alias for get_embeddings."""
        return self.get_embeddings(texts)

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
        
        # Check if RedisSearch is available
        try:
            # Test RedisSearch by checking module list and imports
            modules = self.redis_client.execute_command("MODULE", "LIST")
            search_available = any(module[1] == b'search' or module[1] == 'search' for module in modules)
            
            if search_available and REDISEARCH_IMPORTS_AVAILABLE:
                self.redisearch_available = True
                logger.info("RedisSearch module is available and loaded")
            else:
                self.redisearch_available = False
                if not search_available:
                    logger.warning("RedisSearch module not found in Redis")
                if not REDISEARCH_IMPORTS_AVAILABLE:
                    logger.warning("RedisSearch Python imports not available")
        except Exception as e:
            self.redisearch_available = False
            logger.warning(f"RedisSearch not available: {e}")
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        
        if self.redisearch_available:
            self._create_index()
        else:
            logger.warning("RedisSearch not available - using basic Redis storage without vector search")
    
    def _create_index(self):
        """Create RedisSearch index for vector similarity search."""
        if not self.redisearch_available:
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
        if not self.redisearch_available:
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
        """Search chunks by text content using semantic similarity."""
        if not self.redisearch_available:
            return self._fallback_text_search(query_text, limit)
            
        try:
            # Convert text to embedding for semantic search
            from ..embeddings.bge_embeddings import BGEEmbeddingClient
            embedding_client = BGEEmbeddingClient()
            query_embeddings = embedding_client.get_embeddings([query_text])
            
            if not query_embeddings or not query_embeddings[0]:
                logger.error(f"Failed to get embedding for query: {query_text}")
                return self._fallback_text_search(query_text, limit)
            
            # Use vector search
            return self.search_similar_chunks(query_embeddings[0], limit)
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            # Fallback to simple text search
            return self._fallback_text_search(query_text, limit)
    
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
                    
                    # Convert bytes to strings, skip binary fields like embeddings
                    doc_str = {}
                    for k, v in doc.items():
                        key = k.decode() if isinstance(k, bytes) else k
                        
                        # Skip binary embedding field
                        if key == 'embedding':
                            continue
                            
                        # Safely decode values
                        if isinstance(v, bytes):
                            try:
                                value = v.decode('utf-8')
                            except UnicodeDecodeError:
                                # Skip fields that can't be decoded (likely binary data)
                                continue
                        else:
                            value = v
                        
                        doc_str[key] = value
                    
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
        if not self.redisearch_available:
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
        # Get documents that have been chunked but not embedded yet
        processed_docs = self._get_documents_needing_embedding(limit)
        
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
    
    def _get_documents_needing_embedding(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get documents that have chunks but haven't been embedded yet."""
        # Simple approach: get all documents that have chunks
        # We'll check Redis keys to see if they've been embedded
        query = """
            SELECT DISTINCT d.id, d.document_number, d.title, d.agency, d.publication_date,
                   d.rss_link, d.xml_url, d.xml_content, d.xml_size
            FROM documents d
            INNER JOIN document_chunks dc ON d.id = dc.document_id
            ORDER BY d.publication_date DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        with self.db.get_connection() as conn:
            cursor = conn.execute(query)
            all_docs = [dict(row) for row in cursor.fetchall()]
        
        # Filter out documents that already have embeddings in Redis
        docs_needing_embedding = []
        for doc in all_docs:
            # Check if this document has any embeddings in Redis
            pattern = f"{self.vector_store.index_name}:{doc['document_number']}:*"
            existing_keys = self.vector_store.redis_client.keys(pattern)
            
            if not existing_keys:
                docs_needing_embedding.append(doc)
        
        return docs_needing_embedding
    
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
