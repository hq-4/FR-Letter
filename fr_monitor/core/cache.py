"""
Caching mechanism for Federal Register documents and processing state.
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import redis
import structlog
from .config import settings

logger = structlog.get_logger(__name__)


class DocumentCache:
    """Cache for Federal Register documents and processing state."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.document_ttl = 86400 * 7  # 7 days
        self.processing_ttl = 86400 * 1  # 1 day
        
        # Cache keys
        self.document_key = "fr:document:{document_id}"
        self.processing_key = "fr:processing:{document_id}"
        self.delta_key = "fr:delta:last_processed"
        self.cache_hash_key = "fr:cache:hash:{content_hash}"
    
    def cache_document(self, document_id: str, document_data: Dict[str, Any]) -> bool:
        """Cache a Federal Register document."""
        try:
            key = self.document_key.format(document_id=document_id)
            document_data['cached_at'] = datetime.utcnow().isoformat()
            self.redis.setex(key, self.document_ttl, json.dumps(document_data))
            return True
        except Exception as e:
            logger.error(f"Failed to cache document {document_id}", error=str(e))
            return False
    
    def get_cached_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached document."""
        try:
            key = self.document_key.format(document_id=document_id)
            data = self.redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve cached document {document_id}", error=str(e))
            return None
    
    def is_document_cached(self, document_id: str) -> bool:
        """Check if a document is cached."""
        return self.redis.exists(self.document_key.format(document_id=document_id))
    
    def cache_processing_state(self, document_id: str, state: Dict[str, Any]) -> bool:
        """Cache processing state for a document."""
        try:
            key = self.processing_key.format(document_id=document_id)
            state['updated_at'] = datetime.utcnow().isoformat()
            self.redis.setex(key, self.processing_ttl, json.dumps(state))
            return True
        except Exception as e:
            logger.error(f"Failed to cache processing state {document_id}", error=str(e))
            return False
    
    def get_processing_state(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve processing state for a document."""
        try:
            key = self.processing_key.format(document_id=document_id)
            data = self.redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve processing state {document_id}", error=str(e))
            return None
    
    def set_last_processed_date(self, date: datetime) -> bool:
        """Set the last processed date for delta processing."""
        try:
            self.redis.setex(self.delta_key, 86400 * 30, date.isoformat())  # 30 days
            return True
        except Exception as e:
            logger.error("Failed to set last processed date", error=str(e))
            return False
    
    def get_last_processed_date(self) -> Optional[datetime]:
        """Get the last processed date for delta processing."""
        try:
            date_str = self.redis.get(self.delta_key)
            if date_str:
                return datetime.fromisoformat(date_str.decode())
            return None
        except Exception as e:
            logger.error("Failed to get last processed date", error=str(e))
            return None
    
    def generate_content_hash(self, content: str) -> str:
        """Generate a hash for content deduplication."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_content_cached(self, content_hash: str) -> bool:
        """Check if content is already cached."""
        key = self.cache_hash_key.format(content_hash=content_hash)
        return self.redis.exists(key)
    
    def cache_content_hash(self, content_hash: str, document_id: str) -> bool:
        """Cache content hash for deduplication."""
        try:
            key = self.cache_hash_key.format(content_hash=content_hash)
            self.redis.setex(key, self.document_ttl, document_id)
            return True
        except Exception as e:
            logger.error(f"Failed to cache content hash", error=str(e))
            return False
    
    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries."""
        try:
            # Scan for expired keys and delete them
            pattern = "fr:*"
            keys = []
            cursor = 0
            
            while True:
                cursor, batch_keys = self.redis.scan(cursor, match=pattern, count=100)
                keys.extend(batch_keys)
                if cursor == 0:
                    break
            
            if keys:
                deleted = self.redis.delete(*keys)
                logger.info(f"Cleaned up {deleted} expired cache entries")
                return deleted
            return 0
        except Exception as e:
            logger.error("Failed to cleanup expired cache", error=str(e))
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            pattern = "fr:*"
            keys = []
            cursor = 0
            
            while True:
                cursor, batch_keys = self.redis.scan(cursor, match=pattern, count=100)
                keys.extend(batch_keys)
                if cursor == 0:
                    break
            
            return {
                "total_keys": len(keys),
                "document_keys": len([k for k in keys if k.startswith("fr:document:")]),
                "processing_keys": len([k for k in keys if k.startswith("fr:processing:")]),
                "cache_hash_keys": len([k for k in keys if k.startswith("fr:cache:hash:")])
            }
        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return {"error": str(e)}


class DeltaProcessor:
    """Delta processing for incremental document updates."""
    
    def __init__(self, cache: DocumentCache):
        self.cache = cache
        self.redis = cache.redis
        
    def get_new_documents(self, all_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out already processed documents."""
        new_docs = []
        
        for doc in all_documents:
            doc_id = doc.get('document_number') or doc.get('id')
            if not doc_id:
                continue
                
            # Check if document is already cached
            if not self.cache.is_document_cached(doc_id):
                new_docs.append(doc)
                continue
            
            # Check if content has changed
            content_hash = self.cache.generate_content_hash(
                json.dumps(doc, sort_keys=True)
            )
            
            if not self.cache.is_content_cached(content_hash):
                new_docs.append(doc)
        
        return new_docs
    
    def mark_document_processed(self, document_id: str, content_hash: str) -> bool:
        """Mark a document as processed."""
        success1 = self.cache.cache_content_hash(content_hash, document_id)
        success2 = self.cache.cache_processing_state(document_id, {
            "status": "processed",
            "processed_at": datetime.utcnow().isoformat()
        })
        return success1 and success2
    
    def get_documents_since(self, date: datetime) -> List[str]:
        """Get document IDs processed since a specific date."""
        try:
            pattern = "fr:document:*"
            keys = []
            cursor = 0
            
            while True:
                cursor, batch_keys = self.redis.scan(cursor, match=pattern, count=100)
                keys.extend(batch_keys)
                if cursor == 0:
                    break
            
            recent_docs = []
            for key in keys:
                data = self.redis.get(key)
                if data:
                    doc = json.loads(data)
                    cached_at = datetime.fromisoformat(doc.get('cached_at', '1970-01-01T00:00:00'))
                    if cached_at >= date:
                        doc_id = key.decode().split(':')[-1]
                        recent_docs.append(doc_id)
            
            return recent_docs
        except Exception as e:
            logger.error("Failed to get documents since date", error=str(e))
            return []
