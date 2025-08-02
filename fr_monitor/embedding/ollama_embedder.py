"""
Ollama-based embedding generation for Federal Register documents.
"""

import requests
from typing import List, Optional, Dict, Any
import structlog
import numpy as np
from datetime import datetime

from ..core.models import FederalRegisterDocument, DocumentEmbedding
from ..core.config import settings

logger = structlog.get_logger(__name__)


class OllamaEmbedder:
    """Generates embeddings using local Ollama instance."""
    
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None):
        self.host = host or settings.ollama_host
        self.model = model or settings.embedding_model
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
    
    def generate_embeddings(
        self, 
        documents: List[FederalRegisterDocument],
        text_source: str = "title"
    ) -> List[DocumentEmbedding]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of Federal Register documents
            text_source: Source of text to embed ("title", "abstract", or "full_text")
            
        Returns:
            List of DocumentEmbedding objects
        """
        embeddings = []
        
        logger.info("Generating embeddings", 
                   count=len(documents),
                   model=self.model,
                   text_source=text_source)
        
        for doc in documents:
            try:
                text = self._extract_text(doc, text_source)
                if not text:
                    logger.warning("No text found for embedding", 
                                 document_id=doc.document_id,
                                 text_source=text_source)
                    continue
                
                embedding_vector = self._generate_single_embedding(text)
                if embedding_vector:
                    embedding = DocumentEmbedding(
                        document_id=doc.document_id,
                        embedding=embedding_vector,
                        embedding_model=self.model,
                        text_source=text_source
                    )
                    embeddings.append(embedding)
                    
            except Exception as e:
                logger.error("Failed to generate embedding", 
                           document_id=doc.document_id,
                           error=str(e))
        
        logger.info("Generated embeddings", 
                   successful=len(embeddings),
                   total=len(documents))
        
        return embeddings
    
    def _extract_text(self, document: FederalRegisterDocument, source: str) -> Optional[str]:
        """Extract text from document based on source preference."""
        if source == "title":
            return document.title
        elif source == "abstract":
            return document.abstract
        elif source == "full_text":
            # For full text, we'd need to fetch it separately
            # For now, fall back to abstract or title
            return document.abstract or document.title
        else:
            return document.title
    
    def _generate_single_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text string."""
        try:
            payload = {
                "model": self.model,
                "prompt": text,
                "stream": False
            }
            
            response = self.session.post(
                f"{self.host}/api/embeddings",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            embedding = data.get("embedding")
            
            if embedding and isinstance(embedding, list):
                return embedding
            else:
                logger.error("Invalid embedding response format", 
                           response_keys=list(data.keys()) if data else None)
                return None
                
        except requests.RequestException as e:
            logger.error("Failed to generate embedding via Ollama", 
                        error=str(e),
                        host=self.host,
                        model=self.model)
            return None
    
    def calculate_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error("Failed to calculate similarity", error=str(e))
            return 0.0
    
    def calculate_centroid(self, embeddings: List[List[float]]) -> Optional[List[float]]:
        """Calculate centroid (average) of multiple embeddings."""
        if not embeddings:
            return None
        
        try:
            embedding_matrix = np.array(embeddings)
            centroid = np.mean(embedding_matrix, axis=0)
            return centroid.tolist()
            
        except Exception as e:
            logger.error("Failed to calculate centroid", error=str(e))
            return None
    
    def rank_by_similarity(
        self,
        documents: List[FederalRegisterDocument],
        embeddings: List[DocumentEmbedding],
        reference_embedding: List[float],
        top_n: int = 5
    ) -> List[FederalRegisterDocument]:
        """
        Rank documents by similarity to a reference embedding.
        
        Args:
            documents: List of documents to rank
            embeddings: List of document embeddings
            reference_embedding: Reference embedding to compare against
            top_n: Number of top documents to return
            
        Returns:
            List of top-ranked documents by similarity
        """
        # Create mapping of document_id to embedding
        embedding_map = {emb.document_id: emb.embedding for emb in embeddings}
        
        # Calculate similarities
        similarities = []
        for doc in documents:
            if doc.document_id in embedding_map:
                doc_embedding = embedding_map[doc.document_id]
                similarity = self.calculate_similarity(doc_embedding, reference_embedding)
                similarities.append((doc, similarity))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N documents
        top_documents = [doc for doc, _ in similarities[:top_n]]
        
        logger.info("Ranked documents by similarity", 
                   total_documents=len(documents),
                   ranked_documents=len(similarities),
                   top_n=top_n)
        
        return top_documents
    
    def health_check(self) -> bool:
        """Check if Ollama service is available and model is loaded."""
        try:
            # Check if Ollama is running
            response = self.session.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            if self.model in model_names:
                logger.info("Ollama health check passed", 
                           host=self.host,
                           model=self.model)
                return True
            else:
                logger.warning("Model not found in Ollama", 
                             model=self.model,
                             available_models=model_names)
                return False
                
        except Exception as e:
            logger.error("Ollama health check failed", 
                        host=self.host,
                        error=str(e))
            return False
