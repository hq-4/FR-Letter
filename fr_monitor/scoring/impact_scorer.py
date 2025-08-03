"""
Impact scoring system for Federal Register documents.
"""

import math
from typing import List, Dict, Any
from datetime import datetime
import structlog

from ..core.models import FederalRegisterDocument, ImpactScore, DocumentType
from ..core.config import scoring_config

logger = structlog.get_logger(__name__)


class ImpactScorer:
    """Calculates impact scores for Federal Register documents."""
    
    def __init__(self):
        self.weights = scoring_config.weights
    
    def score_documents(self, documents: List[FederalRegisterDocument]) -> List[ImpactScore]:
        """
        Calculate impact scores for a list of documents.
        
        Args:
            documents: List of Federal Register documents
            
        Returns:
            List of ImpactScore objects, sorted by total_score descending
        """
        scores = []
        
        for doc in documents:
            try:
                score = self._calculate_impact_score(doc)
                scores.append(score)
            except Exception as e:
                logger.warning("Failed to score document", 
                             document_id=doc.document_id,
                             error=str(e))
        
        # Sort by total score descending
        scores.sort(key=lambda x: x.total_score, reverse=True)
        
        logger.info("Calculated impact scores", 
                   total_documents=len(documents),
                   scored_documents=len(scores))
        
        return scores
    
    def get_top_documents(
        self, 
        documents: List[FederalRegisterDocument], 
        top_n: int = 20
    ) -> List[FederalRegisterDocument]:
        """
        Get the top N documents by impact score.
        
        Args:
            documents: List of Federal Register documents
            top_n: Number of top documents to return
            
        Returns:
            List of top-scoring documents
        """
        scores = self.score_documents(documents)
        top_scores = scores[:top_n]
        
        # Create a mapping of document_id to score for quick lookup
        score_map = {score.document_id: score for score in top_scores}
        
        # Filter and sort documents by their scores
        top_documents = []
        for doc in documents:
            if doc.document_id in score_map:
                top_documents.append(doc)
        
        # Sort documents by their impact scores
        top_documents.sort(
            key=lambda doc: score_map[doc.document_id].total_score,
            reverse=True
        )
        
        logger.info("Selected top documents", 
                   requested=top_n,
                   returned=len(top_documents))
        
        return top_documents
    
    def _calculate_impact_score(self, document: FederalRegisterDocument) -> ImpactScore:
        """Calculate impact score for a single document."""
        
        # Calculate component scores
        agency_score = self._calculate_agency_score(document)
        length_score = self._calculate_length_score(document)
        type_score = self._calculate_type_score(document)
        
        # Calculate weighted total score
        total_score = self._calculate_total_score(
            agency_score, length_score, type_score
        )
        
        return ImpactScore(
            document_id=document.document_id,
            total_score=total_score,
            agency_score=agency_score,
            length_score=length_score,
            type_score=type_score
        )
    
    def _calculate_agency_score(self, document: FederalRegisterDocument) -> float:
        """Calculate score based on agency importance."""
        if not document.agencies:
            return 0.0
        
        agency_weights = self.weights.get("agency_importance", {"default": 0.5})
        max_score = 0.0
        
        for agency in document.agencies:
            # Get agency-specific weight or default
            agency_weight = agency_weights.get(
                agency.abbreviation,
                agency_weights.get("default", 0.5)
            )
            max_score = max(max_score, agency_weight)
        
        return min(max_score, 1.0)
    
    def _calculate_length_score(self, document: FederalRegisterDocument) -> float:
        """Calculate score based on document length."""
        if not document.page_length:
            return 0.0
        
        # Normalize page length using logarithmic scale
        # Longer documents generally have higher impact
        length_weight = self.weights.get("document_length_weight", 0.7)
        
        # Use log scale to prevent very long documents from dominating
        # Assume 1-100 pages is typical range
        normalized_length = min(math.log(document.page_length + 1) / math.log(101), 1.0)
        
        return normalized_length * length_weight
    
    def _calculate_type_score(self, document: FederalRegisterDocument) -> float:
        """Calculate score based on document type and characteristics."""
        base_score = 0.0
        
        # Base score by document type
        type_scores = {
            DocumentType.EXECUTIVE_ORDER: 0.9,
            DocumentType.FINAL_RULE: 0.8,
            DocumentType.PROPOSED_RULE: 0.6,
            DocumentType.PRESIDENTIAL_DOCUMENT: 0.7,
            DocumentType.NOTICE: 0.3
        }
        
        base_score = type_scores.get(document.document_type, 0.3)
        
        # Apply bonuses for special characteristics
        if document.is_final_rule:
            base_score += self.weights.get("final_rule_bonus", 0.1)
        
        if document.is_major_rule:
            base_score += self.weights.get("major_rule_bonus", 0.1)
        
        if document.document_type == DocumentType.EXECUTIVE_ORDER:
            base_score += self.weights.get("executive_order_bonus", 0.1)
        
        return min(base_score, 1.0)
    
    def _calculate_total_score(
        self, 
        agency_score: float, 
        length_score: float, 
        type_score: float
    ) -> float:
        """Calculate weighted total impact score."""
        
        # Weighted combination of component scores
        # Agency and type are most important, length provides additional context
        total = (
            agency_score * 0.4 +  # Agency importance: 40%
            type_score * 0.5 +    # Document type: 50%
            length_score * 0.1    # Length: 10%
        )
        
        return min(total, 1.0)
    
    def update_weights(self, new_weights: Dict[str, Any]) -> None:
        """Update scoring weights configuration."""
        self.weights.update(new_weights)
        logger.info("Updated scoring weights", weights=new_weights)
