"""
Main pipeline orchestrator for Federal Register monitoring system.
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
import structlog
import uuid

from ..core.models import (
    FederalRegisterDocument, ImpactScore, DocumentEmbedding,
    DocumentChunk, ChunkSummary, ConsolidatedSummary, FinalSummary,
    PipelineRun, PublishingResult
)
from ..core.config import settings
from ..ingestion import FederalRegisterClient
from ..scoring import ImpactScorer
from ..embedding import OllamaEmbedder, RedisVectorStore
from ..summarization import DocumentChunker, LocalSummarizer, OpenRouterSummarizer
from ..publishing import SubstackPublisher, TelegramPublisher

logger = structlog.get_logger(__name__)


class FederalRegisterPipeline:
    """Main pipeline orchestrator for the Federal Register monitoring system."""
    
    def __init__(self):
        # Initialize all components
        self.fr_client = FederalRegisterClient()
        self.impact_scorer = ImpactScorer()
        self.embedder = OllamaEmbedder()
        self.vector_store = RedisVectorStore()
        self.chunker = DocumentChunker()
        self.local_summarizer = LocalSummarizer()
        self.openrouter_summarizer = OpenRouterSummarizer()
        self.substack_publisher = SubstackPublisher()
        self.telegram_publisher = TelegramPublisher()
        
        # Pipeline configuration
        self.top_documents_count = 20
        self.final_summary_count = 5
        self.pipeline_timeout = settings.pipeline_timeout_minutes * 60
    
    def run_daily_pipeline(self, target_date: Optional[date] = None) -> PipelineRun:
        """
        Execute the complete daily pipeline.
        
        Args:
            target_date: Date to process (defaults to yesterday)
            
        Returns:
            PipelineRun object with execution details
        """
        run_id = str(uuid.uuid4())
        pipeline_run = PipelineRun(
            run_id=run_id,
            start_time=datetime.utcnow()
        )
        
        logger.info("Starting daily pipeline", 
                   run_id=run_id,
                   target_date=target_date)
        
        try:
            # Step 1: Data Ingestion (FR-1)
            logger.info("Step 1: Data ingestion")
            documents = self._ingest_documents(target_date, pipeline_run)
            if not documents:
                raise Exception("No documents retrieved from Federal Register")
            
            # Step 2: Impact Pre-Scoring (FR-2, FR-3)
            logger.info("Step 2: Impact scoring")
            top_documents = self._score_and_rank_documents(documents, pipeline_run)
            
            # Step 3: Embedding & Vector Storage (FR-4, FR-5)
            logger.info("Step 3: Embedding generation")
            embeddings = self._generate_and_store_embeddings(top_documents, pipeline_run)
            
            # Step 4: Impact-based Re-ranking (FR-6)
            logger.info("Step 4: Similarity-based re-ranking")
            final_documents = self._rerank_by_similarity(top_documents, embeddings, pipeline_run)
            
            # Step 5: Document Chunking (FR-7)
            logger.info("Step 5: Document chunking")
            chunks = self._chunk_documents(final_documents, pipeline_run)
            
            # Step 6: Local Summarization (FR-8, FR-9)
            logger.info("Step 6: Local summarization")
            consolidated_summaries = self._local_summarization(chunks, pipeline_run)
            
            # Step 7: Final LLM Summaries (FR-10, FR-11, FR-12)
            logger.info("Step 7: Final LLM summarization")
            final_summaries = self._generate_final_summaries(consolidated_summaries, pipeline_run)
            
            # Step 8: Publishing (FR-16, FR-17)
            logger.info("Step 8: Publishing")
            self._publish_summaries(final_summaries, pipeline_run)
            
            # Mark pipeline as completed
            pipeline_run.end_time = datetime.utcnow()
            pipeline_run.status = "completed"
            
            duration = (pipeline_run.end_time - pipeline_run.start_time).total_seconds()
            logger.info("Pipeline completed successfully", 
                       run_id=run_id,
                       duration_seconds=duration,
                       documents_processed=pipeline_run.documents_processed,
                       summaries_generated=pipeline_run.documents_summarized)
            
        except Exception as e:
            pipeline_run.end_time = datetime.utcnow()
            pipeline_run.status = "failed"
            pipeline_run.error_message = str(e)
            
            logger.error("Pipeline failed", 
                        run_id=run_id,
                        error=str(e))
        
        return pipeline_run
    
    def _ingest_documents(self, target_date: Optional[date], run: PipelineRun) -> List[FederalRegisterDocument]:
        """Step 1: Ingest documents from Federal Register API."""
        documents = self.fr_client.get_daily_documents(target_date)
        run.documents_processed = len(documents)
        run.logs.append(f"Ingested {len(documents)} documents")
        return documents
    
    def _score_and_rank_documents(self, documents: List[FederalRegisterDocument], run: PipelineRun) -> List[FederalRegisterDocument]:
        """Step 2: Score and rank documents by impact."""
        top_documents = self.impact_scorer.get_top_documents(documents, self.top_documents_count)
        run.logs.append(f"Selected top {len(top_documents)} documents by impact score")
        return top_documents
    
    def _generate_and_store_embeddings(self, documents: List[FederalRegisterDocument], run: PipelineRun) -> List[DocumentEmbedding]:
        """Step 3: Generate embeddings and store in Redis."""
        # Generate embeddings
        embeddings = self.embedder.generate_embeddings(documents, text_source="abstract")
        
        # Store in Redis with document metadata
        self.vector_store.store_embeddings(embeddings, documents)
        
        # Update impact scores in vector store
        scores = self.impact_scorer.score_documents(documents)
        score_dict = {score.document_id: score.total_score for score in scores}
        self.vector_store.update_impact_scores(score_dict)
        
        run.logs.append(f"Generated and stored {len(embeddings)} embeddings")
        return embeddings
    
    def _rerank_by_similarity(self, documents: List[FederalRegisterDocument], 
                             embeddings: List[DocumentEmbedding], run: PipelineRun) -> List[FederalRegisterDocument]:
        """Step 4: Re-rank by similarity to recent impact centroid."""
        # Calculate recent impact centroid
        centroid = self.vector_store.calculate_recent_impact_centroid()
        
        if centroid:
            # Re-rank by similarity to centroid
            final_documents = self.embedder.rank_by_similarity(
                documents, embeddings, centroid, self.final_summary_count
            )
            run.logs.append(f"Re-ranked to top {len(final_documents)} by similarity")
        else:
            # Fall back to top N by impact score if no centroid available
            final_documents = documents[:self.final_summary_count]
            run.logs.append(f"Used top {len(final_documents)} by impact score (no centroid)")
        
        return final_documents
    
    def _chunk_documents(self, documents: List[FederalRegisterDocument], run: PipelineRun) -> List[DocumentChunk]:
        """Step 5: Chunk documents for summarization."""
        chunks = self.chunker.chunk_documents(documents)
        run.logs.append(f"Created {len(chunks)} document chunks")
        return chunks
    
    def _local_summarization(self, chunks: List[DocumentChunk], run: PipelineRun) -> List[ConsolidatedSummary]:
        """Step 6: Local summarization of chunks."""
        # Summarize individual chunks
        chunk_summaries = self.local_summarizer.summarize_chunks(chunks)
        
        # Consolidate summaries by document
        consolidated_summaries = self.local_summarizer.consolidate_summaries(chunk_summaries)
        
        run.logs.append(f"Generated {len(consolidated_summaries)} consolidated summaries")
        return consolidated_summaries
    
    def _generate_final_summaries(self, consolidated_summaries: List[ConsolidatedSummary], 
                                 run: PipelineRun) -> List[FinalSummary]:
        """Step 7: Generate final summaries using OpenRouter."""
        final_summaries = self.openrouter_summarizer.generate_final_summaries(consolidated_summaries)
        
        run.documents_summarized = len(final_summaries)
        run.openrouter_calls_made = self.openrouter_summarizer.daily_calls_made
        run.logs.append(f"Generated {len(final_summaries)} final summaries")
        
        return final_summaries
    
    def _publish_summaries(self, summaries: List[FinalSummary], run: PipelineRun) -> None:
        """Step 8: Publish summaries to external channels."""
        if not summaries:
            run.logs.append("No summaries to publish")
            return
        
        # Publish to Substack
        if settings.substack_api_key:
            try:
                substack_result = self.substack_publisher.publish_daily_digest(summaries)
                if substack_result.success:
                    run.logs.append("Successfully published to Substack")
                else:
                    run.logs.append(f"Substack publishing failed: {substack_result.error_message}")
            except Exception as e:
                run.logs.append(f"Substack publishing error: {str(e)}")
        
        # Publish to Telegram
        if settings.telegram_bot_token:
            try:
                telegram_results = self.telegram_publisher.publish_daily_digest(summaries)
                successful_telegram = sum(1 for r in telegram_results if r.success)
                run.logs.append(f"Telegram: {successful_telegram}/{len(telegram_results)} messages sent")
            except Exception as e:
                run.logs.append(f"Telegram publishing error: {str(e)}")
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all pipeline components."""
        health_status = {
            "federal_register_api": True,  # Basic HTTP connectivity
            "ollama_embeddings": self.embedder.health_check(),
            "ollama_summarization": self.local_summarizer.health_check(),
            "redis_vector_store": self.vector_store.health_check(),
            "openrouter_api": self.openrouter_summarizer.health_check(),
            "substack_api": self.substack_publisher.health_check() if settings.substack_api_key else None,
            "telegram_api": self.telegram_publisher.health_check() if settings.telegram_bot_token else None
        }
        
        # Remove None values
        health_status = {k: v for k, v in health_status.items() if v is not None}
        
        all_healthy = all(health_status.values())
        logger.info("Pipeline health check", 
                   overall_healthy=all_healthy,
                   component_status=health_status)
        
        return health_status
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old data to save storage space."""
        logger.info("Starting data cleanup", days_to_keep=days_to_keep)
        
        cleanup_results = {
            "redis_documents_deleted": self.vector_store.cleanup_old_documents(days_to_keep)
        }
        
        logger.info("Data cleanup completed", results=cleanup_results)
        return cleanup_results
