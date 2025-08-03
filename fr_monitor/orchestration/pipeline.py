"""
Main pipeline orchestrator for Federal Register monitoring system.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
import structlog
import uuid
import os
from datetime import datetime, date

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fr_monitor.core.models import (
    FederalRegisterDocument, ImpactScore, DocumentEmbedding,
    DocumentChunk, ChunkSummary, ConsolidatedSummary, FinalSummary,
    PipelineRun, PublishingResult, ProcessedArticle
)
from fr_monitor.core.config import settings
from fr_monitor.core.cache import DocumentCache, DeltaProcessor
from fr_monitor.core.security import CredentialManager, SecureEnvironment
from fr_monitor.ingestion.rss_wrapper import FederalRegisterClient
from fr_monitor.scoring import ImpactScorer
from fr_monitor.embeddings.bge_embeddings import OllamaEmbedder
from fr_monitor.embeddings.redis_vector_store import RedisVectorStore
from fr_monitor.summarization import DocumentChunker, LocalSummarizer, OpenRouterSummarizer
from fr_monitor.publishing import MarkdownPublisher

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
        self.markdown_publisher = MarkdownPublisher()
        
        # Initialize caching and security
        self.cache = DocumentCache(self.vector_store.redis_client)
        self.delta_processor = DeltaProcessor(self.cache)
        self.credential_manager = CredentialManager()
        self.secure_env = SecureEnvironment()
        
        # Pipeline configuration
        self.top_documents_count = 20
        self.final_summary_count = 5
        self.pipeline_timeout = settings.pipeline_timeout_minutes * 60
        
        # Log current working directory and .env location
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Loading .env file from: {Path('.env').absolute()}")
    
    def run_daily_pipeline(self, target_date: Optional[date] = None) -> PipelineRun:
        """
        Execute the complete daily pipeline with caching and delta processing.
        
        Args:
            target_date: Date to process documents for. Defaults to today.
            
        Returns:
            PipelineRun: Results of the pipeline execution.
        """
        if target_date is None:
            target_date = date.today()
        
        run_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Validate environment and credentials
        security_check = self.secure_env.validate_environment()
        if not security_check['environment_valid']:
            logger.error("Security validation failed", issues=security_check)
            
        logger.info("Starting daily pipeline", run_id=run_id, target_date=target_date)
        
        pipeline_run = PipelineRun(
            run_id=run_id,
            start_time=start_time
        )
        
        try:
            # Step 1: Get last processed date for delta processing
            last_processed = self.cache.get_last_processed_date()
            if last_processed:
                logger.info(f"Last processed date: {last_processed}")
            
            # Step 2: Fetch documents from Federal Register
            all_documents = self._ingest_documents(target_date)
            logger.info(f"Ingested {len(all_documents)} total documents")
            
            # Step 3: Apply delta processing to filter new/changed documents
            documents = self.delta_processor.get_new_documents(all_documents)
            logger.info(f"After delta processing: {len(documents)} new/changed documents")
            
            if not documents:
                logger.info("No new documents to process")
                pipeline_run.end_time = datetime.utcnow()
                pipeline_run.status = "completed"
                pipeline_run.documents_processed = 0
                pipeline_run.documents_summarized = 0
                return pipeline_run
            
            logger.info(f"Processing {len(documents)} new/changed documents")
            
            # Step 4: Score and rank documents by impact
            top_documents = self._score_and_rank_documents(documents)
            logger.info(f"Top documents after scoring: {len(top_documents)}")
            
            # Step 5: Generate embeddings and store in Redis
            embeddings = self._generate_and_store_embeddings(top_documents)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Step 6: Re-rank by similarity to recent impact centroid
            reranked_documents = self._rerank_by_similarity(top_documents, embeddings)
            logger.info(f"Final documents after re-ranking: {len(reranked_documents)}")
            
            # Step 7: Chunk documents for summarization
            chunks = self._chunk_documents(reranked_documents)
            logger.info(f"Generated {len(chunks)} document chunks")
            
            # Step 8: Local summarization of chunks
            chunk_summaries = self._local_summarization(chunks)
            logger.info(f"Generated {len(chunk_summaries)} chunk summaries")
            
            # Step 9: Consolidate summaries by document
            consolidated_summaries = self.local_summarizer.consolidate_summaries(chunk_summaries)
            logger.info(f"Consolidated into {len(consolidated_summaries)} document summaries")
            
            # Log summary content lengths for debugging
            for i, summary in enumerate(consolidated_summaries):
                content_length = len(summary.consolidated_summary) if summary.consolidated_summary else 0
                logger.info(f"Summary {i+1}: title='{summary.document_title}', content_length={content_length}")
            
            # Step 10: Generate final summaries using OpenRouter
            final_summaries = self._generate_final_summaries(consolidated_summaries)
            logger.info(f"Generated {len(final_summaries)} final summaries")
            
            # Step 11: Publish summaries
            publishing_results = self._publish_summaries(final_summaries)
            logger.info(f"Publishing results: {[r.success for r in publishing_results]}")
            
            # Update processing state
            self._update_processing_state(documents, publishing_results)
            
            # Mark pipeline as successful
            pipeline_run.end_time = datetime.utcnow()
            pipeline_run.status = "completed"
            pipeline_run.documents_processed = len(documents)
            pipeline_run.documents_summarized = len(final_summaries)
            
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
    
    def _ingest_documents(self, target_date: datetime) -> List[FederalRegisterDocument]:
        """Step 1: Ingest documents from Federal Register API."""
        documents = self.fr_client.get_daily_documents(target_date)
        return documents
    
    def _score_and_rank_documents(self, documents: List[FederalRegisterDocument]) -> List[FederalRegisterDocument]:
        """Step 2: Score and rank documents by impact."""
        top_documents = self.impact_scorer.get_top_documents(documents, self.top_documents_count)
        return top_documents
    
    def _generate_and_store_embeddings(self, documents: List[FederalRegisterDocument]) -> List[DocumentEmbedding]:
        """Step 3: Generate embeddings and store in Redis."""
        # Generate embeddings
        embeddings = self.embedder.generate_embeddings(documents, text_source="abstract")
        
        # Store in Redis with document metadata
        self.vector_store.store_embeddings(embeddings, documents)
        
        # Update impact scores in vector store
        scores = self.impact_scorer.score_documents(documents)
        score_dict = {score.document_id: score.total_score for score in scores}
        self.vector_store.update_impact_scores(score_dict)
        
        return embeddings
    
    def _rerank_by_similarity(self, documents: List[FederalRegisterDocument], embeddings: List[DocumentEmbedding]) -> List[FederalRegisterDocument]:
        """Step 4: Re-rank by similarity to recent impact centroid."""
        # Calculate recent impact centroid
        centroid = self.vector_store.calculate_recent_impact_centroid()
        
        if centroid:
            # Re-rank by similarity to centroid
            final_documents = self.embedder.rank_by_similarity(
                documents, embeddings, centroid, self.final_summary_count
            )
        else:
            # Fall back to top N by impact score if no centroid available
            final_documents = documents[:self.final_summary_count]
        
        return final_documents
    
    def _chunk_documents(self, documents: List[FederalRegisterDocument]) -> List[DocumentChunk]:
        """Step 5: Chunk documents for summarization."""
        chunks = self.chunker.chunk_documents(documents)
        return chunks
    
    def _local_summarization(self, chunks: List[DocumentChunk]) -> List[ConsolidatedSummary]:
        """Step 6: Local summarization of chunks."""
        # Summarize individual chunks
        chunk_summaries = self.local_summarizer.summarize_chunks(chunks)
        
        # Consolidate summaries by document
        consolidated_summaries = self.local_summarizer.consolidate_summaries(chunk_summaries)
        
        return consolidated_summaries
    
    def _generate_final_summaries(self, consolidated_summaries: List[ConsolidatedSummary]) -> List[FinalSummary]:
        """Step 7: Generate final summaries using OpenRouter."""
        final_summaries = self.openrouter_summarizer.generate_final_summaries(consolidated_summaries)
        
        
        return final_summaries
    
    def _publish_summaries(self, summaries: List[FinalSummary]) -> List[PublishingResult]:
        """Publish summaries as a single markdown file for the day."""
        if not summaries:
            return []

        article_date = datetime.now()
        results = []

        try:
            # Combine content from all summaries into a single string
            full_content = []
            for summary in summaries:
                full_content.append(f"### {summary.headline}")
                full_content.extend([f"- {bullet}" for bullet in summary.bullets])
                full_content.append("\n")
            
            article_content = "\n".join(full_content)

            # Create a single article for the day's summaries
            article = ProcessedArticle(
                title=f"Federal Register Summary for {article_date.strftime('%Y-%m-%d')}",
                content=article_content,
                summary=f"A summary of {len(summaries)} significant documents published in the Federal Register.",
                date=article_date,
                source_url="https://www.federalregister.gov"
            )

            # Publish the combined article
            success = self.markdown_publisher.publish(article)
            
            results.append(PublishingResult(
                platform="markdown",
                success=success,
                published_at=datetime.now() if success else None,
                error_message=None if success else "Failed to write markdown file."
            ))

            if success:
                logger.info("Successfully published daily summary to markdown.")
            else:
                logger.error("Failed to publish daily summary to markdown.")

        except Exception as e:
            logger.error("An exception occurred during markdown publishing", error=str(e))
            results.append(PublishingResult(
                platform="markdown",
                success=False,
                error_message=str(e)
            ))

        return results
    
    def _update_processing_state(self, documents: List[FederalRegisterDocument], 
                               publishing_results: List[PublishingResult]) -> None:
        """Update cache and processing state after successful pipeline run."""
        try:
            # Cache processed documents
            for doc in documents:
                self.cache.cache_document(doc.document_number, doc.dict())
                
                # Cache processing state
                self.cache.cache_processing_state(doc.document_number, {
                    "status": "completed",
                    "processed_at": datetime.utcnow().isoformat(),
                    "publishing_results": [r.dict() for r in publishing_results]
                })
                
                # Cache content hash for deduplication
                content_hash = self.cache.generate_content_hash(
                    json.dumps(doc.dict(), sort_keys=True)
                )
                self.cache.cache_content_hash(content_hash, doc.document_number)
            
            logger.info("Updated processing state for all documents")
            
        except Exception as e:
            logger.error("Failed to update processing state", error=str(e))
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all pipeline components."""
        health_status = {
            "federal_register_api": True,  # Basic HTTP connectivity
            "ollama_embeddings": self.embedder.health_check(),
            "ollama_summarization": self.local_summarizer.health_check(),
            "redis_vector_store": self.vector_store.health_check(),
            "openrouter_api": self.openrouter_summarizer.health_check(),
            "markdown_publisher": self.markdown_publisher.health_check()
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
