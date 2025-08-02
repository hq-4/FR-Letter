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
from ..core.cache import DocumentCache, DeltaProcessor
from ..core.security import CredentialManager, SecureEnvironment
from ..ingestion import FederalRegisterClient
from ..scoring import ImpactScorer
from ..embedding import OllamaEmbedder, RedisVectorStore
from ..summarization import DocumentChunker, LocalSummarizer, OpenRouterSummarizer
from ..publishing import MarkdownPublisher

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
        
        try:
            # Step 1: Get last processed date for delta processing
            last_processed = self.cache.get_last_processed_date()
            if last_processed:
                logger.info(f"Last processed date: {last_processed}")
            
            # Step 2: Fetch documents from Federal Register
            all_documents = self._ingest_documents(target_date)
            
            # Step 3: Apply delta processing to filter new/changed documents
            documents = self.delta_processor.get_new_documents(all_documents)
            
            if not documents:
                logger.info("No new documents to process")
                return PipelineRun(
                    run_id=run_id,
                    target_date=target_date,
                    documents_processed=0,
                    documents_selected=0,
                    start_time=start_time,
                    end_time=datetime.now(),
                    status="success",
                    publishing_results=[]
                )
            
            logger.info(f"Processing {len(documents)} new/changed documents")
            
            # Step 4: Score documents for impact
            scored_documents = self._score_and_rank_documents(documents)
            
            # Step 5: Select top documents
            top_documents = scored_documents[:self.top_documents_count]
            
            # Step 6: Process embeddings and summaries
            embeddings = self._generate_and_store_embeddings(top_documents)
            final_documents = self._rerank_by_similarity(top_documents, embeddings)
            chunks = self._chunk_documents(final_documents)
            consolidated_summaries = self._local_summarization(chunks)
            final_summaries = self._generate_final_summaries(consolidated_summaries)
            
            # Step 7: Generate final summary
            final_summary = final_summaries[0]
            
            # Step 8: Publish results
            publishing_results = self._publish_summaries(final_summaries)
            
            # Step 9: Update cache and processing state
            self._update_processing_state(documents, publishing_results)
            
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
