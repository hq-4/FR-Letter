"""
Main entry point for the Federal Register monitoring system.
"""

import sys
import argparse
from datetime import datetime, date
from pathlib import Path
import structlog

from .core.config import settings
from .orchestration import FederalRegisterPipeline


def setup_logging():
    """Configure structured logging."""
    # Ensure logs directory exists
    settings.logs_dir.mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Federal Register Monitoring & Summarization System"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run pipeline command
    run_parser = subparsers.add_parser('run', help='Run the daily pipeline')
    run_parser.add_argument(
        '--date', 
        type=str, 
        help='Date to process (YYYY-MM-DD, defaults to yesterday)'
    )
    run_parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='Run pipeline without publishing'
    )
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Check system health')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data')
    cleanup_parser.add_argument(
        '--days', 
        type=int, 
        default=90, 
        help='Days of data to keep (default: 90)'
    )
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test individual components')
    test_parser.add_argument(
        'component',
        choices=['ingestion', 'scoring', 'embedding', 'summarization', 'publishing'],
        help='Component to test'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = structlog.get_logger(__name__)
    
    # Initialize pipeline
    pipeline = FederalRegisterPipeline()
    
    if args.command == 'run':
        # Parse date if provided
        target_date = None
        if args.date:
            try:
                target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
            except ValueError:
                logger.error("Invalid date format. Use YYYY-MM-DD")
                sys.exit(1)
        
        # Run pipeline
        logger.info("Starting pipeline execution", 
                   target_date=target_date,
                   dry_run=args.dry_run)
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No publishing will occur")
            # TODO: Implement dry run mode
        
        pipeline_run = pipeline.run_daily_pipeline(target_date)
        
        if pipeline_run.status == "completed":
            logger.info("Pipeline completed successfully")
            sys.exit(0)
        else:
            logger.error("Pipeline failed", error=pipeline_run.error_message)
            sys.exit(1)
    
    elif args.command == 'health':
        logger.info("Running health checks")
        health_status = pipeline.health_check()
        
        print("\n=== System Health Check ===")
        for component, status in health_status.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {component.replace('_', ' ').title()}: {'OK' if status else 'FAILED'}")
        
        all_healthy = all(health_status.values())
        print(f"\nOverall Status: {'✅ HEALTHY' if all_healthy else '❌ ISSUES DETECTED'}")
        
        sys.exit(0 if all_healthy else 1)
    
    elif args.command == 'cleanup':
        logger.info("Starting data cleanup", days_to_keep=args.days)
        results = pipeline.cleanup_old_data(args.days)
        
        print(f"\n=== Data Cleanup Results ===")
        for operation, count in results.items():
            print(f"• {operation.replace('_', ' ').title()}: {count}")
        
        logger.info("Cleanup completed", results=results)
    
    elif args.command == 'test':
        logger.info("Testing component", component=args.component)
        
        if args.component == 'ingestion':
            test_ingestion(pipeline)
        elif args.component == 'scoring':
            test_scoring(pipeline)
        elif args.component == 'embedding':
            test_embedding(pipeline)
        elif args.component == 'summarization':
            test_summarization(pipeline)
        elif args.component == 'publishing':
            test_publishing(pipeline)
    
    else:
        parser.print_help()
        sys.exit(1)


def test_ingestion(pipeline):
    """Test document ingestion."""
    print("Testing Federal Register API ingestion...")
    
    try:
        documents = pipeline.fr_client.get_daily_documents(per_page=5)
        print(f"✅ Successfully retrieved {len(documents)} documents")
        
        if documents:
            doc = documents[0]
            print(f"Sample document: {doc.title[:100]}...")
            print(f"Document type: {doc.document_type}")
            print(f"Agencies: {[a.name for a in doc.agencies]}")
    
    except Exception as e:
        print(f"❌ Ingestion test failed: {e}")


def test_scoring(pipeline):
    """Test impact scoring."""
    print("Testing impact scoring...")
    
    try:
        # Get a few documents to test scoring
        documents = pipeline.fr_client.get_daily_documents(per_page=3)
        if not documents:
            print("❌ No documents available for scoring test")
            return
        
        scores = pipeline.impact_scorer.score_documents(documents)
        print(f"✅ Successfully scored {len(scores)} documents")
        
        for score in scores[:3]:
            print(f"Document {score.document_id}: {score.total_score:.3f}")
    
    except Exception as e:
        print(f"❌ Scoring test failed: {e}")


def test_embedding(pipeline):
    """Test embedding generation."""
    print("Testing embedding generation...")
    
    try:
        # Test Ollama health first
        if not pipeline.embedder.health_check():
            print("❌ Ollama service not available")
            return
        
        # Test with sample documents
        documents = pipeline.fr_client.get_daily_documents(per_page=2)
        if not documents:
            print("❌ No documents available for embedding test")
            return
        
        embeddings = pipeline.embedder.generate_embeddings(documents[:1])
        print(f"✅ Successfully generated {len(embeddings)} embeddings")
        
        if embeddings:
            emb = embeddings[0]
            print(f"Embedding dimension: {len(emb.embedding)}")
            print(f"Model used: {emb.embedding_model}")
    
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")


def test_summarization(pipeline):
    """Test summarization components."""
    print("Testing summarization...")
    
    try:
        # Test local summarizer health
        if not pipeline.local_summarizer.health_check():
            print("❌ Local summarizer (Ollama) not available")
            return
        
        # Test OpenRouter health
        if not pipeline.openrouter_summarizer.health_check():
            print("❌ OpenRouter API not available")
            return
        
        print("✅ Both local and remote summarization services available")
    
    except Exception as e:
        print(f"❌ Summarization test failed: {e}")


def test_publishing(pipeline):
    """Test publishing components."""
    print("Testing publishing...")
    
    # Test Substack
    if settings.substack_api_key:
        if pipeline.substack_publisher.health_check():
            print("✅ Substack API connection OK")
        else:
            print("❌ Substack API connection failed")
    else:
        print("⚠️  Substack API key not configured")
    
    # Test Telegram
    if settings.telegram_bot_token:
        if pipeline.telegram_publisher.health_check():
            print("✅ Telegram bot connection OK")
        else:
            print("❌ Telegram bot connection failed")
    else:
        print("⚠️  Telegram bot token not configured")


if __name__ == "__main__":
    main()
