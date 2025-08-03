#!/usr/bin/env python3
"""
Script to process more Federal Register documents through the pipeline.
Downloads XML, chunks documents, and generates embeddings for more documents.
"""

from fr_monitor.orchestration.refactored_pipeline import RefactoredFederalRegisterPipeline

print('=== PROCESSING MORE DOCUMENTS ===\n')

pipeline = RefactoredFederalRegisterPipeline()

# Step 1: Download XML for more documents (ingestion includes XML download)
print('📥 Running ingestion to download XML for more documents...')
ingestion_results = pipeline.run_ingestion_only(download_limit=25)

print(f'✅ Ingestion completed:')
print(f'  Documents ingested: {ingestion_results["steps"]["rss_ingestion"]["documents_ingested"]}')
print(f'  XML downloaded: {ingestion_results["steps"]["xml_download"]["documents_downloaded"]}')

# Step 2: Process (chunk + embed) the newly downloaded documents
print('\n🔧 Running processing to chunk and embed documents...')
processing_results = pipeline.run_processing_only(chunk_limit=25, embed_limit=25)

print(f'✅ Processing completed:')
print(f'  Documents chunked: {processing_results["steps"]["chunking"]["documents_chunked"]}')
print(f'  Documents embedded: {processing_results["steps"]["embedding"]["documents_embedded"]}')

# Show final status
print('\n📊 Final Status:')
status = pipeline.get_status_report()
stats = status['pipeline_stats']
print(f'  Total documents: {stats["total_documents"]}')
print(f'  Documents with XML: {stats["xml_downloaded"]}')
print(f'  Documents chunked: {stats["chunked_documents"]}')
print(f'  Documents embedded: {stats["embedded_documents"]}')
print(f'  Processing rate: {stats["processing_rate"]:.1f}%')

print('\n=== Processing Complete ===')
