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
if 'xml_chunking' in processing_results["steps"]:
    print(f'  Documents chunked: {processing_results["steps"]["xml_chunking"]["documents_chunked"]}')
else:
    print(f'  Documents chunked: No chunking step found')
    
if 'embedding_generation' in processing_results["steps"]:
    print(f'  Documents embedded: {processing_results["steps"]["embedding_generation"]["documents_embedded"]}')
else:
    print(f'  Documents embedded: No embedding step found')

# Show final status
print('\n📊 Final Status:')
status = pipeline.get_status_report()
health = status['pipeline_health']
print(f'  Total documents: {health["total_documents"]}')
print(f'  XML download rate: {health["xml_download_rate_percent"]}%')
print(f'  Processing rate: {health["processing_rate_percent"]}%')
print(f'  Documents ready for search: {health["documents_ready_for_search"]}')

print('\n=== Processing Complete ===')
