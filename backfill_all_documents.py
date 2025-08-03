#!/usr/bin/env python3
"""
Script to backfill ALL Federal Register documents through the pipeline.
Downloads XML for all documents that don't have it, chunks them, and generates embeddings.
"""

import time
from fr_monitor.orchestration.refactored_pipeline import RefactoredFederalRegisterPipeline

def backfill_all_documents():
    print('=== BACKFILLING ALL DOCUMENTS ===\n')
    
    pipeline = RefactoredFederalRegisterPipeline()
    
    # Get initial status
    initial_status = pipeline.get_status_report()
    initial_health = initial_status['pipeline_health']
    
    print(f'📊 Initial Status:')
    print(f'  Total documents: {initial_health["total_documents"]}')
    print(f'  XML download rate: {initial_health["xml_download_rate_percent"]}%')
    print(f'  Processing rate: {initial_health["processing_rate_percent"]}%')
    print(f'  Documents ready for search: {initial_health["documents_ready_for_search"]}')
    print()
    
    # Step 1: Download XML for ALL documents that don't have it
    print('📥 Downloading XML for all documents...')
    start_time = time.time()
    
    # Run ingestion multiple times to download all XML content
    # We'll keep running until no more documents are downloaded
    total_downloaded = 0
    batch_count = 0
    max_batches = 20  # Safety limit to prevent infinite loops
    
    while batch_count < max_batches:
        batch_count += 1
        print(f'  Batch {batch_count}...')
        
        ingestion_results = pipeline.run_ingestion_only()
        downloaded_in_batch = ingestion_results["steps"]["xml_download"]["documents_downloaded"]
        total_downloaded += downloaded_in_batch
        
        print(f'    XML downloaded in batch: {downloaded_in_batch}')
        
        # If no documents were downloaded in this batch, we're done
        if downloaded_in_batch == 0:
            print('    No more documents to download - moving to processing')
            break
        
        # Small delay between batches
        time.sleep(1)
    
    print(f'✅ XML download completed: {total_downloaded} documents downloaded in {batch_count} batches\n')
    
    # Step 2: Process (chunk + embed) ALL documents with XML content
    print('🔧 Processing all documents with XML content...')
    processing_results = pipeline.run_processing_only()
    
    chunked = processing_results["steps"].get("xml_chunking", {}).get("documents_chunked", 0)
    embedded = processing_results["steps"].get("embedding_generation", {}).get("documents_embedded", 0)
    
    print(f'✅ Processing completed:')
    print(f'  Documents chunked: {chunked}')
    print(f'  Documents embedded: {embedded}')
    print()
    
    # Show final status
    print('📊 Final Status:')
    final_status = pipeline.get_status_report()
    final_health = final_status['pipeline_health']
    
    print(f'  Total documents: {final_health["total_documents"]}')
    print(f'  XML download rate: {final_health["xml_download_rate_percent"]}%')
    print(f'  Processing rate: {final_health["processing_rate_percent"]}%')
    print(f'  Documents ready for search: {final_health["documents_ready_for_search"]}')
    
    # Show processing time
    total_time = time.time() - start_time
    print(f'  Total processing time: {total_time:.2f} seconds')
    
    print('\n=== Backfill Complete ===')

if __name__ == "__main__":
    backfill_all_documents()
