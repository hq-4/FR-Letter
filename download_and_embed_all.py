#!/usr/bin/env python3
"""
Script to download XML for all remaining documents and embed them.
"""

import time
import logging
from fr_monitor.orchestration.refactored_pipeline import RefactoredFederalRegisterPipeline
from fr_monitor.storage.database import FederalRegisterDB

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_all_xml():
    """Download XML for all documents that don't have it yet."""
    print('=== DOWNLOADING ALL XML DOCUMENTS ===\n')
    
    db = FederalRegisterDB()
    pipeline = RefactoredFederalRegisterPipeline()
    
    # Get documents without XML content
    with db.get_connection() as conn:
        cursor = conn.execute("""
            SELECT id, document_number, xml_url 
            FROM documents 
            WHERE xml_content IS NULL
            ORDER BY publication_date DESC
        """)
        documents = [dict(row) for row in cursor.fetchall()]
    
    total_documents = len(documents)
    print(f'Found {total_documents} documents without XML content')
    
    if total_documents == 0:
        print('No documents need XML content download')
        return 0
    
    downloaded_count = 0
    failed_count = 0
    
    for i, doc in enumerate(documents):
        try:
            print(f'Processing document {i+1}/{total_documents}: {doc["document_number"]}')
            xml_content = pipeline.rss_client._download_xml(doc['xml_url'])
            if xml_content:
                db.update_document_xml(doc['id'], xml_content)
                downloaded_count += 1
                print(f'  ✅ Downloaded XML for document {doc["document_number"]}')
            else:
                failed_count += 1
                print(f'  ❌ Failed to download XML for {doc["document_number"]}')
                
        except Exception as e:
            failed_count += 1
            print(f'  ❌ Error downloading XML for {doc["document_number"]}: {e}')
            continue
    
    print(f'\n✅ XML download completed:')
    print(f'  Successfully downloaded: {downloaded_count}')
    print(f'  Failed downloads: {failed_count}')
    print(f'  Total processed: {downloaded_count + failed_count}')
    
    return downloaded_count

def embed_all_documents():
    """Embed all documents that have XML but haven't been embedded yet."""
    print('\n=== EMBEDDING ALL PROCESSED DOCUMENTS ===\n')
    
    pipeline = RefactoredFederalRegisterPipeline()
    
    # Process all documents with XML content
    embedded_count = pipeline.run_processing_only()
    
    print(f'✅ Embedding completed:')
    print(f'  Documents chunked: {embedded_count["steps"].get("xml_chunking", {}).get("documents_chunked", 0)}')
    print(f'  Documents embedded: {embedded_count["steps"].get("embedding_generation", {}).get("documents_embedded", 0)}')
    
    return embedded_count

def main():
    start_time = time.time()
    
    # Download all XML content
    downloaded = download_all_xml()
    
    # Embed all documents
    embedded = embed_all_documents()
    
    # Show final status
    print('\n📊 Final Status:')
    pipeline = RefactoredFederalRegisterPipeline()
    status = pipeline.get_status_report()
    health = status['pipeline_health']
    
    print(f'  Total documents: {health["total_documents"]}')
    print(f'  XML download rate: {health["xml_download_rate_percent"]}%')
    print(f'  Processing rate: {health["processing_rate_percent"]}%')
    print(f'  Documents ready for search: {health["documents_ready_for_search"]}')
    
    # Show processing time
    total_time = time.time() - start_time
    print(f'  Total processing time: {total_time:.2f} seconds')

if __name__ == "__main__":
    main()
