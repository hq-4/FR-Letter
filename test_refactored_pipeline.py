#!/usr/bin/env python3
"""
Test script for the refactored Federal Register pipeline.
"""
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fr_monitor.orchestration.refactored_pipeline import RefactoredFederalRegisterPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_pipeline_steps():
    """Test individual pipeline steps."""
    print("=== Testing Refactored Federal Register Pipeline ===\n")
    
    # Initialize pipeline
    pipeline = RefactoredFederalRegisterPipeline()
    
    # Get initial status
    print("1. Initial Pipeline Status:")
    status = pipeline.get_status_report()
    print(f"   Total documents: {status['pipeline_health']['total_documents']}")
    print(f"   XML download rate: {status['pipeline_health']['xml_download_rate_percent']}%")
    print(f"   Processing rate: {status['pipeline_health']['processing_rate_percent']}%")
    print(f"   Next actions: {status['next_actions']}")
    print()
    
    # Test RSS ingestion (limit to 5 documents for testing)
    print("2. Testing RSS Ingestion:")
    ingestion_result = pipeline.run_ingestion_only(download_limit=5)
    print(f"   Success: {ingestion_result['success']}")
    if ingestion_result['success']:
        print(f"   Documents ingested: {ingestion_result['steps']['rss_ingestion']['documents_ingested']}")
        print(f"   Documents downloaded: {ingestion_result['steps']['xml_download']['documents_downloaded']}")
    else:
        print(f"   Errors: {ingestion_result['errors']}")
    print()
    
    # Test processing (limit to 3 documents for testing)
    print("3. Testing Document Processing:")
    processing_result = pipeline.run_processing_only(chunk_limit=3, embed_limit=3)
    print(f"   Success: {processing_result['success']}")
    if processing_result['success']:
        print(f"   Documents chunked: {processing_result['steps']['xml_chunking']['documents_chunked']}")
        print(f"   Documents embedded: {processing_result['steps']['embedding_generation']['documents_embedded']}")
    else:
        print(f"   Errors: {processing_result['errors']}")
    print()
    
    # Test search functionality
    print("4. Testing Search Functionality:")
    try:
        # Test environmental search for NY/NJ
        results = pipeline.search_environmental_ny_nj("air quality emissions", limit=3)
        print(f"   Found {len(results)} relevant chunks")
        
        for i, result in enumerate(results[:2], 1):
            print(f"   Result {i}:")
            print(f"     Document: {result['document_number']}")
            print(f"     Agency: {result['agency']}")
            print(f"     Chunk Type: {result['chunk_type']}")
            print(f"     Similarity Score: {result.get('similarity_score', 'N/A')}")
            print(f"     Content Preview: {result['content'][:100]}...")
            print()
    except Exception as e:
        print(f"   Search failed: {e}")
    
    # Final status
    print("5. Final Pipeline Status:")
    final_status = pipeline.get_status_report()
    print(f"   Total documents: {final_status['pipeline_health']['total_documents']}")
    print(f"   Documents ready for search: {final_status['pipeline_health']['documents_ready_for_search']}")
    print(f"   Processing rate: {final_status['pipeline_health']['processing_rate_percent']}%")
    print()
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    test_pipeline_steps()
