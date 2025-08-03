#!/usr/bin/env python3
"""
Test RSS ingestion with debug logging enabled.
"""
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fr_monitor.ingestion.rss_ingestion import RSSIngestionClient

# Configure debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_rss_ingestion():
    """Test RSS ingestion with debug output."""
    print("=== Testing RSS Ingestion with Debug Logging ===\n")
    
    client = RSSIngestionClient()
    
    try:
        # Test just the RSS fetch and parse (no database storage)
        print("Fetching RSS feed...")
        ingested_count = client.fetch_and_store_rss()
        print(f"Result: {ingested_count} documents ingested")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rss_ingestion()
