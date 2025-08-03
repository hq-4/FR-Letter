#!/usr/bin/env python3
"""
Test script to verify Federal Register data ingestion.
"""

import sys
import os
from datetime import date, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fr_monitor.ingestion.federal_register import FederalRegisterClient

def test_federal_register_ingestion():
    """Test fetching documents from Federal Register API."""
    print("Testing Federal Register API ingestion...")
    
    client = FederalRegisterClient()
    
    # Test with yesterday's date (today might not have documents yet)
    target_date = date.today() - timedelta(days=1)
    print(f"Fetching documents for: {target_date}")
    
    try:
        documents = client.get_daily_documents(target_date=target_date)
        
        print(f"✅ Successfully fetched {len(documents)} documents")
        
        if documents:
            # Show details of first few documents
            for i, doc in enumerate(documents[:3]):
                print(f"\n--- Document {i+1} ---")
                print(f"ID: {doc.document_id}")
                print(f"Title: {doc.title}")
                print(f"Type: {doc.document_type}")
                print(f"Agencies: {[agency.name for agency in doc.agencies]}")
                print(f"Abstract length: {len(doc.abstract) if doc.abstract else 0} chars")
                print(f"Page length: {doc.page_length}")
                print(f"HTML URL: {doc.html_url}")
        else:
            print("⚠️  No documents found for this date")
            
    except Exception as e:
        print(f"❌ Error fetching documents: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_federal_register_ingestion()
    sys.exit(0 if success else 1)
