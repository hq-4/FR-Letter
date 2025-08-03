#!/usr/bin/env python3
"""
Simple test script to verify RSS feed approach works.
"""

import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import re

def test_rss_feed():
    """Test the Federal Register RSS feed directly."""
    print("🔍 Testing Federal Register RSS Feed")
    print("=" * 50)
    
    RSS_URL = "https://www.federalregister.gov/api/v1/documents.rss"
    
    try:
        print("📡 Fetching RSS feed...")
        response = requests.get(RSS_URL, timeout=30)
        response.raise_for_status()
        
        print(f"✅ RSS feed fetched successfully ({len(response.content)} bytes)")
        
        # Parse XML
        root = ET.fromstring(response.content)
        items = root.findall('.//item')
        
        print(f"📄 Found {len(items)} documents in RSS feed")
        
        # Show first few items
        for i, item in enumerate(items[:5]):
            title_elem = item.find('title')
            link_elem = item.find('link')
            pub_date_elem = item.find('pubDate')
            creator_elem = item.find('.//{http://purl.org/dc/elements/1.1/}creator')
            
            title = title_elem.text if title_elem is not None else "No title"
            link = link_elem.text if link_elem is not None else "No link"
            pub_date = pub_date_elem.text if pub_date_elem is not None else "No date"
            creator = creator_elem.text if creator_elem is not None else "No creator"
            
            # Extract document ID
            doc_id_match = re.search(r'/documents/\d{4}/\d{2}/\d{2}/([^/]+)/', link)
            doc_id = doc_id_match.group(1) if doc_id_match else "Unknown"
            
            print(f"\n📋 Document {i+1}:")
            print(f"   ID: {doc_id}")
            print(f"   Title: {title[:80]}...")
            print(f"   Agency: {creator}")
            print(f"   Date: {pub_date}")
            print(f"   URL: {link}")
        
        print(f"\n🎉 RSS feed test successful! Found {len(items)} recent documents")
        return True
        
    except Exception as e:
        print(f"❌ RSS feed test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_rss_feed()
    if success:
        print("\n✅ RSS approach is working - we can use this for reliable document ingestion!")
    else:
        print("\n❌ RSS approach failed - need to investigate further")
