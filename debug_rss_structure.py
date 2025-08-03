#!/usr/bin/env python3
"""
Debug script to examine the actual RSS feed structure.
"""
import requests
from xml.etree import ElementTree as ET

def debug_rss_structure():
    """Debug the RSS feed structure to understand parsing issues."""
    rss_url = "https://www.federalregister.gov/api/v1/documents.rss"
    
    print("=== Fetching RSS Feed ===")
    response = requests.get(rss_url, timeout=30)
    response.raise_for_status()
    
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    print(f"Content Length: {len(response.text)} characters")
    print()
    
    # Parse XML
    root = ET.fromstring(response.text)
    print(f"Root element: {root.tag}")
    
    # Find channel
    channel = root.find('channel')
    if channel is not None:
        print(f"Channel found with {len(channel)} children")
        
        # Find items
        items = channel.findall('item')
        print(f"Found {len(items)} items")
        print()
        
        # Examine first few items
        for i, item in enumerate(items[:3]):
            print(f"=== Item {i+1} ===")
            print(f"Item has {len(item)} children:")
            
            for child in item:
                print(f"  {child.tag}: {child.text[:100] if child.text else 'None'}...")
            print()
    else:
        print("No channel found")
        
        # Look for items directly
        items = root.findall('.//item')
        print(f"Found {len(items)} items directly")
        
        if items:
            item = items[0]
            print(f"First item has {len(item)} children:")
            for child in item:
                print(f"  {child.tag}: {child.text[:100] if child.text else 'None'}...")

if __name__ == "__main__":
    debug_rss_structure()
