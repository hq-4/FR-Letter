#!/usr/bin/env python3
"""
Simple debug script to test RSS parsing step by step.
"""
import requests
import xml.etree.ElementTree as ET
import re
from datetime import datetime

def test_document_number_extraction(rss_link):
    """Test document number extraction logic."""
    print(f"Testing link: {rss_link}")
    
    # Pattern from the code
    pattern = r'/documents/\d{4}/\d{2}/\d{2}/([^/]+)/'
    match = re.search(pattern, rss_link)
    
    if match:
        doc_number = match.group(1)
        print(f"  ✅ Extracted document number: {doc_number}")
        return doc_number
    else:
        print(f"  ❌ Could not extract document number")
        
        # Try fallback logic
        parts = rss_link.rstrip('/').split('/')
        print(f"  URL parts: {parts}")
        
        for part in reversed(parts):
            if re.match(r'\d{4}-\d+', part):
                print(f"  ✅ Fallback found: {part}")
                return part
        
        print(f"  ❌ Fallback also failed")
        return None

def test_xml_url_conversion(rss_link):
    """Test XML URL conversion."""
    print(f"Converting to XML URL: {rss_link}")
    
    # Pattern from the code
    pattern = r'/documents/(\d{4})/(\d{2})/(\d{2})/([^/]+)/'
    match = re.search(pattern, rss_link)
    
    if match:
        year, month, day, doc_number = match.groups()
        xml_url = f"https://www.federalregister.gov/documents/full_text/xml/{year}/{month}/{day}/{doc_number}.xml"
        print(f"  ✅ XML URL: {xml_url}")
        return xml_url
    else:
        print(f"  ❌ Could not convert to XML URL")
        return rss_link

def debug_rss_items():
    """Debug RSS items step by step."""
    print("=== RSS Parsing Debug ===\n")
    
    rss_url = "https://www.federalregister.gov/api/v1/documents.rss"
    
    try:
        response = requests.get(rss_url, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.text)
        items = root.findall('.//item')
        
        print(f"Found {len(items)} items in RSS feed\n")
        
        # Test first 3 items
        for i, item in enumerate(items[:3]):
            print(f"=== Item {i+1} ===")
            
            # Extract fields
            title_elem = item.find('title')
            link_elem = item.find('link')
            pub_date_elem = item.find('pubDate')
            dc_creator = item.find('.//{http://purl.org/dc/elements/1.1/}creator')
            
            # Use more robust text extraction
            title = ''.join(title_elem.itertext()).strip() if title_elem is not None else ""
            rss_link = ''.join(link_elem.itertext()).strip() if link_elem is not None else ""
            pub_date = ''.join(pub_date_elem.itertext()).strip() if pub_date_elem is not None else ""
            agency = ''.join(dc_creator.itertext()).strip() if dc_creator is not None else ""
            
            # Debug: show raw element info
            print(f"Raw title_elem: {title_elem}, text: '{title_elem.text if title_elem else None}'")
            print(f"Raw link_elem: {link_elem}, text: '{link_elem.text if link_elem else None}'")
            print(f"Raw pub_date_elem: {pub_date_elem}, text: '{pub_date_elem.text if pub_date_elem else None}'")
            print(f"Raw dc_creator: {dc_creator}, text: '{dc_creator.text if dc_creator else None}'")
            
            print(f"Title: {title[:80]}...")
            print(f"Link: {rss_link}")
            print(f"PubDate: {pub_date}")
            print(f"Agency (dc:creator): {agency}")
            
            # Test document number extraction
            doc_number = test_document_number_extraction(rss_link)
            
            # Test XML URL conversion
            if doc_number:
                xml_url = test_xml_url_conversion(rss_link)
            
            # Test date parsing
            if pub_date:
                try:
                    if pub_date.endswith('GMT'):
                        parsed_date = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
                    else:
                        parsed_date = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z")
                    print(f"  ✅ Parsed date: {parsed_date.date().isoformat()}")
                except ValueError as e:
                    print(f"  ❌ Date parsing failed: {e}")
            
            print(f"Would store: {bool(title and rss_link and doc_number)}")
            print()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_rss_items()
