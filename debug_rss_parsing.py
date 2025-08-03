#!/usr/bin/env python3
"""
Debug script to identify RSS parsing issues.
"""

import requests
import xml.etree.ElementTree as ET
import sys
import os
from datetime import datetime
import re

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fr_monitor.core.models import DocumentType, Agency

def debug_rss_parsing():
    """Debug RSS parsing to see where it's failing."""
    print("🔍 Debugging RSS Parsing")
    print("=" * 50)
    
    RSS_URL = "https://www.federalregister.gov/api/v1/documents.rss"
    
    try:
        print("📡 Fetching RSS feed...")
        response = requests.get(RSS_URL, timeout=30)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        items = root.findall('.//item')
        
        print(f"📄 Found {len(items)} items in RSS feed")
        
        # Debug first item in detail
        if items:
            item = items[0]
            print(f"\n🔍 Debugging first RSS item:")
            
            # Check all child elements
            for child in item:
                print(f"   {child.tag}: {child.text[:100] if child.text else 'None'}...")
            
            # Try parsing step by step
            print(f"\n📋 Step-by-step parsing:")
            
            # 1. Basic fields
            title_elem = item.find('title')
            link_elem = item.find('link')
            description_elem = item.find('description')
            pub_date_elem = item.find('pubDate')
            creator_elem = item.find('.//{http://purl.org/dc/elements/1.1/}creator')
            
            print(f"   Title element: {title_elem is not None}")
            print(f"   Link element: {link_elem is not None}")
            print(f"   Description element: {description_elem is not None}")
            print(f"   PubDate element: {pub_date_elem is not None}")
            print(f"   Creator element: {creator_elem is not None}")
            
            if not title_elem or not link_elem:
                print("❌ Missing required title or link element!")
                return False
            
            title = title_elem.text or ""
            link = link_elem.text or ""
            description = description_elem.text if description_elem is not None else ""
            
            print(f"   Title: {title[:100]}...")
            print(f"   Link: {link}")
            print(f"   Description length: {len(description)}")
            
            # 2. Extract document ID
            print(f"\n🆔 Extracting document ID:")
            match = re.search(r'/documents/\d{4}/\d{2}/\d{2}/([^/]+)/', link)
            if match:
                document_id = match.group(1)
                print(f"   Document ID: {document_id}")
            else:
                print(f"   ❌ Could not extract document ID from URL: {link}")
                
                # Try fallback method
                parts = link.rstrip('/').split('/')
                if len(parts) >= 2:
                    potential_id = parts[-2]
                    if re.match(r'\d{4}-\d+', potential_id):
                        document_id = potential_id
                        print(f"   Fallback document ID: {document_id}")
                    else:
                        print(f"   ❌ Fallback failed, potential_id: {potential_id}")
                        return False
                else:
                    print(f"   ❌ URL structure unexpected: {parts}")
                    return False
            
            # 3. Parse publication date
            print(f"\n📅 Parsing publication date:")
            if pub_date_elem is not None:
                try:
                    pub_date = datetime.strptime(pub_date_elem.text, "%a, %d %b %Y %H:%M:%S %Z").date()
                    print(f"   Publication date: {pub_date}")
                except (ValueError, TypeError) as e:
                    print(f"   ❌ Date parsing failed: {e}")
                    print(f"   Raw date text: {pub_date_elem.text}")
            else:
                print(f"   ⚠️  No publication date element")
            
            # 4. Parse agencies
            print(f"\n🏢 Parsing agencies:")
            if creator_elem is not None and creator_elem.text:
                agency_names = [name.strip() for name in creator_elem.text.split(',')]
                print(f"   Agency names: {agency_names}")
                
                agencies = []
                for name in agency_names:
                    abbreviation = create_agency_abbreviation(name)
                    agencies.append(Agency(name=name, abbreviation=abbreviation))
                    print(f"     {name} -> {abbreviation}")
            else:
                print(f"   ⚠️  No creator/agency information")
            
            # 5. Determine document type
            print(f"\n📝 Determining document type:")
            document_type = determine_document_type_from_text(title, description)
            print(f"   Document type: {document_type}")
            
            print(f"\n✅ Parsing appears to work for first item!")
            return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_agency_abbreviation(agency_name: str) -> str:
    """Create an abbreviation from agency name."""
    mappings = {
        'Agriculture Department': 'USDA',
        'Agricultural Marketing Service': 'AMS',
        'Environmental Protection Agency': 'EPA',
        'Department of Health and Human Services': 'HHS',
        'Food and Drug Administration': 'FDA',
        'Securities and Exchange Commission': 'SEC',
        'Federal Trade Commission': 'FTC',
        'Department of Transportation': 'DOT',
        'Federal Aviation Administration': 'FAA',
        'Department of Energy': 'DOE',
        'Department of Defense': 'DOD',
        'Department of Justice': 'DOJ',
        'Internal Revenue Service': 'IRS',
        'Department of Treasury': 'TREAS',
        'Department of Commerce': 'DOC',
        'Department of Labor': 'DOL',
        'Department of Education': 'ED',
        'Department of Veterans Affairs': 'VA',
        'Department of Homeland Security': 'DHS',
        'Homeland Security Department': 'DHS',
        'Federal Emergency Management Agency': 'FEMA',
        'Social Security Administration': 'SSA',
        'Centers for Medicare & Medicaid Services': 'CMS',
        'Centers for Disease Control and Prevention': 'CDC',
        'Coast Guard': 'USCG',
        'Executive Office of the President': 'EOP'
    }
    
    if agency_name in mappings:
        return mappings[agency_name]
    
    # Generate abbreviation from first letters of significant words
    words = agency_name.replace('Department of', '').replace('Office of', '').strip().split()
    abbreviation = ''.join(word[0].upper() for word in words if word and len(word) > 2)
    return abbreviation[:10]  # Limit length

def determine_document_type_from_text(title: str, description: str) -> DocumentType:
    """Determine document type from title and description."""
    text = f"{title} {description}".lower()
    
    if any(keyword in text for keyword in ['final rule', 'final regulation']):
        return DocumentType.FINAL_RULE
    elif any(keyword in text for keyword in ['proposed rule', 'proposed regulation', 'notice of proposed']):
        return DocumentType.PROPOSED_RULE
    elif any(keyword in text for keyword in ['presidential', 'executive order', 'proclamation', 'national emergency']):
        return DocumentType.PRESIDENTIAL_DOCUMENT
    elif 'correction' in text:
        return DocumentType.CORRECTION
    elif 'notice' in text:
        return DocumentType.NOTICE
    else:
        return DocumentType.OTHER

if __name__ == "__main__":
    success = debug_rss_parsing()
    if success:
        print("\n🎉 RSS parsing debug completed successfully!")
    else:
        print("\n🚨 RSS parsing has issues that need to be fixed!")
