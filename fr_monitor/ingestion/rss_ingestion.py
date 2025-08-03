"""
RSS-based Federal Register ingestion with SQLite storage.
"""
import re
import requests
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from xml.etree import ElementTree as ET
from urllib.parse import urlparse, urljoin

from ..storage.database import FederalRegisterDB

logger = logging.getLogger(__name__)


class RSSIngestionClient:
    """Client for ingesting Federal Register RSS feeds into SQLite."""
    
    def __init__(self, db_path: str = "federal_register.db"):
        self.db = FederalRegisterDB(db_path)
        self.rss_url = "https://www.federalregister.gov/api/v1/documents.rss"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Federal Register Monitor/1.0'
        })
    
    def fetch_and_store_rss(self) -> int:
        """Fetch RSS feed and store in database."""
        logger.info(f"Fetching RSS feed from {self.rss_url}")
        
        try:
            response = self.session.get(self.rss_url, timeout=30)
            response.raise_for_status()
            
            rss_content = response.text
            fetch_timestamp = datetime.now()
            
            # Parse RSS to count documents
            root = ET.fromstring(rss_content)
            items = root.findall('.//item')
            document_count = len(items)
            
            # Store RSS dump
            dump_id = self.db.store_rss_dump(rss_content, document_count, fetch_timestamp)
            
            # Parse and store individual documents
            stored_count = 0
            for i, item in enumerate(items):
                try:
                    doc_data = self._parse_rss_item(item)
                    if doc_data:
                        self.db.store_document(**doc_data)
                        stored_count += 1
                        if stored_count <= 3:  # Log first few successful parses
                            logger.info(f"Successfully parsed item {i+1}: {doc_data['document_number']} - {doc_data['title'][:50]}...")
                    else:
                        if i < 5:  # Log first few failures for debugging
                            logger.warning(f"Failed to parse RSS item {i+1} - returned None")
                except Exception as e:
                    if i < 5:  # Log first few exceptions for debugging
                        logger.error(f"Exception parsing RSS item {i+1}: {e}")
                    continue
            
            logger.info(f"Stored {stored_count}/{document_count} documents from RSS feed")
            return stored_count
            
        except Exception as e:
            logger.error(f"Failed to fetch RSS feed: {e}")
            raise
    
    def _parse_rss_item(self, item: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse individual RSS item into document data."""
        try:
            # Extract basic fields
            title_elem = item.find('title')
            link_elem = item.find('link')
            pub_date_elem = item.find('pubDate')
            
            # Only require title and link - pubDate can be missing
            if title_elem is None or link_elem is None:
                logger.debug(f"Missing required elements: title_elem={title_elem is not None}, link_elem={link_elem is not None}")
                return None
                
            # Use robust text extraction (itertext handles child elements)
            title = ''.join(title_elem.itertext()).strip()
            rss_link = ''.join(link_elem.itertext()).strip()
            
            logger.debug(f"Extracted: title='{title[:50]}...', link='{rss_link}'")
            
            # Skip if we don't have the essential data
            if not title or not rss_link:
                logger.debug(f"Empty essential data: title_empty={not bool(title)}, link_empty={not bool(rss_link)}")
                return None
            
            # Extract document number from link
            document_number = self._extract_document_number(rss_link)
            if not document_number:
                logger.debug(f"Could not extract document number from {rss_link}")
                return None
            
            # Parse publication date
            publication_date = None
            if pub_date_elem and pub_date_elem.text:
                try:
                    # RSS pubDate format: "Mon, 04 Aug 2025 04:00:00 GMT"
                    pub_date_text = pub_date_elem.text.strip()
                    # Handle both GMT and +0000 formats
                    if pub_date_text.endswith('GMT'):
                        pub_date = datetime.strptime(pub_date_text, "%a, %d %b %Y %H:%M:%S %Z")
                    else:
                        pub_date = datetime.strptime(pub_date_text, "%a, %d %b %Y %H:%M:%S %z")
                    publication_date = pub_date.date().isoformat()
                except ValueError as e:
                    logger.warning(f"Could not parse publication date '{pub_date_elem.text}': {e}")
            
            # Extract agency from dc:creator or fallback methods
            agency = self._extract_agency(item)
            
            # Convert RSS link to XML URL
            xml_url = self._convert_to_xml_url(rss_link)
            
            return {
                'document_number': document_number,
                'title': title,
                'agency': agency,
                'publication_date': publication_date,
                'rss_link': rss_link,
                'xml_url': xml_url
            }
            
        except Exception as e:
            logger.error(f"Error parsing RSS item: {e}")
            return None
    
    def _extract_document_number(self, rss_link: str) -> Optional[str]:
        """Extract document number from RSS link."""
        # Example: https://www.federalregister.gov/documents/2025/08/04/2025-14681/medicare-program-hospital...
        # Document number: 2025-14681
        
        pattern = r'/documents/\d{4}/\d{2}/\d{2}/([^/]+)/'
        match = re.search(pattern, rss_link)
        
        if match:
            return match.group(1)
        
        # Fallback: try to extract from end of URL
        parts = rss_link.rstrip('/').split('/')
        if len(parts) >= 2:
            # Look for pattern like "2025-14681"
            for part in reversed(parts):
                if re.match(r'\d{4}-\d+', part):
                    return part
        
        return None
    
    def _extract_agency(self, item: ET.Element) -> str:
        """Extract agency name from RSS item."""
        # First try dc:creator (most reliable)
        dc_creator = item.find('.//{http://purl.org/dc/elements/1.1/}creator')
        if dc_creator is not None:
            agency = ''.join(dc_creator.itertext()).strip()
            if agency:
                # Clean up common agency name formats
                agency = self._normalize_agency_name(agency)
                return agency
        
        # Try description
        desc_elem = item.find('description')
        if desc_elem and desc_elem.text:
            # Look for agency patterns in description
            desc = desc_elem.text
            
            # Common agency abbreviations
            agency_patterns = [
                r'EPA\b', r'FDA\b', r'DOT\b', r'HHS\b', r'DOE\b', r'USDA\b',
                r'Treasury\b', r'Commerce\b', r'Labor\b', r'Interior\b',
                r'Justice\b', r'Defense\b', r'State\b', r'Homeland Security\b',
                r'Veterans Affairs\b', r'Education\b', r'Housing\b'
            ]
            
            for pattern in agency_patterns:
                if re.search(pattern, desc, re.IGNORECASE):
                    return re.search(pattern, desc, re.IGNORECASE).group()
        
        # Fallback: try to extract from title
        title_elem = item.find('title')
        if title_elem and title_elem.text:
            title = title_elem.text
            # Look for agency in parentheses or brackets
            agency_match = re.search(r'[\(\[]([A-Z]{2,}|[A-Z][a-z]+ [A-Z][a-z]+)[\)\]]', title)
            if agency_match:
                return agency_match.group(1)
        
        return "Unknown"
    
    def _normalize_agency_name(self, agency: str) -> str:
        """Normalize agency name to a consistent format."""
        # Common agency name mappings
        agency_mappings = {
            "Executive Office of the President": "Executive Office",
            "Department of Homeland Security": "DHS",
            "Department of Transportation": "DOT",
            "Department of Health and Human Services": "HHS",
            "Department of Energy": "DOE",
            "Department of Agriculture": "USDA",
            "Department of the Treasury": "Treasury",
            "Department of Commerce": "Commerce",
            "Department of Labor": "Labor",
            "Department of the Interior": "Interior",
            "Department of Justice": "Justice",
            "Department of Defense": "Defense",
            "Department of State": "State",
            "Department of Veterans Affairs": "VA",
            "Department of Education": "Education",
            "Department of Housing and Urban Development": "HUD",
            "Environmental Protection Agency": "EPA",
            "Food and Drug Administration": "FDA"
        }
        
        # Return mapped name if found, otherwise return original
        return agency_mappings.get(agency, agency)
    
    def _convert_to_xml_url(self, rss_link: str) -> str:
        """Convert RSS link to XML full-text URL."""
        # Example conversion:
        # From: https://www.federalregister.gov/documents/2025/08/04/2025-14681/medicare-program-hospital...
        # To: https://www.federalregister.gov/documents/full_text/xml/2025/08/04/2025-14681.xml
        
        # Extract date and document number
        pattern = r'/documents/(\d{4})/(\d{2})/(\d{2})/([^/]+)/'
        match = re.search(pattern, rss_link)
        
        if match:
            year, month, day, doc_number = match.groups()
            xml_url = f"https://www.federalregister.gov/documents/full_text/xml/{year}/{month}/{day}/{doc_number}.xml"
            return xml_url
        
        # Fallback: construct from document number if we can extract it
        doc_number = self._extract_document_number(rss_link)
        if doc_number:
            # Try to extract date from document number (format: YYYY-NNNNN)
            if '-' in doc_number:
                year = doc_number.split('-')[0]
                # This is a fallback - we'd need the actual date
                # For now, return a generic pattern
                return f"https://www.federalregister.gov/documents/full_text/xml/{doc_number}.xml"
        
        logger.warning(f"Could not convert RSS link to XML URL: {rss_link}")
        return rss_link  # Return original as fallback
    
    def download_xml_content(self, limit: Optional[int] = None) -> int:
        """Download XML content for documents that don't have it yet."""
        # Get documents without XML content
        with self.db.get_connection() as conn:
            query = """
                SELECT id, document_number, xml_url 
                FROM documents 
                WHERE xml_content IS NULL
                ORDER BY publication_date DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor = conn.execute(query)
            documents = [dict(row) for row in cursor.fetchall()]
        
        if not documents:
            logger.info("No documents need XML content download")
            return 0
        
        logger.info(f"Downloading XML content for {len(documents)} documents")
        
        downloaded_count = 0
        for doc in documents:
            try:
                xml_content = self._download_xml(doc['xml_url'])
                if xml_content:
                    self.db.update_document_xml(doc['id'], xml_content)
                    downloaded_count += 1
                    logger.info(f"Downloaded XML for document {doc['document_number']}")
                else:
                    logger.warning(f"Failed to download XML for {doc['document_number']}")
                    
            except Exception as e:
                logger.error(f"Error downloading XML for {doc['document_number']}: {e}")
                continue
        
        logger.info(f"Successfully downloaded XML for {downloaded_count}/{len(documents)} documents")
        return downloaded_count
    
    def _download_xml(self, xml_url: str) -> Optional[str]:
        """Download XML content from URL."""
        try:
            response = self.session.get(xml_url, timeout=60)
            response.raise_for_status()
            
            # Verify it's XML content
            content_type = response.headers.get('content-type', '').lower()
            if 'xml' not in content_type and not xml_url.endswith('.xml'):
                logger.warning(f"URL {xml_url} doesn't appear to be XML content")
            
            return response.text
            
        except requests.RequestException as e:
            logger.error(f"Failed to download XML from {xml_url}: {e}")
            return None
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        stats = self.db.get_stats()
        
        # Add ingestion-specific stats
        with self.db.get_connection() as conn:
            # Documents needing XML download
            cursor = conn.execute("SELECT COUNT(*) as count FROM documents WHERE xml_content IS NULL")
            stats['documents_needing_xml'] = cursor.fetchone()['count']
            
            # Average XML size
            cursor = conn.execute("SELECT AVG(xml_size) as avg_size FROM documents WHERE xml_size IS NOT NULL")
            result = cursor.fetchone()
            stats['avg_xml_size_bytes'] = int(result['avg_size']) if result['avg_size'] else 0
        
        return stats
