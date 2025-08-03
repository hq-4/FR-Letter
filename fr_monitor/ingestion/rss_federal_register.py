"""
RSS-based Federal Register client for reliable document ingestion.

This module provides a client that uses the Federal Register RSS feed
to get the most recent documents, avoiding weekend/holiday issues.
"""

import requests
import xml.etree.ElementTree as ET
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import structlog
from urllib.parse import urlparse, parse_qs
import re

from ..core.models import FederalRegisterDocument, DocumentType, Agency

logger = structlog.get_logger(__name__)

class RSSFederalRegisterClient:
    """RSS-based client for Federal Register documents.
    
    Uses the RSS feed to get recent documents reliably, regardless of
    weekends or holidays.
    """
    
    RSS_URL = "https://www.federalregister.gov/api/v1/documents.rss"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "FederalRegisterMonitor/1.0",
            "Accept": "application/rss+xml, application/xml, text/xml"
        })
    
    def get_recent_documents(
        self,
        max_documents: int = 100,
        days_back: int = 7
    ) -> List[FederalRegisterDocument]:
        """
        Retrieve recent documents from Federal Register RSS feed.
        
        Args:
            max_documents: Maximum number of documents to return
            days_back: How many days back to consider (for filtering)
        
        Returns:
            List of FederalRegisterDocument objects
        """
        try:
            logger.info("Fetching Federal Register RSS feed")
            response = self.session.get(self.RSS_URL, timeout=30)
            response.raise_for_status()
            
            # Parse RSS XML
            root = ET.fromstring(response.content)
            
            documents = []
            items = root.findall('.//item')
            
            logger.info(f"Found {len(items)} items in RSS feed")
            
            for item in items[:max_documents]:
                try:
                    document = self._parse_rss_item(item)
                    if document:
                        documents.append(document)
                except Exception as e:
                    title = item.find('title')
                    title_text = title.text if title is not None else "Unknown"
                    logger.error("Failed to parse RSS item", 
                               title=title_text[:100],
                               error=str(e))
            
            logger.info(f"Successfully parsed {len(documents)} documents from RSS")
            return documents
            
        except requests.RequestException as e:
            logger.error("Failed to fetch RSS feed", error=str(e))
            return []
        except ET.ParseError as e:
            logger.error("Failed to parse RSS XML", error=str(e))
            return []
    
    def _parse_rss_item(self, item: ET.Element) -> Optional[FederalRegisterDocument]:
        """Parse an RSS item into a FederalRegisterDocument.
        
        Args:
            item: RSS item XML element
            
        Returns:
            Parsed FederalRegisterDocument or None if parsing fails
        """
        # Extract basic fields
        title_elem = item.find('title')
        link_elem = item.find('link')
        description_elem = item.find('description')
        pub_date_elem = item.find('pubDate')
        creator_elem = item.find('.//{http://purl.org/dc/elements/1.1/}creator')
        
        if not title_elem or not link_elem:
            logger.warning("RSS item missing required title or link")
            return None
        
        title = title_elem.text or ""
        link = link_elem.text or ""
        description = description_elem.text if description_elem is not None else ""
        
        # Extract document ID from URL
        # URL format: https://www.federalregister.gov/documents/2025/08/04/2025-14789/title
        document_id = self._extract_document_id(link)
        if not document_id:
            logger.warning("Could not extract document ID from URL", url=link)
            return None
        
        # Parse publication date
        pub_date = None
        if pub_date_elem is not None:
            try:
                # RSS date format: "Mon, 04 Aug 2025 04:00:00 GMT"
                pub_date = datetime.strptime(pub_date_elem.text, "%a, %d %b %Y %H:%M:%S %Z").date()
            except (ValueError, TypeError):
                logger.warning("Could not parse publication date", date_text=pub_date_elem.text)
        
        # Parse agencies from creator field
        agencies = []
        if creator_elem is not None and creator_elem.text:
            # Creator format: "Agriculture Department, Agricultural Marketing Service"
            agency_names = [name.strip() for name in creator_elem.text.split(',')]
            for name in agency_names:
                # Create abbreviation from name
                abbreviation = self._create_agency_abbreviation(name)
                agencies.append(Agency(name=name, abbreviation=abbreviation))
        
        # Determine document type from title and description
        document_type = self._determine_document_type(title, description)
        
        # Create URLs
        pdf_url = self._construct_pdf_url(link, pub_date)
        
        return FederalRegisterDocument(
            document_id=document_id,
            title=title,
            abstract=description,  # Use description as abstract
            html_url=link,
            pdf_url=pdf_url,
            full_text_url=None,  # Not available in RSS
            
            document_type=document_type,
            agencies=agencies,
            publication_date=pub_date,
            effective_date=None,  # Not available in RSS
            
            page_length=None,  # Not available in RSS
            is_final_rule=document_type == DocumentType.FINAL_RULE,
            is_major_rule=False,  # Would need full document to determine
            
            raw_data={
                'rss_title': title,
                'rss_link': link,
                'rss_description': description,
                'rss_creator': creator_elem.text if creator_elem is not None else None
            }
        )
    
    def _extract_document_id(self, url: str) -> Optional[str]:
        """Extract document ID from Federal Register URL.
        
        Args:
            url: Federal Register document URL
            
        Returns:
            Document ID or None if not found
        """
        # URL format: https://www.federalregister.gov/documents/2025/08/04/2025-14789/title
        match = re.search(r'/documents/\d{4}/\d{2}/\d{2}/([^/]+)/', url)
        if match:
            return match.group(1)
        
        # Fallback: try to extract from end of path
        parts = url.rstrip('/').split('/')
        if len(parts) >= 2:
            potential_id = parts[-2]
            if re.match(r'\d{4}-\d+', potential_id):
                return potential_id
        
        return None
    
    def _create_agency_abbreviation(self, agency_name: str) -> str:
        """Create an abbreviation from agency name.
        
        Args:
            agency_name: Full agency name
            
        Returns:
            Agency abbreviation
        """
        # Common agency mappings
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
            'Federal Emergency Management Agency': 'FEMA',
            'Social Security Administration': 'SSA',
            'Centers for Medicare & Medicaid Services': 'CMS',
            'Centers for Disease Control and Prevention': 'CDC'
        }
        
        if agency_name in mappings:
            return mappings[agency_name]
        
        # Generate abbreviation from first letters of words
        words = agency_name.replace('Department of', '').replace('Office of', '').strip().split()
        abbreviation = ''.join(word[0].upper() for word in words if word and len(word) > 2)
        return abbreviation[:10]  # Limit length
    
    def _determine_document_type(self, title: str, description: str) -> DocumentType:
        """Determine document type from title and description.
        
        Args:
            title: Document title
            description: Document description
            
        Returns:
            DocumentType enum value
        """
        text = f"{title} {description}".lower()
        
        if any(keyword in text for keyword in ['final rule', 'final regulation']):
            return DocumentType.FINAL_RULE
        elif any(keyword in text for keyword in ['proposed rule', 'proposed regulation', 'notice of proposed']):
            return DocumentType.PROPOSED_RULE
        elif any(keyword in text for keyword in ['presidential', 'executive order', 'proclamation']):
            return DocumentType.PRESIDENTIAL_DOCUMENT
        elif 'correction' in text:
            return DocumentType.CORRECTION
        elif 'notice' in text:
            return DocumentType.NOTICE
        else:
            return DocumentType.OTHER
    
    def _construct_pdf_url(self, html_url: str, pub_date: Optional[date]) -> Optional[str]:
        """Construct PDF URL from HTML URL and publication date.
        
        Args:
            html_url: HTML document URL
            pub_date: Publication date
            
        Returns:
            PDF URL or None if cannot be constructed
        """
        if not pub_date:
            return None
        
        document_id = self._extract_document_id(html_url)
        if not document_id:
            return None
        
        # PDF URL format: https://www.govinfo.gov/content/pkg/FR-2025-08-04/pdf/2025-14789.pdf
        return f"https://www.govinfo.gov/content/pkg/FR-{pub_date:%Y-%m-%d}/pdf/{document_id}.pdf"
