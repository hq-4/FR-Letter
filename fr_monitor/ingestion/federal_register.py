"""
Federal Register API client for document ingestion.

This module provides a client for interacting with the Federal Register API.
The API is public and does not require an API key.
"""

import requests
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
import structlog
from urllib.parse import urljoin

from ..core.models import FederalRegisterDocument, DocumentType, Agency

logger = structlog.get_logger(__name__)

class FederalRegisterClient:
    """Client for interacting with the Federal Register RSS feed.
    
    Uses RSS feed for reliable document retrieval, avoiding weekend/holiday issues.
    Falls back to API for detailed document information when needed.
    """
    
    RSS_URL = "https://www.federalregister.gov/api/v1/documents.rss"
    BASE_URL = "https://www.federalregister.gov/api/v1/"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "FederalRegisterMonitor/1.0",
            "Accept": "application/json, application/rss+xml, application/xml"
        })
    
    def get_recent_documents_from_rss(
        self,
        max_documents: int = 100
    ) -> List[FederalRegisterDocument]:
        """Get recent documents from RSS feed (recommended approach).
        
        Args:
            max_documents: Maximum number of documents to return
            
        Returns:
            List of FederalRegisterDocument objects
        """
        try:
            import xml.etree.ElementTree as ET
            import re
            
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
            
        except Exception as e:
            logger.error("Failed to fetch RSS feed", error=str(e))
            return []
    
    def get_daily_documents(
        self,
        target_date: Optional[date] = None,
        document_types: Optional[List[str]] = None,
        agencies: Optional[List[str]] = None,
        per_page: int = 1000
    ) -> List[FederalRegisterDocument]:
        """Get documents (now uses RSS feed by default for reliability).
        
        This method now uses the RSS feed approach by default since it's more
        reliable than date-based queries, especially for weekends/holidays.
        """
        # Use RSS feed approach by default
        if target_date is None:
            return self.get_recent_documents_from_rss(max_documents=per_page)
        
        # For specific dates, still try the original API but with better error handling
        return self._get_daily_documents_api(target_date, document_types, agencies, per_page)
    
    def _get_daily_documents_api(
        self,
        target_date: date,
        document_types: Optional[List[str]] = None,
        agencies: Optional[List[str]] = None,
        per_page: int = 1000
    ) -> List[FederalRegisterDocument]:
        """
        Retrieve documents from Federal Register for a specific date.
        
        Args:
            target_date: Date to retrieve documents for (defaults to yesterday)
            document_types: Filter by document types (e.g., ["RULE", "NOTICE", "PRORULE"])
            agencies: Filter by agency slugs (e.g., ["environmental-protection-agency"])
            per_page: Number of documents per page (max 1000)
        
        Returns:
            List of FederalRegisterDocument objects
        """
        if target_date is None:
            target_date = date.today() - timedelta(days=1)
        
        if document_types is None:
            document_types = ["RULE", "PRORULE", "PRESDOCU", "NOTICE"]
        
        documents = []
        page = 1
        
        while True:
            try:
                # Build query parameters
                params = {
                    "format": "json",
                    "conditions[publication_date][is]": target_date.isoformat(),
                    "per_page": min(per_page, 1000),  # API max is 1000
                    "page": page,
                    "order": "newest"
                }
                
                # Add document type filters if specified
                if document_types:
                    params["conditions[type][]"] = document_types
                    
                # Add agency filters if specified
                if agencies:
                    params["conditions[agencies][]"] = agencies
                
                # Make the API request
                response = self.session.get(
                    self.BASE_URL + "documents.json",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                results = data.get("results", [])
                
                # Parse each document
                for doc_data in results:
                    try:
                        document = self._parse_document(doc_data)
                        documents.append(document)
                    except Exception as e:
                        logger.error("Failed to parse document", 
                                   document_id=doc_data.get("document_number"),
                                   error=str(e))
                
                # Check if we've reached the end of results
                if len(results) < per_page:
                    break
                
                page += 1
                
            except requests.RequestException as e:
                logger.error("Failed to fetch documents", 
                           page=page, 
                           error=str(e))
                break
        
        logger.info("Retrieved documents", count=len(documents), date=target_date)
        return documents
    
    def _parse_document(self, doc_data: Dict[str, Any]) -> FederalRegisterDocument:
        """Parse raw API response into FederalRegisterDocument.
        
        Args:
            doc_data: Raw document data from the Federal Register API
            
        Returns:
            Parsed FederalRegisterDocument object
        """
        # Map document types from API to our enum
        type_mapping = {
            "RULE": DocumentType.FINAL_RULE,
            "PRORULE": DocumentType.PROPOSED_RULE,
            "PRESDOCU": DocumentType.PRESIDENTIAL_DOCUMENT,
            "NOTICE": DocumentType.NOTICE,
            "PRESDOCU": DocumentType.PRESIDENTIAL_DOCUMENT,
            "PRESNOTICE": DocumentType.PRESIDENTIAL_DOCUMENT,
            "CORRECT": DocumentType.CORRECTION,
            "CORRECTION": DocumentType.CORRECTION,
            "UNKNOWN": DocumentType.OTHER
        }
        
        # Get document type with fallback to NOTICE
        doc_type = doc_data.get("type", "NOTICE")
        document_type = type_mapping.get(doc_type, DocumentType.NOTICE)
        
        # Parse agencies
        agencies = []
        for agency_data in doc_data.get("agencies", []):
            # Use the agency slug as the abbreviation
            slug = agency_data.get("slug", "")
            agency = Agency(
                name=agency_data.get("name", ""),
                abbreviation=slug.upper() if slug else ""
            )
            agencies.append(agency)
        
        # Parse publication date (required field)
        pub_date = datetime.strptime(
            doc_data["publication_date"].split("T")[0],  # Handle ISO8601 with time
            "%Y-%m-%d"
        ).date()
        
        # Parse effective date if available
        effective_date = None
        if doc_data.get("effective_on"):
            try:
                effective_date = datetime.strptime(
                    doc_data["effective_on"].split("T")[0],
                    "%Y-%m-%d"
                ).date()
            except (ValueError, TypeError, AttributeError):
                effective_date = None
        
        # Determine if this is a final rule and if it's major
        is_final_rule = document_type == DocumentType.FINAL_RULE
        is_major_rule = is_final_rule and self._is_major_rule(doc_data)
        
        # Get document URLs
        document_number = doc_data["document_number"]
        html_url = f"https://www.federalregister.gov/documents/{pub_date.year}/{pub_date.month:02d}/{pub_date.day:02d}/{document_number}"
        pdf_url = f"https://www.govinfo.gov/content/pkg/FR-{pub_date:%Y-%m-%d}/pdf/{document_number}.pdf"
        
        return FederalRegisterDocument(
            document_id=document_number,
            title=doc_data.get("title", ""),
            abstract=doc_data.get("abstract"),
            html_url=html_url,
            pdf_url=pdf_url,
            full_text_url=doc_data.get("full_text_xml_url"),
            
            document_type=document_type,
            agencies=agencies,
            publication_date=pub_date,
            effective_date=effective_date,
            
            page_length=doc_data.get("page_length"),
            is_final_rule=is_final_rule,
            is_major_rule=is_major_rule,
            
            raw_data=doc_data
        )
    
    def _is_major_rule(self, doc_data: Dict[str, Any]) -> bool:
        """Determine if a document represents a major rule.
        
        Args:
            doc_data: Raw document data from the API
            
        Returns:
            True if the document is a major rule, False otherwise
        """
        # Check for explicit major rule flag
        if doc_data.get("major"):
            return True
            
        # Check regulation ID number info for major rule indicators
        reg_info = doc_data.get("regulation_id_number_info", {})
        
        # Major rules are often marked as significant
        if reg_info.get("significant") == "1":
            return True
        
        # Check for economic impact indicators in title/abstract
        text_to_check = (
            (doc_data.get("title", "") + " " + 
             doc_data.get("abstract", "") + " " +
             doc_data.get("action", "") + " " +
             doc_data.get("summary", ""))
        ).lower()
        
        major_rule_indicators = [
            "major rule",
            "significant regulatory action",
            "economically significant",
            "$100 million",
            "$1 billion",
            "executive order 12866",
            "executive order 13563",
            "omb significant"
        ]
        
        return any(indicator in text_to_check for indicator in major_rule_indicators)
    
    def get_document_full_text(self, document_id: str) -> Optional[str]:
        """
        Retrieve full text content for a specific document.
        
        Args:
            document_id: Federal Register document number (e.g., "2023-12345")
            
        Returns:
            Full text content or None if not available
        """
        try:
            # First try to get the document with full text fields
            response = self.session.get(
                f"{self.BASE_URL}documents/{document_id}.json",
                params={"fields[]": ["body_html", "body", "full_text_xml_url"]},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Try to get full text from body_html or body fields first
            full_text = data.get("body_html") or data.get("body")
            if full_text:
                return full_text
            
            # Fall back to XML URL if available
            xml_url = data.get("full_text_xml_url")
            if xml_url:
                xml_response = self.session.get(xml_url, timeout=30)
                xml_response.raise_for_status()
                return xml_response.text
            
            logger.warning("No full text available for document", 
                         document_id=document_id)
            return None
            
        except requests.RequestException as e:
            logger.error("Failed to fetch document full text", 
                       document_id=document_id, 
                       error=str(e))
            return None
