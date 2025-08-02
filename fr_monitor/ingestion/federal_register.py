"""
Federal Register API client for document ingestion.
"""

import requests
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
import structlog
from urllib.parse import urljoin

from ..core.models import FederalRegisterDocument, DocumentType, Agency
from ..core.config import settings

logger = structlog.get_logger(__name__)


class FederalRegisterClient:
    """Client for interacting with the Federal Register API."""
    
    BASE_URL = "https://www.federalregister.gov/api/v1/"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.federal_register_api_key
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        
        self.session.headers.update({
            "User-Agent": "FederalRegisterMonitor/1.0",
            "Accept": "application/json"
        })
    
    def get_daily_documents(
        self,
        target_date: Optional[date] = None,
        document_types: Optional[List[str]] = None,
        agencies: Optional[List[str]] = None,
        per_page: int = 1000
    ) -> List[FederalRegisterDocument]:
        """
        Retrieve documents from Federal Register for a specific date.
        
        Args:
            target_date: Date to retrieve documents for (defaults to yesterday)
            document_types: Filter by document types
            agencies: Filter by agency abbreviations
            per_page: Number of documents per page (max 1000)
        
        Returns:
            List of FederalRegisterDocument objects
        """
        if target_date is None:
            target_date = date.today() - timedelta(days=1)
        
        # Default document types of interest
        if document_types is None:
            document_types = [
                "RULE",  # Final rules
                "PRORULE",  # Proposed rules
                "PRESDOCU",  # Presidential documents
                "NOTICE"  # Notices
            ]
        
        params = {
            "conditions[publication_date][is]": target_date.strftime("%Y-%m-%d"),
            "conditions[type][]": document_types,
            "per_page": per_page,
            "page": 1,
            "fields[]": [
                "document_number",
                "title",
                "abstract",
                "type",
                "agencies",
                "publication_date",
                "effective_on",
                "page_length",
                "html_url",
                "pdf_url",
                "full_text_xml_url",
                "regulation_id_number_info"
            ]
        }
        
        if agencies:
            params["conditions[agencies][]"] = agencies
        
        logger.info("Fetching Federal Register documents", 
                   date=target_date, 
                   document_types=document_types,
                   agencies=agencies)
        
        documents = []
        page = 1
        
        while True:
            params["page"] = page
            
            try:
                response = self.session.get(
                    urljoin(self.BASE_URL, "documents.json"),
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                results = data.get("results", [])
                
                if not results:
                    break
                
                for doc_data in results:
                    try:
                        document = self._parse_document(doc_data)
                        documents.append(document)
                    except Exception as e:
                        logger.warning("Failed to parse document", 
                                     document_id=doc_data.get("document_number"),
                                     error=str(e))
                
                # Check if there are more pages
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
        """Parse raw API response into FederalRegisterDocument."""
        
        # Map document types
        type_mapping = {
            "RULE": DocumentType.FINAL_RULE,
            "PRORULE": DocumentType.PROPOSED_RULE,
            "PRESDOCU": DocumentType.PRESIDENTIAL_DOCUMENT,
            "NOTICE": DocumentType.NOTICE
        }
        
        document_type = type_mapping.get(
            doc_data.get("type"), 
            DocumentType.NOTICE
        )
        
        # Parse agencies
        agencies = []
        for agency_data in doc_data.get("agencies", []):
            agency = Agency(
                name=agency_data.get("name", ""),
                abbreviation=agency_data.get("slug", "").upper()
            )
            agencies.append(agency)
        
        # Parse dates
        pub_date = datetime.strptime(
            doc_data["publication_date"], 
            "%Y-%m-%d"
        )
        
        effective_date = None
        if doc_data.get("effective_on"):
            effective_date = datetime.strptime(
                doc_data["effective_on"], 
                "%Y-%m-%d"
            )
        
        # Determine if it's a final rule or major rule
        is_final_rule = document_type == DocumentType.FINAL_RULE
        is_major_rule = self._is_major_rule(doc_data)
        
        return FederalRegisterDocument(
            document_id=doc_data["document_number"],
            title=doc_data.get("title", ""),
            abstract=doc_data.get("abstract"),
            html_url=doc_data.get("html_url"),
            pdf_url=doc_data.get("pdf_url"),
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
        """Determine if a document represents a major rule."""
        # Check regulation ID number info for major rule indicators
        reg_info = doc_data.get("regulation_id_number_info", {})
        
        # Major rules often have specific RIN patterns or are marked as significant
        if reg_info.get("significant") == "1":
            return True
        
        # Check for economic impact indicators in title/abstract
        text_to_check = (
            doc_data.get("title", "") + " " + 
            doc_data.get("abstract", "")
        ).lower()
        
        major_rule_indicators = [
            "major rule",
            "significant regulatory action",
            "economically significant",
            "$100 million",
            "$1 billion"
        ]
        
        return any(indicator in text_to_check for indicator in major_rule_indicators)
    
    def get_document_full_text(self, document_id: str) -> Optional[str]:
        """
        Retrieve full text content for a specific document.
        
        Args:
            document_id: Federal Register document number
            
        Returns:
            Full text content or None if not available
        """
        try:
            response = self.session.get(
                urljoin(self.BASE_URL, f"documents/{document_id}.json"),
                params={"fields[]": ["full_text_xml_url", "body"]},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Try to get full text from body field first
            if data.get("body"):
                return data["body"]
            
            # If no body, try to fetch from XML URL
            xml_url = data.get("full_text_xml_url")
            if xml_url:
                xml_response = self.session.get(xml_url, timeout=30)
                xml_response.raise_for_status()
                return xml_response.text
            
            return None
            
        except requests.RequestException as e:
            logger.error("Failed to fetch document full text", 
                        document_id=document_id, 
                        error=str(e))
            return None
