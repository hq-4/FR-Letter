"""
Document chunking system for Federal Register documents.
"""

import re
from typing import List, Optional
import structlog
from bs4 import BeautifulSoup

from ..core.models import FederalRegisterDocument, DocumentChunk
from ..ingestion.federal_register import FederalRegisterClient

logger = structlog.get_logger(__name__)


class DocumentChunker:
    """Splits Federal Register documents into logical chunks for summarization."""
    
    def __init__(self, 
                 target_chunk_size: int = 500,
                 max_chunk_size: int = 750,
                 overlap_size: int = 50):
        """
        Initialize document chunker.
        
        Args:
            target_chunk_size: Target number of tokens per chunk
            max_chunk_size: Maximum tokens per chunk
            overlap_size: Number of tokens to overlap between chunks
        """
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.fr_client = FederalRegisterClient()
    
    def chunk_documents(self, documents: List[FederalRegisterDocument]) -> List[DocumentChunk]:
        """
        Chunk multiple documents into logical sections.
        
        Args:
            documents: List of Federal Register documents
            
        Returns:
            List of document chunks
        """
        all_chunks = []
        
        for doc in documents:
            try:
                doc_chunks = self.chunk_single_document(doc)
                all_chunks.extend(doc_chunks)
                
                logger.info("Chunked document", 
                           document_id=doc.document_id,
                           chunk_count=len(doc_chunks))
                           
            except Exception as e:
                logger.error("Failed to chunk document", 
                           document_id=doc.document_id,
                           error=str(e))
        
        logger.info("Completed document chunking", 
                   total_documents=len(documents),
                   total_chunks=len(all_chunks))
        
        return all_chunks
    
    def chunk_single_document(self, document: FederalRegisterDocument) -> List[DocumentChunk]:
        """
        Chunk a single document into logical sections.
        
        Args:
            document: Federal Register document
            
        Returns:
            List of chunks for the document
        """
        # Get full text content
        full_text = self._get_document_content(document)
        if not full_text:
            # Fall back to abstract if no full text
            full_text = document.abstract or document.title
        
        # Clean and preprocess text
        cleaned_text = self._clean_text(full_text)
        
        # Split into logical sections
        sections = self._split_into_sections(cleaned_text)
        
        # Create chunks from sections
        chunks = []
        for i, section in enumerate(sections):
            # Further split large sections if needed
            section_chunks = self._split_section_by_size(section, document.document_id, i)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _get_document_content(self, document: FederalRegisterDocument) -> Optional[str]:
        """Retrieve full text content for a document."""
        try:
            # Try to get full text from the API
            full_text = self.fr_client.get_document_full_text(document.document_id)
            
            if full_text:
                return full_text
            
            # Fall back to abstract + title
            content_parts = []
            if document.title:
                content_parts.append(f"Title: {document.title}")
            if document.abstract:
                content_parts.append(f"Abstract: {document.abstract}")
            
            return "\n\n".join(content_parts) if content_parts else None
            
        except Exception as e:
            logger.warning("Failed to retrieve document content", 
                         document_id=document.document_id,
                         error=str(e))
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess document text."""
        if not text:
            return ""
        
        # Remove HTML tags if present
        if '<' in text and '>' in text:
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove excessive punctuation
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'-{3,}', '---', text)
        
        return text.strip()
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections based on structure."""
        if not text:
            return []
        
        # Common section headers in Federal Register documents
        section_patterns = [
            r'\n\s*SUMMARY:\s*',
            r'\n\s*DATES:\s*',
            r'\n\s*ADDRESSES:\s*',
            r'\n\s*FOR FURTHER INFORMATION CONTACT:\s*',
            r'\n\s*SUPPLEMENTARY INFORMATION:\s*',
            r'\n\s*BACKGROUND:\s*',
            r'\n\s*DISCUSSION:\s*',
            r'\n\s*ANALYSIS:\s*',
            r'\n\s*CONCLUSION:\s*',
            r'\n\s*EFFECTIVE DATE:\s*',
            r'\n\s*I\.\s+',  # Roman numerals
            r'\n\s*II\.\s+',
            r'\n\s*III\.\s+',
            r'\n\s*IV\.\s+',
            r'\n\s*V\.\s+',
            r'\n\s*A\.\s+',  # Letter sections
            r'\n\s*B\.\s+',
            r'\n\s*C\.\s+',
            r'\n\s*\d+\.\s+',  # Numbered sections
        ]
        
        # Try to split by section headers
        combined_pattern = '|'.join(f'({pattern})' for pattern in section_patterns)
        
        sections = re.split(combined_pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up sections and remove empty ones
        cleaned_sections = []
        current_section = ""
        
        for section in sections:
            if section is None:
                continue
                
            section = section.strip()
            if not section:
                continue
            
            # If this looks like a header, start a new section
            if any(re.match(pattern.strip(), section, re.IGNORECASE) for pattern in section_patterns):
                if current_section:
                    cleaned_sections.append(current_section.strip())
                current_section = section
            else:
                current_section += " " + section
        
        # Add the last section
        if current_section:
            cleaned_sections.append(current_section.strip())
        
        # If no clear sections found, split by paragraphs
        if len(cleaned_sections) <= 1:
            paragraphs = text.split('\n\n')
            cleaned_sections = [p.strip() for p in paragraphs if p.strip()]
        
        return cleaned_sections
    
    def _split_section_by_size(self, section: str, document_id: str, section_index: int) -> List[DocumentChunk]:
        """Split a section into appropriately sized chunks."""
        if not section:
            return []
        
        # Rough token estimation (1 token ≈ 4 characters)
        estimated_tokens = len(section) // 4
        
        if estimated_tokens <= self.max_chunk_size:
            # Section fits in one chunk
            return [DocumentChunk(
                document_id=document_id,
                chunk_id=f"{document_id}_chunk_{section_index}_0",
                content=section,
                chunk_index=section_index * 100,  # Leave room for sub-chunks
                token_count=estimated_tokens
            )]
        
        # Split large section into smaller chunks
        sentences = self._split_into_sentences(section)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_counter = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence) // 4
            
            # If adding this sentence would exceed max size, start new chunk
            if current_tokens + sentence_tokens > self.target_chunk_size and current_chunk:
                chunks.append(DocumentChunk(
                    document_id=document_id,
                    chunk_id=f"{document_id}_chunk_{section_index}_{chunk_counter}",
                    content=current_chunk.strip(),
                    chunk_index=section_index * 100 + chunk_counter,
                    token_count=current_tokens
                ))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.overlap_size * 4)
                current_chunk = overlap_text + " " + sentence
                current_tokens = len(current_chunk) // 4
                chunk_counter += 1
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                document_id=document_id,
                chunk_id=f"{document_id}_chunk_{section_index}_{chunk_counter}",
                content=current_chunk.strip(),
                chunk_index=section_index * 100 + chunk_counter,
                token_count=current_tokens
            ))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be improved with NLTK
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str, max_chars: int) -> str:
        """Get the last portion of text for overlap between chunks."""
        if len(text) <= max_chars:
            return text
        
        # Try to break at sentence boundary
        truncated = text[-max_chars:]
        sentence_start = truncated.find('. ')
        
        if sentence_start > 0:
            return truncated[sentence_start + 2:]
        
        return truncated
