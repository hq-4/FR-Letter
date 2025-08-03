"""
Hierarchical XML chunking for Federal Register documents.
Handles large documents by leveraging the standardized schema structure.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from xml.etree import ElementTree as ET
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class XMLChunk:
    """Represents a chunk of XML content with metadata."""
    chunk_type: str  # 'preamble', 'rule', 'section', 'paragraph', 'header'
    chunk_level: int  # hierarchical depth (0 = root, 1 = major section, etc.)
    parent_chunk_id: Optional[int]
    title: Optional[str]
    content: str
    xml_path: str  # XPath to original element
    token_count: int


class FederalRegisterXMLChunker:
    """Hierarchical chunker for Federal Register XML documents."""
    
    def __init__(self, max_chunk_tokens: int = 2048):
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_counter = 0
        
        # Element type mappings for chunking strategy
        self.chunk_type_map = {
            'PREAMB': 'preamble',
            'REGTEXT': 'regulatory_text',
            'RULE': 'rule',
            'SECTION': 'section',
            'SUBSEC': 'subsection',
            'P': 'paragraph',
            'HD': 'header',
            'LSTSUB': 'list_subject',
            'SIG': 'signature',
            'DATED': 'date_section',
            'AGENCY': 'agency_info',
            'CFR': 'cfr_citation',
            'RIN': 'rin_info'
        }
        
        # Priority elements that should be kept together
        self.priority_elements = {'AGENCY', 'CFR', 'RIN', 'DATES', 'SUMMARY'}
        
        # Elements that typically contain large amounts of text
        self.large_content_elements = {'PREAMB', 'REGTEXT', 'RULE'}
    
    def chunk_document(self, xml_content: str, document_id: int) -> List[Dict[str, Any]]:
        """Chunk a Federal Register XML document hierarchically."""
        try:
            root = ET.fromstring(xml_content)
            self.chunk_counter = 0
            
            chunks = []
            self._chunk_element_recursive(root, chunks, level=0, parent_id=None, document_id=document_id)
            
            logger.info(f"Generated {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML for document {document_id}: {e}")
            # Fallback: create a single chunk with the raw content
            return [{
                'chunk_type': 'raw_text',
                'chunk_level': 0,
                'parent_chunk_id': None,
                'title': 'Raw Document Content',
                'content': xml_content[:self.max_chunk_tokens * 4],  # Rough token estimate
                'xml_path': '/raw',
                'token_count': self._estimate_tokens(xml_content[:self.max_chunk_tokens * 4])
            }]
        except Exception as e:
            logger.error(f"Unexpected error chunking document {document_id}: {e}")
            return []
    
    def _chunk_element_recursive(self, element: ET.Element, chunks: List[Dict[str, Any]], 
                                level: int, parent_id: Optional[int], document_id: int) -> Optional[int]:
        """Recursively chunk XML elements."""
        element_tag = element.tag.upper()
        
        # Get element content (text + tail)
        element_text = self._extract_element_text(element)
        
        # Skip empty elements
        if not element_text.strip() and len(element) == 0:
            return None
        
        # Determine chunk type
        chunk_type = self.chunk_type_map.get(element_tag, 'unknown')
        
        # Create title from element attributes or content
        title = self._extract_title(element, element_text)
        
        # Get XPath
        xml_path = self._get_element_xpath(element)
        
        current_chunk_id = None
        
        # Handle large content elements that need subdivision
        if element_tag in self.large_content_elements and len(element) > 0:
            # Create a parent chunk for the container
            if element_text.strip():
                current_chunk_id = self._create_chunk(
                    chunks, chunk_type, level, parent_id, title,
                    element_text, xml_path, document_id
                )
            
            # Process children
            for child in element:
                child_chunk_id = self._chunk_element_recursive(
                    child, chunks, level + 1, current_chunk_id or parent_id, document_id
                )
                
                # If we didn't create a parent chunk, use the first child as reference
                if current_chunk_id is None and child_chunk_id is not None:
                    current_chunk_id = child_chunk_id
        
        # Handle elements that should be kept as single chunks
        elif element_tag in self.priority_elements or len(element) == 0:
            current_chunk_id = self._create_chunk(
                chunks, chunk_type, level, parent_id, title,
                element_text, xml_path, document_id
            )
        
        # Handle medium-sized elements with children
        else:
            # Collect all text content including children
            full_content = self._extract_full_element_text(element)
            token_count = self._estimate_tokens(full_content)
            
            if token_count <= self.max_chunk_tokens:
                # Small enough to be a single chunk
                current_chunk_id = self._create_chunk(
                    chunks, chunk_type, level, parent_id, title,
                    full_content, xml_path, document_id
                )
            else:
                # Too large, need to split by children
                if element_text.strip():
                    # Create parent chunk with direct text content
                    current_chunk_id = self._create_chunk(
                        chunks, chunk_type, level, parent_id, title,
                        element_text, xml_path, document_id
                    )
                
                # Process children separately
                for child in element:
                    child_chunk_id = self._chunk_element_recursive(
                        child, chunks, level + 1, current_chunk_id or parent_id, document_id
                    )
                    
                    if current_chunk_id is None and child_chunk_id is not None:
                        current_chunk_id = child_chunk_id
        
        return current_chunk_id
    
    def _create_chunk(self, chunks: List[Dict[str, Any]], chunk_type: str, level: int,
                     parent_id: Optional[int], title: Optional[str], content: str,
                     xml_path: str, document_id: int) -> int:
        """Create a chunk and add it to the list."""
        # Clean and truncate content if necessary
        content = self._clean_content(content)
        token_count = self._estimate_tokens(content)
        
        # If still too large, truncate
        if token_count > self.max_chunk_tokens:
            content = self._truncate_content(content, self.max_chunk_tokens)
            token_count = self._estimate_tokens(content)
        
        chunk_id = self.chunk_counter
        self.chunk_counter += 1
        
        chunk = {
            'chunk_type': chunk_type,
            'chunk_level': level,
            'parent_chunk_id': parent_id,
            'title': title,
            'content': content,
            'xml_path': xml_path,
            'token_count': token_count
        }
        
        chunks.append(chunk)
        return chunk_id
    
    def _extract_element_text(self, element: ET.Element) -> str:
        """Extract direct text content from element (not including children)."""
        text_parts = []
        
        if element.text:
            text_parts.append(element.text.strip())
        
        if element.tail:
            text_parts.append(element.tail.strip())
        
        return ' '.join(filter(None, text_parts))
    
    def _extract_full_element_text(self, element: ET.Element) -> str:
        """Extract all text content from element including children."""
        return ''.join(element.itertext()).strip()
    
    def _extract_title(self, element: ET.Element, content: str) -> Optional[str]:
        """Extract or generate a title for the chunk."""
        # Check for title attributes
        if 'title' in element.attrib:
            return element.attrib['title']
        
        # For headers, use the content as title
        if element.tag.upper() == 'HD':
            return content[:100] if content else None
        
        # For sections, look for section numbers or identifiers
        if element.tag.upper() in ['SECTION', 'SUBSEC']:
            # Look for section identifiers in attributes
            for attr in ['id', 'section', 'number']:
                if attr in element.attrib:
                    return f"Section {element.attrib[attr]}"
        
        # For rules, try to extract rule title
        if element.tag.upper() == 'RULE':
            # Look for first header child
            hd_elem = element.find('.//HD')
            if hd_elem is not None and hd_elem.text:
                return hd_elem.text.strip()[:100]
        
        # Generate title from content
        if content:
            # Take first sentence or first 50 characters
            first_sentence = re.split(r'[.!?]', content)[0]
            if len(first_sentence) <= 100:
                return first_sentence.strip()
            else:
                return content[:50].strip() + "..."
        
        return None
    
    def _get_element_xpath(self, element: ET.Element) -> str:
        """Generate XPath for element (simplified)."""
        # This is a simplified XPath - in a full implementation,
        # you'd want to track the full path from root
        tag = element.tag
        
        # Add position if there are attributes that help identify
        if 'id' in element.attrib:
            return f"//{tag}[@id='{element.attrib['id']}']"
        elif 'section' in element.attrib:
            return f"//{tag}[@section='{element.attrib['section']}']"
        else:
            return f"//{tag}"
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove XML artifacts
        content = re.sub(r'<[^>]+>', '', content)
        
        # Normalize quotes and dashes
        content = content.replace('"', '"').replace('"', '"')
        content = content.replace(''', "'").replace(''', "'")
        content = content.replace('—', '-').replace('–', '-')
        
        return content.strip()
    
    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within token limit."""
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        if len(content) <= max_chars:
            return content
        
        # Try to truncate at sentence boundary
        truncated = content[:max_chars]
        last_sentence = truncated.rfind('.')
        
        if last_sentence > max_chars * 0.8:  # If we can keep 80% of content
            return truncated[:last_sentence + 1]
        else:
            return truncated + "..."
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: 1 token ≈ 4 characters for English text
        # This is conservative - actual tokenization would be more accurate
        return max(1, len(text) // 4)
    
    def get_chunk_summary(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics for chunks."""
        if not chunks:
            return {}
        
        chunk_types = {}
        total_tokens = 0
        max_tokens = 0
        levels = set()
        
        for chunk in chunks:
            chunk_type = chunk['chunk_type']
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            tokens = chunk['token_count']
            total_tokens += tokens
            max_tokens = max(max_tokens, tokens)
            
            levels.add(chunk['chunk_level'])
        
        return {
            'total_chunks': len(chunks),
            'chunk_types': chunk_types,
            'total_tokens': total_tokens,
            'avg_tokens_per_chunk': total_tokens // len(chunks),
            'max_tokens_per_chunk': max_tokens,
            'hierarchy_levels': sorted(levels)
        }
