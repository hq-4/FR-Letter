"""
SQLite database for Federal Register document storage and management.
"""
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class FederalRegisterDB:
    """SQLite database for storing RSS feeds and XML documents."""
    
    def __init__(self, db_path: str = "federal_register.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with self.get_connection() as conn:
            # RSS dumps table - stores unique RSS feed snapshots
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rss_dumps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fetch_timestamp DATETIME NOT NULL,
                    rss_content TEXT NOT NULL,
                    document_count INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(fetch_timestamp)
                )
            """)
            
            # Documents table - stores individual document metadata and XML
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_number TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    agency TEXT,
                    publication_date DATE,
                    rss_link TEXT NOT NULL,
                    xml_url TEXT NOT NULL,
                    xml_content TEXT,
                    xml_size INTEGER,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Document chunks table - stores hierarchical chunks for embedding
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_type TEXT NOT NULL,  -- 'preamble', 'rule', 'section', 'paragraph'
                    chunk_level INTEGER NOT NULL,  -- hierarchical depth
                    parent_chunk_id INTEGER,
                    title TEXT,
                    content TEXT NOT NULL,
                    xml_path TEXT,  -- XPath to original element
                    token_count INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id),
                    FOREIGN KEY (parent_chunk_id) REFERENCES document_chunks (id)
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_number ON documents(document_number)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_date ON documents(publication_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_processed ON documents(processed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON document_chunks(chunk_type)")
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()
    
    def store_rss_dump(self, rss_content: str, document_count: int, fetch_timestamp: datetime) -> int:
        """Store a unique RSS dump."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT OR IGNORE INTO rss_dumps (fetch_timestamp, rss_content, document_count)
                VALUES (?, ?, ?)
            """, (fetch_timestamp, rss_content, document_count))
            
            if cursor.rowcount > 0:
                dump_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Stored RSS dump {dump_id} with {document_count} documents")
                return dump_id
            else:
                # Get existing dump ID
                cursor = conn.execute(
                    "SELECT id FROM rss_dumps WHERE fetch_timestamp = ?",
                    (fetch_timestamp,)
                )
                row = cursor.fetchone()
                return row['id'] if row else None
    
    def store_document(self, document_number: str, title: str, agency: str,
                      publication_date: str, rss_link: str, xml_url: str) -> int:
        """Store document metadata."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT OR REPLACE INTO documents 
                (document_number, title, agency, publication_date, rss_link, xml_url, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (document_number, title, agency, publication_date, rss_link, xml_url))
            
            doc_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Stored document {document_number} (ID: {doc_id})")
            return doc_id
    
    def update_document_xml(self, document_id: int, xml_content: str) -> None:
        """Update document with downloaded XML content."""
        xml_size = len(xml_content.encode('utf-8'))
        
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE documents 
                SET xml_content = ?, xml_size = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (xml_content, xml_size, document_id))
            conn.commit()
            logger.info(f"Updated document {document_id} with XML content ({xml_size} bytes)")
    
    def get_unprocessed_documents(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get documents that haven't been processed yet."""
        query = """
            SELECT id, document_number, title, agency, publication_date, 
                   rss_link, xml_url, xml_content, xml_size
            FROM documents 
            WHERE processed = FALSE AND xml_content IS NOT NULL
            ORDER BY publication_date DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        with self.get_connection() as conn:
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]
    
    def store_document_chunks(self, document_id: int, chunks: List[Dict[str, Any]]) -> None:
        """Store document chunks for embedding."""
        with self.get_connection() as conn:
            # Clear existing chunks for this document
            conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
            
            # Insert new chunks
            for chunk in chunks:
                conn.execute("""
                    INSERT INTO document_chunks 
                    (document_id, chunk_type, chunk_level, parent_chunk_id, 
                     title, content, xml_path, token_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    document_id,
                    chunk['chunk_type'],
                    chunk['chunk_level'],
                    chunk.get('parent_chunk_id'),
                    chunk.get('title'),
                    chunk['content'],
                    chunk.get('xml_path'),
                    chunk.get('token_count', 0)
                ))
            
            # Mark document as processed
            conn.execute(
                "UPDATE documents SET processed = TRUE WHERE id = ?",
                (document_id,)
            )
            
            conn.commit()
            logger.info(f"Stored {len(chunks)} chunks for document {document_id}")
    
    def get_document_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, chunk_type, chunk_level, parent_chunk_id,
                       title, content, xml_path, token_count
                FROM document_chunks 
                WHERE document_id = ?
                ORDER BY chunk_level, id
            """, (document_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        with self.get_connection() as conn:
            stats = {}
            
            # RSS dumps
            cursor = conn.execute("SELECT COUNT(*) as count FROM rss_dumps")
            stats['rss_dumps'] = cursor.fetchone()['count']
            
            # Documents
            cursor = conn.execute("SELECT COUNT(*) as count FROM documents")
            stats['total_documents'] = cursor.fetchone()['count']
            
            cursor = conn.execute("SELECT COUNT(*) as count FROM documents WHERE xml_content IS NOT NULL")
            stats['documents_with_xml'] = cursor.fetchone()['count']
            
            cursor = conn.execute("SELECT COUNT(*) as count FROM documents WHERE processed = TRUE")
            stats['processed_documents'] = cursor.fetchone()['count']
            
            # Chunks
            cursor = conn.execute("SELECT COUNT(*) as count FROM document_chunks")
            stats['total_chunks'] = cursor.fetchone()['count']
            
            return stats
