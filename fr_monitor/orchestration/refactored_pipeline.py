"""
Refactored Federal Register monitoring pipeline.
Uses SQLite for document storage, hierarchical XML chunking, and BGE-large embeddings.
"""
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..ingestion.rss_ingestion import RSSIngestionClient
from ..processing.xml_chunker import FederalRegisterXMLChunker
from ..embeddings.bge_embeddings import DocumentEmbeddingProcessor
from ..storage.database import FederalRegisterDB

logger = logging.getLogger(__name__)


class RefactoredFederalRegisterPipeline:
    """Main pipeline for Federal Register document processing."""
    
    def __init__(self, db_path: str = "federal_register.db", 
                 max_chunk_tokens: int = 2048):
        self.db_path = db_path
        self.db = FederalRegisterDB(db_path)
        self.rss_client = RSSIngestionClient(db_path)
        self.xml_chunker = FederalRegisterXMLChunker(max_chunk_tokens)
        self.embedding_processor = DocumentEmbeddingProcessor(db_path)
        
        logger.info("Initialized refactored Federal Register pipeline")
    
    def run_full_pipeline(self, download_limit: Optional[int] = None,
                         chunk_limit: Optional[int] = None,
                         embed_limit: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete pipeline: ingest -> chunk -> embed."""
        start_time = time.time()
        results = {
            "start_time": datetime.now().isoformat(),
            "steps": {},
            "errors": []
        }
        
        try:
            # Step 1: Ingest RSS feed and store documents
            logger.info("=== Step 1: RSS Ingestion ===")
            step_start = time.time()
            
            ingested_count = self.rss_client.fetch_and_store_rss()
            results["steps"]["rss_ingestion"] = {
                "documents_ingested": ingested_count,
                "duration_seconds": time.time() - step_start
            }
            
            # Step 2: Download XML content for documents
            logger.info("=== Step 2: XML Download ===")
            step_start = time.time()
            
            downloaded_count = self.rss_client.download_xml_content(limit=download_limit)
            results["steps"]["xml_download"] = {
                "documents_downloaded": downloaded_count,
                "duration_seconds": time.time() - step_start
            }
            
            # Step 3: Chunk XML documents
            logger.info("=== Step 3: XML Chunking ===")
            step_start = time.time()
            
            chunked_count = self._chunk_documents(limit=chunk_limit)
            results["steps"]["xml_chunking"] = {
                "documents_chunked": chunked_count,
                "duration_seconds": time.time() - step_start
            }
            
            # Step 4: Generate embeddings and store in Redis
            logger.info("=== Step 4: Embedding Generation ===")
            step_start = time.time()
            
            embedded_count = self.embedding_processor.process_unembedded_documents(limit=embed_limit)
            results["steps"]["embedding_generation"] = {
                "documents_embedded": embedded_count,
                "duration_seconds": time.time() - step_start
            }
            
            # Get final statistics
            results["final_stats"] = self._get_pipeline_stats()
            results["total_duration_seconds"] = time.time() - start_time
            results["success"] = True
            
            logger.info(f"Pipeline completed successfully in {results['total_duration_seconds']:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results["errors"].append(str(e))
            results["success"] = False
            results["total_duration_seconds"] = time.time() - start_time
        
        return results
    
    def run_ingestion_only(self, download_limit: Optional[int] = None) -> Dict[str, Any]:
        """Run only the ingestion steps (RSS + XML download)."""
        start_time = time.time()
        results = {
            "start_time": datetime.now().isoformat(),
            "steps": {},
            "errors": []
        }
        
        try:
            # RSS ingestion
            logger.info("=== RSS Ingestion ===")
            ingested_count = self.rss_client.fetch_and_store_rss()
            
            # XML download
            logger.info("=== XML Download ===")
            downloaded_count = self.rss_client.download_xml_content(limit=download_limit)
            
            results["steps"] = {
                "rss_ingestion": {"documents_ingested": ingested_count},
                "xml_download": {"documents_downloaded": downloaded_count}
            }
            results["success"] = True
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            results["errors"].append(str(e))
            results["success"] = False
        
        results["total_duration_seconds"] = time.time() - start_time
        return results
    
    def run_processing_only(self, chunk_limit: Optional[int] = None,
                           embed_limit: Optional[int] = None) -> Dict[str, Any]:
        """Run only the processing steps (chunking + embedding)."""
        start_time = time.time()
        results = {
            "start_time": datetime.now().isoformat(),
            "steps": {},
            "errors": []
        }
        
        try:
            # XML chunking
            logger.info("=== XML Chunking ===")
            chunked_count = self._chunk_documents(limit=chunk_limit)
            
            # Embedding generation
            logger.info("=== Embedding Generation ===")
            embedded_count = self.embedding_processor.process_unembedded_documents(limit=embed_limit)
            
            results["steps"] = {
                "xml_chunking": {"documents_chunked": chunked_count},
                "embedding_generation": {"documents_embedded": embedded_count}
            }
            results["success"] = True
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            results["errors"].append(str(e))
            results["success"] = False
        
        results["total_duration_seconds"] = time.time() - start_time
        return results
    
    def _chunk_documents(self, limit: Optional[int] = None) -> int:
        """Chunk documents that have XML content but haven't been processed."""
        # Get unprocessed documents with XML content
        with self.db.get_connection() as conn:
            query = """
                SELECT id, document_number, title, agency, xml_content
                FROM documents 
                WHERE xml_content IS NOT NULL AND processed = FALSE
                ORDER BY publication_date DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor = conn.execute(query)
            documents = [dict(row) for row in cursor.fetchall()]
        
        if not documents:
            logger.info("No documents need chunking")
            return 0
        
        logger.info(f"Chunking {len(documents)} documents")
        
        chunked_count = 0
        for doc in documents:
            try:
                # Chunk the XML content
                chunks = self.xml_chunker.chunk_document(doc['xml_content'], doc['id'])
                
                if chunks:
                    # Store chunks in database
                    self.db.store_document_chunks(doc['id'], chunks)
                    chunked_count += 1
                    
                    # Log chunk summary
                    summary = self.xml_chunker.get_chunk_summary(chunks)
                    logger.info(f"Chunked document {doc['document_number']}: {summary['total_chunks']} chunks, "
                              f"{summary['total_tokens']} total tokens")
                else:
                    logger.warning(f"No chunks generated for document {doc['document_number']}")
                
            except Exception as e:
                logger.error(f"Failed to chunk document {doc['document_number']}: {e}")
                continue
        
        logger.info(f"Successfully chunked {chunked_count}/{len(documents)} documents")
        return chunked_count
    
    def search_documents(self, query: str, limit: int = 10, 
                        filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search documents using the embedding processor."""
        return self.embedding_processor.search_documents(query, limit, filters)
    
    def search_environmental_ny_nj(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for environmental regulations affecting NY/NJ using predefined filters."""
        filters = {
            "agency": ["EPA", "Interior", "Commerce", "Transportation"],
            "chunk_type": ["rule", "preamble", "regulatory_text"]
        }
        
        # Enhance query with geographic terms
        enhanced_query = f"{query} New York New Jersey environmental regulation"
        
        return self.search_documents(enhanced_query, limit, filters)
    
    def get_document_details(self, document_number: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific document."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, document_number, title, agency, publication_date,
                       rss_link, xml_url, xml_size, processed
                FROM documents 
                WHERE document_number = ?
            """, (document_number,))
            
            doc = cursor.fetchone()
            if not doc:
                return None
            
            doc_dict = dict(doc)
            
            # Get chunks if available
            if doc_dict['processed']:
                chunks = self.db.get_document_chunks(doc_dict['id'])
                doc_dict['chunks'] = chunks
                doc_dict['chunk_summary'] = self.xml_chunker.get_chunk_summary(chunks)
            
            return doc_dict
    
    def _get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        db_stats = self.db.get_stats()
        ingestion_stats = self.rss_client.get_ingestion_stats()
        processing_stats = self.embedding_processor.get_processing_stats()
        
        return {
            "database": db_stats,
            "ingestion": ingestion_stats,
            "processing": processing_stats
        }
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get a comprehensive status report of the pipeline."""
        stats = self._get_pipeline_stats()
        
        # Calculate processing rates
        total_docs = stats["database"]["total_documents"]
        docs_with_xml = stats["database"]["documents_with_xml"]
        processed_docs = stats["database"]["processed_documents"]
        
        processing_rate = (processed_docs / total_docs * 100) if total_docs > 0 else 0
        xml_download_rate = (docs_with_xml / total_docs * 100) if total_docs > 0 else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "pipeline_health": {
                "total_documents": total_docs,
                "xml_download_rate_percent": round(xml_download_rate, 2),
                "processing_rate_percent": round(processing_rate, 2),
                "documents_ready_for_search": processed_docs
            },
            "detailed_stats": stats,
            "next_actions": self._get_next_actions(stats)
        }
    
    def _get_next_actions(self, stats: Dict[str, Any]) -> List[str]:
        """Suggest next actions based on current pipeline state."""
        actions = []
        
        db_stats = stats["database"]
        
        # Check for documents needing XML download
        docs_needing_xml = db_stats["total_documents"] - db_stats["documents_with_xml"]
        if docs_needing_xml > 0:
            actions.append(f"Download XML content for {docs_needing_xml} documents")
        
        # Check for documents needing processing
        docs_needing_processing = db_stats["documents_with_xml"] - db_stats["processed_documents"]
        if docs_needing_processing > 0:
            actions.append(f"Process (chunk + embed) {docs_needing_processing} documents")
        
        # Check if we should ingest new RSS data
        actions.append("Run RSS ingestion to check for new documents")
        
        if not actions:
            actions.append("Pipeline is up to date - ready for searches")
        
        return actions
