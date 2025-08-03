from fr_monitor.embeddings.bge_embeddings import RedisVectorStore, DocumentEmbeddingProcessor
from fr_monitor.storage.database import FederalRegisterDB
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recreate_index():
    logger.info("Recreating RedisSearch index...")
    
    # This will create a new index with the proper schema
    vector_store = RedisVectorStore()
    
    # Now reindex all documents
    db = FederalRegisterDB()
    processor = DocumentEmbeddingProcessor()
    
    # Get all processed documents
    with db.get_connection() as conn:
        cursor = conn.execute("""
            SELECT id, document_number, title, agency, publication_date,
                   rss_link, xml_url, xml_content
            FROM documents 
            WHERE processed = TRUE
            ORDER BY publication_date DESC
        """)
        documents = [dict(row) for row in cursor.fetchall()]
    
    logger.info(f"Reindexing {len(documents)} documents...")
    
    for doc in documents:
        try:
            # Get chunks for this document
            chunks = db.get_document_chunks(doc['id'])
            if not chunks:
                logger.warning(f"No chunks found for document {doc['document_number']}")
                continue
            
            # Get embeddings (should be fast since they're already in Redis)
            chunk_texts = [chunk['content'] for chunk in chunks]
            embeddings = processor.embedding_client.get_embeddings(chunk_texts)
            
            # Store in vector store (will update the index)
            vector_store.store_chunk_embeddings(doc, chunks, embeddings)
            
            logger.info(f"Reindexed document {doc['document_number']}")
            
        except Exception as e:
            logger.error(f"Error reindexing document {doc.get('document_number')}: {e}")
    
    logger.info("Index recreation complete!")

if __name__ == "__main__":
    recreate_index()