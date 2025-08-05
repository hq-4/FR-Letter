#!/usr/bin/env python3
"""
Test script for Federal Register .txt content processing workflow.
Tests the complete pipeline: JSON ingestion -> .txt fetching -> chunking -> embedding.

Created: 2025-08-05
Updated for .txt content processing architecture
"""

import os
import sys
import json
import psycopg2
import requests
from datetime import datetime
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chunking_embeddings import ChunkingEmbeddingProcessor

load_dotenv()

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        database=os.getenv("POSTGRES_DB", "federalregister"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "")
    )

def test_txt_content_ingestion():
    """Test ingesting a document with .txt content fetching"""
    print("🧪 Testing .txt content ingestion and processing...")
    
    # Use the sample document we have
    test_slug = "2025-14564"
    
    # Load the JSON metadata
    with open(f"{test_slug}.json", "r") as f:
        document_json = json.load(f)
    
    print(f"📄 Test document: {test_slug}")
    print(f"   Title: {document_json.get('title', 'N/A')[:80]}...")
    print(f"   Raw text URL: {document_json.get('raw_text_url', 'N/A')}")
    print(f"   Page length: {document_json.get('page_length', 'N/A')}")
    
    # Insert document into database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # [CA] Insert document with JSON metadata
    cursor.execute("""
        INSERT INTO processed_documents (slug, raw_json, publication_date, embedding_status)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (slug) DO UPDATE SET
            raw_json = EXCLUDED.raw_json,
            embedding_status = EXCLUDED.embedding_status,
            updated_at = CURRENT_TIMESTAMP
    """, (test_slug, json.dumps(document_json), document_json.get('publication_date'), 'pending'))
    
    conn.commit()
    print(f"✅ Inserted document {test_slug} into database")
    
    # Test the chunking processor
    processor = ChunkingEmbeddingProcessor(max_tokens_per_chunk=256)  # Smaller chunks for testing
    
    print(f"🔄 Processing document with chunking processor...")
    success = processor.process_document_chunks(test_slug)
    
    if success:
        print(f"✅ Document processing completed successfully")
        
        # Verify results
        cursor.execute("""
            SELECT title, word_count, chunk_count, impact_score, embedding_status,
                   LENGTH(full_text_content) as content_length
            FROM processed_documents 
            WHERE slug = %s
        """, (test_slug,))
        
        doc_result = cursor.fetchone()
        if doc_result:
            title, word_count, chunk_count, impact_score, status, content_length = doc_result
            print(f"📊 Document Results:")
            print(f"   Title: {title[:60]}...")
            print(f"   Word count: {word_count:,}")
            print(f"   Content length: {content_length:,} characters")
            print(f"   Chunks generated: {chunk_count}")
            print(f"   Impact score: {impact_score:.4f}")
            print(f"   Status: {status}")
            
            # Check chunks
            cursor.execute("""
                SELECT chunk_index, token_count, start_position, end_position,
                       CASE WHEN embedding IS NOT NULL THEN 'YES' ELSE 'NO' END as has_embedding,
                       similarity_score,
                       LEFT(chunk_text, 80) as preview
                FROM document_chunks 
                WHERE document_slug = %s
                ORDER BY chunk_index
                LIMIT 5
            """, (test_slug,))
            
            chunks = cursor.fetchall()
            print(f"\n📝 Chunk Details (showing first 5):")
            print("Idx  Tokens  Pos Range    Embedding  Similarity  Preview")
            print("-" * 80)
            for chunk in chunks:
                idx, tokens, start, end, has_emb, sim, preview = chunk
                print(f"{idx:3d}  {tokens:6d}  {start:6d}-{end:6d}  {has_emb:9s}  {sim:8.4f}  {preview}...")
            
            # Validate expectations
            if word_count > 10000:  # Should be substantial .txt content
                print(f"✅ PASS: Document has substantial content ({word_count:,} words)")
            else:
                print(f"❌ FAIL: Expected substantial content, got {word_count} words")
                
            if chunk_count > 10:  # Should generate many chunks
                print(f"✅ PASS: Document generated multiple chunks ({chunk_count})")
            else:
                print(f"❌ FAIL: Expected many chunks, got {chunk_count}")
                
            if len(chunks) == min(chunk_count, 5):
                print(f"✅ PASS: Chunks properly stored in database")
            else:
                print(f"❌ FAIL: Chunk storage mismatch")
                
        else:
            print("❌ FAIL: Document not found after processing")
            
    else:
        print(f"❌ Document processing failed")
    
    conn.close()
    return success

def test_chunking_logic():
    """Test the chunking logic with position tracking"""
    print("\n🧪 Testing chunking logic with position tracking...")
    
    processor = ChunkingEmbeddingProcessor(max_tokens_per_chunk=50)
    
    # Test text with known structure
    test_text = "This is a test document. " * 100  # 500 words
    
    chunks = processor.chunk_text(test_text)
    
    print(f"📊 Chunking Results:")
    print(f"   Input: {len(test_text.split())} words, {len(test_text)} characters")
    print(f"   Output: {len(chunks)} chunks")
    
    # Validate chunks
    total_tokens = sum(chunk['token_count'] for chunk in chunks)
    expected_chunks = len(test_text.split()) // 50 + (1 if len(test_text.split()) % 50 else 0)
    
    print(f"   Expected chunks: ~{expected_chunks}")
    print(f"   Total tokens: {total_tokens}")
    
    # Check position tracking
    for i, chunk in enumerate(chunks[:3]):  # Show first 3
        print(f"   Chunk {i}: {chunk['token_count']} tokens, pos {chunk['start_position']}-{chunk['end_position']}")
        
    if len(chunks) >= expected_chunks - 1:  # Allow for rounding
        print("✅ PASS: Chunking generated expected number of chunks")
    else:
        print(f"❌ FAIL: Expected ~{expected_chunks} chunks, got {len(chunks)}")
        
    if total_tokens == len(test_text.split()):
        print("✅ PASS: All tokens accounted for in chunks")
    else:
        print(f"❌ FAIL: Token count mismatch: {total_tokens} vs {len(test_text.split())}")

def main():
    """Run all tests"""
    print("🚀 Federal Register .txt Content Processing Tests")
    print("=" * 60)
    
    try:
        # Test 1: Chunking logic
        test_chunking_logic()
        
        # Test 2: Full pipeline with .txt content
        test_txt_content_ingestion()
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
