#!/usr/bin/env python3
"""
Test script to debug the Federal Register pipeline step by step.
"""

import sys
import os
from datetime import date, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fr_monitor.ingestion.rss_wrapper import FederalRegisterClient
from fr_monitor.scoring.impact_scorer import ImpactScorer
from fr_monitor.summarization.document_chunker import DocumentChunker
from fr_monitor.summarization.local_summarizer import LocalSummarizer

def test_pipeline_steps():
    """Test each pipeline step individually to identify where the issue occurs."""
    print("🔍 Testing Federal Register Pipeline Step by Step")
    print("=" * 60)
    
    # Step 1: Test Federal Register RSS ingestion
    print("\n1️⃣ Testing Federal Register RSS ingestion...")
    client = FederalRegisterClient()
    
    try:
        # Use RSS feed approach (no date needed - gets recent documents)
        print("   Fetching recent documents from RSS feed...")
        documents = client.get_daily_documents()  # Uses RSS by default now
        
        if documents:
            print(f"✅ Fetched {len(documents)} recent documents from RSS feed")
        else:
            print("❌ No documents found in RSS feed")
            return False
            
    except Exception as e:
        print(f"❌ RSS ingestion failed: {e}")
        return False
        
    # Show sample document
    sample_doc = documents[0]
    print(f"📄 Sample document: {sample_doc.title[:100]}...")
    print(f"   Abstract length: {len(sample_doc.abstract) if sample_doc.abstract else 0} chars")
    
    # Step 2: Test impact scoring
    print("\n2️⃣ Testing impact scoring...")
    try:
        scorer = ImpactScorer()
        top_documents = scorer.get_top_documents(documents, 5)
        print(f"✅ Scored documents, top 5 selected: {len(top_documents)}")
        
        for i, doc in enumerate(top_documents[:3]):
            scores = scorer.score_documents([doc])
            if scores:
                score = scores[0].total_score
                print(f"   Doc {i+1}: {doc.title[:50]}... (score: {score:.2f})")
        
    except Exception as e:
        print(f"❌ Impact scoring failed: {e}")
        return False
    
    # Step 3: Test document chunking
    print("\n3️⃣ Testing document chunking...")
    try:
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(top_documents[:2])  # Test with 2 docs
        print(f"✅ Generated {len(chunks)} chunks from {len(top_documents[:2])} documents")
        
        if chunks:
            sample_chunk = chunks[0]
            print(f"   Sample chunk: {len(sample_chunk.content)} chars")
            print(f"   Content preview: {sample_chunk.content[:100]}...")
        
    except Exception as e:
        print(f"❌ Document chunking failed: {e}")
        return False
    
    # Step 4: Test local summarization
    print("\n4️⃣ Testing local summarization...")
    try:
        local_summarizer = LocalSummarizer()
        
        # Test chunk summarization
        chunk_summaries = local_summarizer.summarize_chunks(chunks[:3])  # Test with 3 chunks
        print(f"✅ Generated {len(chunk_summaries)} chunk summaries")
        
        if chunk_summaries:
            sample_summary = chunk_summaries[0]
            print(f"   Sample summary length: {len(sample_summary.summary)} chars")
            print(f"   Summary preview: {sample_summary.summary[:100]}...")
        
        # Test consolidation
        consolidated = local_summarizer.consolidate_summaries(chunk_summaries)
        print(f"✅ Consolidated into {len(consolidated)} document summaries")
        
        for i, summary in enumerate(consolidated):
            content_length = len(summary.summary) if summary.summary else 0
            print(f"   Consolidated {i+1}: '{summary.document_id[:50]}...' ({content_length} chars)")
            if content_length > 0:
                print(f"      Preview: {summary.summary[:100]}...")
            else:
                print(f"      ⚠️  EMPTY CONSOLIDATED SUMMARY!")
        
        # Step 5: Check if we have valid content for OpenRouter
        print("\n5️⃣ Checking content for OpenRouter...")
        valid_summaries = [s for s in consolidated if s.summary and len(s.summary.strip()) > 10]
        print(f"📊 Valid summaries for OpenRouter: {len(valid_summaries)}/{len(consolidated)}")
        
        if not valid_summaries:
            print("❌ No valid summaries found - this explains the short OpenRouter calls!")
            print("   The consolidated summaries are empty or too short.")
            return False
    
    except Exception as e:
        print(f"❌ Local summarization failed: {e}")
        return False
    
    
    print("\n✅ Pipeline test completed successfully!")
    print(f"📈 Summary: {len(documents)} docs → {len(top_documents)} scored → {len(chunks)} chunks → {len(consolidated)} summaries → {len(valid_summaries)} valid")
    
    return True

if __name__ == "__main__":
    success = test_pipeline_steps()
    if success:
        print("\n🎉 Pipeline appears to be working correctly!")
        print("   If you're still having issues, the problem might be in:")
        print("   - Environment variables (OPENROUTER_API_KEY missing?)")
        print("   - Redis connectivity")
        print("   - Ollama models not available")
    else:
        print("\n🚨 Pipeline has issues that need to be fixed!")
    
    sys.exit(0 if success else 1)
