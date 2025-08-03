#!/usr/bin/env python3
"""
Test script to demonstrate ranking and filtering functionality
with the current embedded Federal Register documents.
"""

import logging
from fr_monitor.embeddings.bge_embeddings import DocumentEmbeddingProcessor
from fr_monitor.storage.database import FederalRegisterDB

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environmental_ranking():
    """Test environmental regulation ranking with NY/NJ filters."""
    print("=== Federal Register Environmental Ranking Demo ===\n")
    
    # Initialize components
    processor = DocumentEmbeddingProcessor()
    db = FederalRegisterDB()
    
    # Environmental search queries from our selection criteria
    environmental_queries = [
        "environmental impact assessment",
        "water quality standards", 
        "air pollution control",
        "environmental justice",
        "climate change regulation",
        "toxic substances control",
        "waste management",
        "endangered species protection"
    ]
    
    print("🔍 Testing Environmental Regulation Queries:\n")
    
    all_results = []
    
    for query in environmental_queries:
        print(f"Query: '{query}'")
        
        # Perform semantic search
        results = processor.vector_store.search_by_text(query, limit=5)
        
        if results:
            print(f"  ✅ Found {len(results)} results")
            for i, result in enumerate(results[:3]):  # Show top 3
                score = result.get('similarity_score', 0)
                doc_num = result['document_number']
                agency = result.get('agency', 'Unknown')
                content_preview = result['content'][:120].replace('\n', ' ')
                
                print(f"    {i+1}. Score: {score:.3f} | {doc_num} | {agency}")
                print(f"       {content_preview}...")
                
                # Add to all results for ranking
                result['query'] = query
                result['environmental_relevance'] = score
                all_results.append(result)
        else:
            print(f"  ❌ No results found")
        
        print()
    
    # Demonstrate ranking and filtering
    print("\n🏆 RANKING DEMONSTRATION:\n")
    
    if all_results:
        # Sort by similarity score (environmental relevance)
        ranked_results = sorted(all_results, key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        print("Top 10 Most Environmentally Relevant Chunks:")
        print("-" * 80)
        
        for i, result in enumerate(ranked_results[:10]):
            score = result.get('similarity_score', 0)
            doc_num = result['document_number']
            agency = result.get('agency', 'Unknown')
            query = result.get('query', 'Unknown')
            chunk_type = result.get('chunk_type', 'Unknown')
            
            # Calculate priority score based on our criteria
            priority_score = calculate_priority_score(result)
            
            print(f"{i+1:2d}. Priority: {priority_score:.1f} | Similarity: {score:.3f}")
            print(f"    Document: {doc_num} | Agency: {agency}")
            print(f"    Query: '{query}' | Type: {chunk_type}")
            print(f"    Content: {result['content'][:100].replace(chr(10), ' ')}...")
            print()
    
    # Test geographic filtering (NY/NJ focus)
    print("\n🗺️  GEOGRAPHIC FILTERING (NY/NJ Focus):\n")
    
    ny_nj_queries = [
        "New York environmental regulation",
        "New Jersey water quality", 
        "Hudson River pollution",
        "Long Island environmental impact"
    ]
    
    for query in ny_nj_queries:
        print(f"Query: '{query}'")
        results = processor.vector_store.search_by_text(query, limit=3)
        
        if results:
            print(f"  ✅ Found {len(results)} results")
            for i, result in enumerate(results):
                score = result.get('similarity_score', 0)
                doc_num = result['document_number']
                content_preview = result['content'][:100].replace('\n', ' ')
                print(f"    {i+1}. Score: {score:.3f} | {doc_num}")
                print(f"       {content_preview}...")
        else:
            print(f"  ❌ No results found")
        print()

def calculate_priority_score(result):
    """Calculate priority score based on our environmental criteria."""
    base_score = result.get('similarity_score', 0) * 10  # Scale to 0-10
    
    # Agency priority multipliers (from selection_criteria.md)
    agency_multipliers = {
        'Environmental Protection Agency': 1.5,
        'EPA': 1.5,
        'Department of the Interior': 1.3,
        'Department of Commerce': 1.2,
        'Department of Transportation': 1.1,
        'Coast Guard': 1.2
    }
    
    agency = result.get('agency', '').lower()
    multiplier = 1.0
    
    for key_agency, mult in agency_multipliers.items():
        if key_agency.lower() in agency:
            multiplier = mult
            break
    
    # Content-based priority boosts
    content = result.get('content', '').lower()
    
    # High priority keywords
    high_priority_keywords = [
        'environmental justice', 'climate change', 'toxic', 'pollution',
        'endangered species', 'water quality', 'air quality'
    ]
    
    keyword_boost = sum(0.2 for keyword in high_priority_keywords if keyword in content)
    
    # Geographic relevance (NY/NJ)
    geographic_keywords = ['new york', 'new jersey', 'hudson', 'long island', 'ny', 'nj']
    geo_boost = sum(0.3 for geo in geographic_keywords if geo in content)
    
    final_score = (base_score * multiplier) + keyword_boost + geo_boost
    return min(final_score, 10.0)  # Cap at 10

def show_current_data_stats():
    """Show statistics about current embedded data."""
    print("\n📊 CURRENT DATA STATISTICS:\n")
    
    db = FederalRegisterDB()
    
    # Get document stats
    with db.get_connection() as conn:
        # Total documents
        total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        
        # Documents with XML
        xml_docs = conn.execute("SELECT COUNT(*) FROM documents WHERE xml_content IS NOT NULL").fetchone()[0]
        
        # Documents with chunks
        chunked_docs = conn.execute("""
            SELECT COUNT(DISTINCT document_id) FROM document_chunks
        """).fetchone()[0]
        
        # Total chunks
        total_chunks = conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]
        
        # Agencies represented
        agencies = conn.execute("""
            SELECT agency, COUNT(*) as count 
            FROM documents 
            WHERE agency IS NOT NULL 
            GROUP BY agency 
            ORDER BY count DESC 
            LIMIT 10
        """).fetchall()
    
    print(f"Total Documents: {total_docs}")
    print(f"Documents with XML: {xml_docs}")
    print(f"Documents Chunked: {chunked_docs}")
    print(f"Total Chunks: {total_chunks}")
    print(f"Chunks per Document (avg): {total_chunks/chunked_docs if chunked_docs > 0 else 0:.1f}")
    
    print(f"\nTop Agencies:")
    for agency, count in agencies:
        print(f"  {agency}: {count} documents")

if __name__ == "__main__":
    show_current_data_stats()
    test_environmental_ranking()
    print("\n=== Ranking Demo Complete ===")
