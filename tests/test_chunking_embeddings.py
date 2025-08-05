#!/usr/bin/env python3
"""
Comprehensive tests for chunking and embedding functionality
Tests cover chunking logic, embedding generation, similarity calculation, and database operations
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest
import json
import psycopg2
from unittest.mock import Mock, patch, MagicMock
from chunking_embeddings import ChunkingEmbeddingProcessor


class TestChunkingEmbeddingProcessor:
    """Test suite for ChunkingEmbeddingProcessor"""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance for testing"""
        return ChunkingEmbeddingProcessor(max_tokens_per_chunk=512)
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing chunking"""
        return """
        This is a sample Federal Register document for testing chunking functionality.
        It contains multiple sentences and should be split into appropriate chunks.
        The chunking algorithm should respect token boundaries and create manageable segments.
        Each chunk should contain approximately 512 tokens or fewer for optimal embedding generation.
        This text is designed to test various edge cases and ensure robust chunking behavior.
        """ * 50  # Make it long enough to require multiple chunks
    
    @pytest.fixture
    def sample_embedding(self):
        """Sample 1024-dimensional embedding vector"""
        return [0.1] * 1024
    
    def test_chunk_text_basic(self, processor, sample_text):
        """Test basic text chunking functionality"""
        chunks = processor.chunk_text(sample_text)
        
        # Verify chunks were created
        assert len(chunks) > 0, "No chunks were created"
        
        # Verify chunk structure
        for i, chunk in enumerate(chunks):
            assert "text" in chunk, f"Chunk {i} missing 'text' field"
            assert "token_count" in chunk, f"Chunk {i} missing 'token_count' field"
            assert "chunk_index" in chunk, f"Chunk {i} missing 'chunk_index' field"
            assert chunk["chunk_index"] == i, f"Chunk {i} has incorrect index"
            assert chunk["token_count"] <= 512, f"Chunk {i} exceeds token limit: {chunk['token_count']}"
            assert len(chunk["text"].strip()) > 0, f"Chunk {i} is empty"
    
    def test_chunk_text_empty_input(self, processor):
        """Test chunking with empty or None input"""
        # Test empty string
        chunks = processor.chunk_text("")
        assert chunks == [], "Empty string should return empty list"
        
        # Test whitespace only
        chunks = processor.chunk_text("   \n\t  ")
        assert chunks == [], "Whitespace-only string should return empty list"
        
        # Test None input
        chunks = processor.chunk_text(None)
        assert chunks == [], "None input should return empty list"
    
    def test_chunk_text_single_token(self, processor):
        """Test chunking with very short text"""
        short_text = "Hello"
        chunks = processor.chunk_text(short_text)
        
        assert len(chunks) == 1, "Single token should create one chunk"
        assert chunks[0]["text"] == "Hello", "Single token chunk content incorrect"
        assert chunks[0]["token_count"] == 1, "Single token count incorrect"
        assert chunks[0]["chunk_index"] == 0, "Single token chunk index incorrect"
    
    def test_chunk_text_exact_limit(self, processor):
        """Test chunking with text exactly at token limit"""
        # Create text with exactly 512 tokens
        exact_text = " ".join(["token"] * 512)
        chunks = processor.chunk_text(exact_text)
        
        assert len(chunks) == 1, "512-token text should create exactly one chunk"
        assert chunks[0]["token_count"] == 512, "Chunk should have exactly 512 tokens"
    
    def test_chunk_text_over_limit(self, processor):
        """Test chunking with text over token limit"""
        # Create text with 1000 tokens (should create 2 chunks)
        over_text = " ".join(["token"] * 1000)
        chunks = processor.chunk_text(over_text)
        
        assert len(chunks) == 2, "1000-token text should create 2 chunks"
        assert chunks[0]["token_count"] == 512, "First chunk should have 512 tokens"
        assert chunks[1]["token_count"] == 488, "Second chunk should have 488 tokens"
    
    @patch('requests.post')
    def test_generate_embedding_success(self, mock_post, processor, sample_embedding):
        """Test successful embedding generation"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"embedding": sample_embedding}
        mock_post.return_value = mock_response
        
        result = processor.generate_embedding("test text")
        
        assert result == sample_embedding, "Should return the embedding from API"
        mock_post.assert_called_once()
        
        # Verify API call parameters
        call_args = mock_post.call_args
        assert "api/embeddings" in call_args[0][0], "Should call embeddings endpoint"
        assert call_args[1]["json"]["prompt"] == "test text", "Should pass correct text"
    
    @patch('requests.post')
    def test_generate_embedding_empty_text(self, mock_post, processor):
        """Test embedding generation with empty text"""
        result = processor.generate_embedding("")
        assert result is None, "Empty text should return None"
        mock_post.assert_not_called()
        
        result = processor.generate_embedding("   ")
        assert result is None, "Whitespace-only text should return None"
        mock_post.assert_not_called()
    
    @patch('requests.post')
    def test_generate_embedding_api_failure(self, mock_post, processor):
        """Test embedding generation with API failure"""
        # Mock API failure
        mock_post.side_effect = Exception("API Error")
        
        result = processor.generate_embedding("test text")
        assert result is None, "API failure should return None"
        
        # Should retry 3 times
        assert mock_post.call_count == 3, "Should retry 3 times on failure"
    
    @patch('requests.post')
    def test_generate_embedding_invalid_dimensions(self, mock_post, processor):
        """Test embedding generation with wrong dimensions"""
        # Mock response with wrong embedding size
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"embedding": [0.1] * 512}  # Wrong size
        mock_post.return_value = mock_response
        
        result = processor.generate_embedding("test text")
        assert result is None, "Wrong embedding dimensions should return None"
    
    def test_calculate_similarity_score_basic(self, processor, sample_embedding):
        """Test basic similarity score calculation"""
        # Test identical embeddings (should be 1.0)
        criteria_embeddings = [sample_embedding]
        score = processor.calculate_similarity_score(sample_embedding, criteria_embeddings)
        assert abs(score - 1.0) < 0.001, "Identical embeddings should have similarity ~1.0"
    
    def test_calculate_similarity_score_orthogonal(self, processor):
        """Test similarity with orthogonal vectors"""
        # Create orthogonal vectors (should be 0.0 similarity)
        embedding1 = [1.0] + [0.0] * 1023
        embedding2 = [0.0] + [1.0] + [0.0] * 1022
        
        score = processor.calculate_similarity_score(embedding1, [embedding2])
        assert abs(score) < 0.001, "Orthogonal vectors should have similarity ~0.0"
    
    def test_calculate_similarity_score_empty_inputs(self, processor):
        """Test similarity calculation with empty inputs"""
        sample_embedding = [0.1] * 1024
        
        # Empty embedding
        score = processor.calculate_similarity_score([], [sample_embedding])
        assert score == 0.0, "Empty embedding should return 0.0"
        
        # Empty criteria
        score = processor.calculate_similarity_score(sample_embedding, [])
        assert score == 0.0, "Empty criteria should return 0.0"
        
        # Both empty
        score = processor.calculate_similarity_score([], [])
        assert score == 0.0, "Both empty should return 0.0"
    
    def test_calculate_similarity_score_multiple_criteria(self, processor):
        """Test similarity with multiple criteria (should return max)"""
        base_embedding = [1.0] + [0.0] * 1023
        
        # Create criteria with different similarities
        criteria1 = [0.5] + [0.0] * 1023  # Lower similarity
        criteria2 = [0.9] + [0.0] * 1023  # Higher similarity
        
        score = processor.calculate_similarity_score(base_embedding, [criteria1, criteria2])
        
        # Should return the higher similarity
        assert score > 0.8, "Should return maximum similarity among criteria"
    
    @patch.object(ChunkingEmbeddingProcessor, 'get_db_connection')
    def test_load_impact_criteria_embeddings(self, mock_get_db, processor, sample_embedding):
        """Test loading impact criteria embeddings from database"""
        # Mock database connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_db.return_value = mock_conn
        
        # Mock database results
        mock_cursor.fetchall.return_value = [(sample_embedding,), ([0.2] * 1024,)]
        
        result = processor.load_impact_criteria_embeddings()
        
        assert len(result) == 2, "Should return 2 embeddings"
        assert result[0] == sample_embedding, "First embedding should match"
        assert result[1] == [0.2] * 1024, "Second embedding should match"
        
        # Verify database query
        mock_cursor.execute.assert_called_once_with(
            "SELECT embedding FROM impact_criteria WHERE embedding IS NOT NULL"
        )
    
    @patch.object(ChunkingEmbeddingProcessor, 'get_db_connection')
    @patch.object(ChunkingEmbeddingProcessor, 'chunk_text')
    @patch.object(ChunkingEmbeddingProcessor, 'generate_embedding')
    @patch.object(ChunkingEmbeddingProcessor, 'load_impact_criteria_embeddings')
    @patch.object(ChunkingEmbeddingProcessor, 'calculate_similarity_score')
    def test_process_document_chunks_success(self, mock_calc_sim, mock_load_criteria, 
                                           mock_gen_embed, mock_chunk, mock_get_db, 
                                           processor, sample_embedding):
        """Test successful document chunk processing"""
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_db.return_value = mock_conn
        
        # Mock document data
        sample_doc = {
            "title": "Test Document",
            "body_html": "<p>This is test content for chunking</p>",
            "agencies": [{"name": "Test Agency"}]
        }
        mock_cursor.fetchone.return_value = (sample_doc,)
        
        # Mock chunking
        mock_chunk.return_value = [
            {"text": "chunk1", "token_count": 100, "chunk_index": 0},
            {"text": "chunk2", "token_count": 150, "chunk_index": 1}
        ]
        
        # Mock embedding generation
        mock_gen_embed.return_value = sample_embedding
        
        # Mock criteria loading
        mock_load_criteria.return_value = [sample_embedding]
        
        # Mock similarity calculation
        mock_calc_sim.return_value = 0.8
        
        # Test processing
        result = processor.process_document_chunks("test-slug")
        
        assert result is True, "Processing should succeed"
        
        # Verify database operations
        assert mock_cursor.execute.call_count >= 3, "Should make multiple database calls"
        mock_conn.commit.assert_called()
    
    @patch.object(ChunkingEmbeddingProcessor, 'get_db_connection')
    def test_process_document_chunks_not_found(self, mock_get_db, processor):
        """Test processing non-existent document"""
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_db.return_value = mock_conn
        
        # Mock document not found
        mock_cursor.fetchone.return_value = None
        
        result = processor.process_document_chunks("non-existent-slug")
        
        assert result is False, "Should return False for non-existent document"
    
    @patch.object(ChunkingEmbeddingProcessor, 'get_db_connection')
    def test_process_all_pending_documents(self, mock_get_db, processor):
        """Test processing all pending documents"""
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_db.return_value = mock_conn
        
        # Mock pending documents
        mock_cursor.fetchall.return_value = [("slug1",), ("slug2",), ("slug3",)]
        
        # Mock successful processing
        with patch.object(processor, 'process_document_chunks', return_value=True):
            stats = processor.process_all_pending_documents()
        
        assert stats["total"] == 3, "Should process 3 documents"
        assert stats["successful"] == 3, "All should succeed"
        assert stats["failed"] == 0, "None should fail"


def test_integration_chunking_pipeline():
    """Integration test for the complete chunking pipeline"""
    processor = ChunkingEmbeddingProcessor(max_tokens_per_chunk=10)  # Small chunks for testing
    
    # Test text that should create multiple chunks
    test_text = "This is a test document. " * 20  # 100 tokens, should create ~10 chunks
    
    # Test chunking
    chunks = processor.chunk_text(test_text)
    assert len(chunks) >= 5, "Should create multiple chunks"
    
    # Verify all chunks are within token limit
    for chunk in chunks:
        assert chunk["token_count"] <= 10, f"Chunk exceeds token limit: {chunk['token_count']}"
    
    # Verify chunks cover all content
    total_tokens = sum(chunk["token_count"] for chunk in chunks)
    original_tokens = len(test_text.split())
    assert total_tokens == original_tokens, "Chunks should cover all original tokens"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
