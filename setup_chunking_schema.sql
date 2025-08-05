-- Federal Register Document Processing Schema (Updated for .txt Content)
-- Supports ingestion, storage, chunking and embedding of Federal Register documents
-- Updated: 2025-08-05 - Added .txt content storage and processing

-- Enable pgvector extension for embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Main table for processed Federal Register documents
-- Now includes full .txt content storage alongside JSON metadata
CREATE TABLE IF NOT EXISTS processed_documents (
    id SERIAL PRIMARY KEY,
    slug VARCHAR(50) UNIQUE NOT NULL,
    raw_json JSONB NOT NULL,                    -- Original JSON metadata from API
    title TEXT,                                 -- Extracted title from JSON
    full_text_content TEXT,                     -- Full .txt document content
    raw_text_url TEXT,                          -- URL to .txt file
    body_html_url TEXT,                         -- URL to HTML version
    publication_date DATE,
    page_length INTEGER,                        -- Number of pages from JSON
    word_count INTEGER DEFAULT 0,              -- Word count of full_text_content
    chunk_count INTEGER DEFAULT 0,
    impact_score FLOAT DEFAULT 0.0,
    embedding_status VARCHAR(20) DEFAULT 'pending',
    content_fetched_at TIMESTAMP,              -- When .txt content was fetched
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document chunks table for storing 512-token segments from .txt content
-- Each chunk contains a portion of the full document text with its embedding
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_slug VARCHAR(50) NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,                  -- Text chunk from full_text_content
    token_count INTEGER NOT NULL,
    start_position INTEGER,                    -- Character position in full_text_content
    end_position INTEGER,                      -- End character position
    embedding vector(1024),                    -- BGE-large embedding (1024 dimensions)
    similarity_score FLOAT DEFAULT 0.0,       -- Similarity to impact criteria
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_slug) REFERENCES processed_documents(slug) ON DELETE CASCADE,
    UNIQUE(document_slug, chunk_index)
);

-- Impact criteria table for scoring
CREATE TABLE IF NOT EXISTS impact_criteria (
    id SERIAL PRIMARY KEY,
    criteria_text TEXT NOT NULL,
    criteria_type VARCHAR(50) DEFAULT 'general', -- general, environmental, economic, health
    weight FLOAT DEFAULT 1.0,
    embedding VECTOR(1024),            -- Pre-computed embedding for criteria
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_processed_documents_slug ON processed_documents(slug);
CREATE INDEX IF NOT EXISTS idx_processed_documents_status ON processed_documents(embedding_status);
CREATE INDEX IF NOT EXISTS idx_processed_documents_publication_date ON processed_documents(publication_date);

CREATE INDEX IF NOT EXISTS idx_document_chunks_slug ON document_chunks(document_slug);
CREATE INDEX IF NOT EXISTS idx_document_chunks_slug_index ON document_chunks(document_slug, chunk_index);
CREATE INDEX IF NOT EXISTS idx_document_chunks_similarity ON document_chunks(similarity_score DESC);

CREATE INDEX IF NOT EXISTS idx_impact_criteria_type ON impact_criteria(criteria_type);

-- Vector similarity search indexes (for pgvector)
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_cosine 
    ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_impact_criteria_embedding_cosine 
    ON impact_criteria USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 10);

-- Comments and documentation
COMMENT ON TABLE processed_documents IS 'Federal Register documents with JSON metadata, .txt content, and processing status';
COMMENT ON COLUMN processed_documents.slug IS 'Unique document identifier from Federal Register';
COMMENT ON COLUMN processed_documents.raw_json IS 'Original JSON metadata from Federal Register API';
COMMENT ON COLUMN processed_documents.title IS 'Document title extracted from JSON metadata';
COMMENT ON COLUMN processed_documents.full_text_content IS 'Complete document text content fetched from .txt URL';
COMMENT ON COLUMN processed_documents.raw_text_url IS 'URL to fetch full .txt content';
COMMENT ON COLUMN processed_documents.word_count IS 'Total word count of full_text_content';
COMMENT ON COLUMN processed_documents.embedding_status IS 'Processing status: pending, processing, completed, failed';

COMMENT ON TABLE document_chunks IS '512-token chunks from .txt content of Federal Register documents';
COMMENT ON COLUMN document_chunks.chunk_text IS 'Text content of this chunk from full_text_content (max ~512 tokens)';
COMMENT ON COLUMN document_chunks.start_position IS 'Character start position in full_text_content';
COMMENT ON COLUMN document_chunks.end_position IS 'Character end position in full_text_content';
COMMENT ON COLUMN document_chunks.embedding IS 'BGE-large embedding vector (1024 dimensions)';
COMMENT ON COLUMN document_chunks.similarity_score IS 'Cosine similarity to impact criteria embeddings';

COMMENT ON TABLE impact_criteria IS 'Pre-defined impact criteria with embeddings for similarity scoring';

-- Function to update chunk count and status
CREATE OR REPLACE FUNCTION update_document_chunk_stats(doc_slug VARCHAR(50))
RETURNS VOID AS $$
BEGIN
    UPDATE processed_documents 
    SET 
        chunk_count = (SELECT COUNT(*) FROM document_chunks WHERE document_slug = doc_slug),
        impact_score = (SELECT COALESCE(MAX(similarity_score), 0.0) FROM document_chunks WHERE document_slug = doc_slug),
        embedding_status = CASE 
            WHEN (SELECT COUNT(*) FROM document_chunks WHERE document_slug = doc_slug AND embedding IS NULL) > 0 
            THEN 'processing'
            WHEN (SELECT COUNT(*) FROM document_chunks WHERE document_slug = doc_slug) > 0 
            THEN 'completed'
            ELSE 'pending'
        END
    WHERE slug = doc_slug;
END;
$$ LANGUAGE plpgsql;
