-- Main documents table with UNIQUE constraint on slug for deduplication
CREATE TABLE IF NOT EXISTS processed_documents (
    id SERIAL PRIMARY KEY,
    slug VARCHAR(20) UNIQUE NOT NULL,  -- Enforces deduplication
    publication_date DATE NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_json JSONB NOT NULL,  -- Store full JSON document
    title TEXT,
    agencies TEXT,
    document_type TEXT,
    impact_score FLOAT DEFAULT 0.0
);

-- Document chunks table for 512-token embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_slug VARCHAR(20) NOT NULL REFERENCES processed_documents(slug) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,  -- Order of chunk within document
    chunk_text TEXT NOT NULL,      -- 512-token chunk content
    token_count INTEGER,           -- Actual token count for this chunk
    embedding VECTOR(1024),        -- BGE-large embedding (1024 dimensions)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_slug, chunk_index)  -- Prevent duplicate chunks
);

-- Index for efficient chunk retrieval
CREATE INDEX IF NOT EXISTS idx_document_chunks_slug ON document_chunks(document_slug);
CREATE INDEX IF NOT EXISTS idx_processed_documents_slug ON processed_documents(slug);
