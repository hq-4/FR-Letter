-- Setup script for Postgres with pgvector extension
-- This script creates the necessary tables for the FR-Letter pipeline

-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table for Federal Register entries
CREATE TABLE IF NOT EXISTS fr_entries (
    id SERIAL PRIMARY KEY,
    document_number VARCHAR(255) UNIQUE NOT NULL,
    title TEXT,
    agency TEXT,
    document_type TEXT,
    publication_date DATE,
    full_text TEXT,
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for embeddings
CREATE TABLE IF NOT EXISTS fr_embeddings (
    id SERIAL PRIMARY KEY,
    fr_entry_id INTEGER REFERENCES fr_entries(id) ON DELETE CASCADE,
    embedding VECTOR(1024), -- bge-large has 1024 dimensions
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for impact criteria
CREATE TABLE IF NOT EXISTS impact_criteria (
    id SERIAL PRIMARY KEY,
    criteria_text TEXT NOT NULL,
    criteria_type VARCHAR(50), -- keyword, agency, document_type
    weight FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_fr_entries_document_number ON fr_entries(document_number);
CREATE INDEX IF NOT EXISTS idx_fr_entries_publication_date ON fr_entries(publication_date);
CREATE INDEX IF NOT EXISTS idx_fr_embeddings_fr_entry_id ON fr_embeddings(fr_entry_id);
