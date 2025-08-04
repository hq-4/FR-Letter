# FR-Letter: Federal Register Daily Summarization Pipeline

Automate daily summarization of Federal Register entries, ranking by impact, and generate markdown summaries in Politico-style prose.

## System Overview

- **Self-hosted:** All processing and storage on-premises
- **Data sources:** 
  - Fetch entries from https://www.federalregister.gov/api/v1/documents.rss
  - Parse each entry; use unique slug to fetch full JSON
  - Parse all entries for training, daily new entries for production
- **Database:**
  - Postgres for entry tracking and archival
  - pgvector extension for embedding storage
  - `.env` for all DB credentials and config
- **Embeddings:**
  - Ollama with bge-large for all text embeddings
- **Ranking:**
  - Impact criteria loaded from `criteria.md`
  - Cosine similarity between entry and criteria embeddings
  - Agency/document-type heuristics (e.g., EPA prioritized for environmental)
- **Summarization:**
  - DeepSeek 1.5B for summary generation
  - System prompt loads `style.md` (Politico-style)
- **Output:**
  - Top 5 impactful entries per day
  - Single daily markdown file: `summaries/YYYY-MM-DD.md`

## Setup

1. **PostgreSQL with pgvector:**
   ```bash
   # Install pgvector extension (if not already installed)
   # Follow instructions at https://github.com/pgvector/pgvector#installation
   ```

2. **Ollama with bge-large:**
   ```bash
   # Install Ollama from https://ollama.com/download
   ollama pull bge-large
   ```

3. **DeepSeek model:**
   ```bash
   # Pull DeepSeek 1.5B model
   ollama pull deepseek-coder:1.5b
   ```

4. **Environment variables:**
   ```bash
   # Copy .env.example to .env and fill in your values
   cp .env.example .env
   ```

5. **Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the pipeline:
```bash
python fr_pipeline.py
```

## Project Structure

```
.
├── documents.rss              # Sample RSS feed
├── 2025-14789.json            # Sample document JSON
├── criteria.md                # Impact criteria keywords
├── style.md                   # Politico-style summary guide
├── setup_pgvector.sql         # Database setup script
├── .env.example               # Environment variables template
├── requirements.txt           # Python dependencies
├── fr_pipeline.py             # Main pipeline script
└── summaries/                 # Daily summary output directory
    └── YYYY-MM-DD.md          # Daily top 5 summaries
```

## Configuration Files

### criteria.md
Contains keywords, agencies, and document types that define "impactful" entries.

### style.md
Defines the Politico-style writing guide for summaries.

### .env
Configure database connections and model endpoints.

## Pipeline Steps

1. Parse Federal Register RSS feed
2. Fetch full JSON documents for each entry
3. Generate embeddings using Ollama bge-large
4. Calculate impact scores using cosine similarity
5. Apply agency/document-type heuristics
6. Select top 5 entries
7. Generate Politico-style summaries with DeepSeek
8. Write to daily markdown file

## Logging

Detailed logs are written to `fr_pipeline.log` and stdout for debugging.
