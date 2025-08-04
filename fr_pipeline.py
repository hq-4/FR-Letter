#!/usr/bin/env python3
"""
Federal Register Daily Entry Summarization Pipeline
"""
import os
import json
import logging
import psycopg2
import requests
import feedparser
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fr_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database connection
def get_db_connection():
    """Establish connection to PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
        logger.info("Database connection established")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

# Ollama embedding function
def get_embedding(text):
    """Generate embedding using Ollama bge-large model"""
    try:
        response = requests.post(
            f"{os.getenv('OLLAMA_HOST')}/api/embeddings",
            json={
                "model": os.getenv("OLLAMA_EMBEDDING_MODEL"),
                "prompt": text
            }
        )
        response.raise_for_status()
        embedding = response.json()["embedding"]
        logger.info(f"Generated embedding for text (length: {len(embedding)})")
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise

# DeepSeek summarization function
def generate_summary(text):
    """Generate Politico-style summary using DeepSeek model"""
    try:
        # Load style guide
        with open("style.md", "r") as f:
            style_guide = f.read()
        
        prompt = f"{style_guide}\n\nDocument to summarize:\n{text}"
        
        response = requests.post(
            f"{os.getenv('DEEPSEEK_HOST')}/v1/completions",
            json={
                "model": os.getenv("DEEPSEEK_MODEL"),
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0.7
            }
        )
        response.raise_for_status()
        summary = response.json()["choices"][0]["text"]
        logger.info("Generated summary with DeepSeek model")
        return summary
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        raise

# Parse RSS feed
def parse_rss_feed():
    """Parse Federal Register RSS feed and extract document links"""
    try:
        # For production, we would fetch from the actual URL:
        # feed = feedparser.parse("https://www.federalregister.gov/api/v1/documents.rss")
        
        # For development, we'll parse the local file:
        feed = feedparser.parse("documents.rss")
        
        entries = []
        for item in feed.entries:
            # Extract document number from link
            link_parts = item.link.split("/")
            document_number = link_parts[5]  # e.g., "2025-14789"
            
            entries.append({
                "title": item.title,
                "link": item.link,
                "document_number": document_number,
                "publication_date": item.published,
                "agency": item.get("dc_creator", "")  # From dc:creator tag
            })
        
        logger.info(f"Parsed {len(entries)} entries from RSS feed")
        return entries
    except Exception as e:
        logger.error(f"Failed to parse RSS feed: {e}")
        raise

# Fetch full document JSON
def fetch_document_json(document_number):
    """Fetch full JSON document from Federal Register API"""
    try:
        # For production, we would fetch from the actual URL:
        # url = f"https://www.federalregister.gov/api/v1/documents/{document_number}.json"
        
        # For development, we'll use the local file for the sample document:
        if document_number == "2025-14789":
            with open("2025-14789.json", "r") as f:
                return json.load(f)
        else:
            url = f"https://www.federalregister.gov/api/v1/documents/{document_number}.json"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch document {document_number}: {e}")
        raise

# Calculate impact score
def calculate_impact_score(embedding, criteria_embeddings):
    """Calculate impact score based on cosine similarity to criteria"""
    try:
        # Calculate cosine similarity between document embedding and all criteria embeddings
        similarities = cosine_similarity([embedding], criteria_embeddings)[0]
        
        # Return the maximum similarity as the impact score
        impact_score = float(np.max(similarities))
        
        logger.info(f"Calculated impact score: {impact_score}")
        return impact_score
    except Exception as e:
        logger.error(f"Failed to calculate impact score: {e}")
        raise

# Load impact criteria
def load_impact_criteria():
    """Load impact criteria from criteria.md and generate embeddings"""
    try:
        # Parse criteria.md to extract keywords, agencies, and document types
        with open("criteria.md", "r") as f:
            lines = f.readlines()
        
        criteria = []
        in_keywords = False
        in_agencies = False
        in_document_types = False
        
        for line in lines:
            line = line.strip()
            if line == "## Keywords":
                in_keywords = True
                in_agencies = False
                in_document_types = False
                continue
            elif line == "## Agencies":
                in_keywords = False
                in_agencies = True
                in_document_types = False
                continue
            elif line == "## Document Types":
                in_keywords = False
                in_agencies = False
                in_document_types = True
                continue
            elif line.startswith("## "):
                in_keywords = False
                in_agencies = False
                in_document_types = False
                continue
            
            if line and not line.startswith("#") and not line.startswith("-"):
                if in_keywords:
                    criteria.append({"text": line, "type": "keyword"})
                elif in_agencies:
                    criteria.append({"text": line, "type": "agency"})
                elif in_document_types:
                    criteria.append({"text": line, "type": "document_type"})
        
        # Generate embeddings for all criteria
        criteria_embeddings = []
        for criterion in criteria:
            embedding = get_embedding(criterion["text"])
            criteria_embeddings.append(embedding)
            
            # Store in database
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO impact_criteria (criteria_text, criteria_type) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (criterion["text"], criterion["type"])
            )
            conn.commit()
            cur.close()
            conn.close()
        
        logger.info(f"Loaded {len(criteria)} impact criteria")
        return criteria_embeddings
    except Exception as e:
        logger.error(f"Failed to load impact criteria: {e}")
        raise

# Main pipeline function
def run_pipeline():
    """Run the complete Federal Register summarization pipeline"""
    try:
        logger.info("Starting Federal Register pipeline")
        
        # Load impact criteria
        criteria_embeddings = load_impact_criteria()
        
        # Parse RSS feed
        rss_entries = parse_rss_feed()
        
        # Process each entry
        processed_entries = []
        for entry in rss_entries:
            document_number = entry["document_number"]
            
            # Check if already processed
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM fr_entries WHERE document_number = %s",
                (document_number,)
            )
            result = cur.fetchone()
            cur.close()
            conn.close()
            
            if result:
                logger.info(f"Document {document_number} already processed, skipping")
                continue
            
            # Fetch full document
            doc_json = fetch_document_json(document_number)
            
            # Extract full text content
            # For production, we would fetch the full text from the raw_text_url
            # full_text = requests.get(doc_json.get("raw_text_url")).text
            
            # For development with sample data, we'll use available fields
            full_text = doc_json.get("title", "") + "\n\n" + (doc_json.get("abstract", "") or "")
            
            # Generate embedding
            embedding = get_embedding(full_text)
            
            # Calculate base impact score
            impact_score = calculate_impact_score(embedding, criteria_embeddings)
            
            # Apply agency heuristics
            agency = doc_json.get("agencies", [{}])[0].get("raw_name", "") if doc_json.get("agencies") else ""
            document_type = doc_json.get("type", "")
            
            # Agency weighting
            agency_weights = {
                "Executive Office of the President": 1.5,
                "Homeland Security Department": 1.3,
                "EPA": 1.4
            }
            agency_multiplier = agency_weights.get(agency, 1.0)
            
            # Document type weighting
            type_weights = {
                "Rule": 1.5,
                "Proposed Rule": 1.3,
                "Notice": 1.2,
                "Presidential Document": 1.4
            }
            type_multiplier = type_weights.get(document_type, 1.0)
            
            # Apply multipliers to impact score
            final_impact_score = impact_score * agency_multiplier * type_multiplier
            
            # Store in database
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO fr_entries (document_number, title, agency, document_type, publication_date, full_text, raw_data)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (
                    document_number,
                    doc_json.get("title", ""),
                    agency,
                    document_type,
                    doc_json.get("publication_date", ""),
                    full_text,
                    json.dumps(doc_json)
                )
            )
            entry_id = cur.fetchone()[0]
            
            # Store embedding
            cur.execute(
                "INSERT INTO fr_embeddings (fr_entry_id, embedding) VALUES (%s, %s)",
                (entry_id, embedding)
            )
            conn.commit()
            cur.close()
            conn.close()
            
            processed_entries.append({
                "document_number": document_number,
                "title": entry["title"],
                "full_text": full_text,
                "impact_score": impact_score
            })
            
            logger.info(f"Processed document {document_number}")
        
        # Sort by impact score and select top 5
        processed_entries.sort(key=lambda x: x["impact_score"], reverse=True)
        top_entries = processed_entries[:5]
        
        logger.info(f"Selected top 5 entries by impact score")
        
        # Generate summaries for top entries
        summaries = []
        for entry in top_entries:
            summary = generate_summary(entry["full_text"])
            summaries.append({
                "document_number": entry["document_number"],
                "title": entry["title"],
                "summary": summary,
                "impact_score": entry["impact_score"]
            })
            logger.info(f"Generated summary for document {entry['document_number']}")
        
        # Write summaries to markdown file
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_dir = os.getenv("DAILY_SUMMARY_OUTPUT_DIR", "summaries")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{date_str}.md")
        with open(output_file, "w") as f:
            f.write(f"# Federal Register Highlights - {date_str}\n\n")
            for summary in summaries:
                # Parse the summary to extract headline and bullets
                lines = summary["summary"].split("\n")
                headline = ""
                bullets = []
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("headline: "):
                        headline = line.replace("headline: ", "").strip('"')
                    elif line.startswith("- "):
                        bullets.append(line)
                
                # Write in Politico format
                f.write(f"## {headline}\n")
                f.write(f"**Document:** {summary['title']} ({summary['document_number']})\n")
                f.write(f"**Impact Score:** {summary['impact_score']:.3f}\n\n")
                
                if bullets:
                    f.write("Key Points:\n")
                    for bullet in bullets:
                        f.write(f"{bullet}\n")
                else:
                    # Fallback if summary doesn't follow format
                    f.write(f"{summary['summary']}\n")
                
                f.write("\n---\n\n")
        
        logger.info(f"Summaries written to {output_file}")
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_pipeline()
