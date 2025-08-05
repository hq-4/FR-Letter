#!/usr/bin/env python3
"""
Simple document ingestion: Use existing working parser, focus on JSON fetch + DB storage
"""
import os
import json
import logging
import psycopg2
import requests
import time
from datetime import datetime
from dotenv import load_dotenv

# Import the working parser from fr_pipeline
from fr_pipeline import parse_rss_feed, get_db_connection, fetch_document_json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simple_ingest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def is_document_exists(slug):
    """Check if document already exists in database"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM processed_documents WHERE slug = %s", (slug,))
        return cursor.fetchone() is not None
    finally:
        conn.close()

def store_document(slug, document_json):
    """Store document JSON in database with deduplication"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Extract key fields
        title = document_json.get("title", "")
        agencies = ", ".join([ag.get("name", "Unnamed Agency") for ag in document_json.get("agencies", [])])
        document_type = document_json.get("type", "")
        publication_date = document_json.get("publication_date")
        
        # Convert publication_date to proper format if needed
        if publication_date:
            try:
                # Assume format is YYYY-MM-DD
                pub_date = datetime.strptime(publication_date, "%Y-%m-%d").date()
            except:
                pub_date = None
        else:
            pub_date = None
        
        cursor.execute(
            """
            INSERT INTO processed_documents (slug, publication_date, raw_json, title, agencies, document_type)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (slug) DO NOTHING
            """,
            (slug, pub_date, json.dumps(document_json), title, agencies, document_type)
        )
        
        if cursor.rowcount > 0:
            logger.info(f"✓ Stored document {slug}")
            return True
        else:
            logger.debug(f"⚠ Document {slug} already exists, skipped")
            return False
        
        conn.commit()
        
    except Exception as e:
        logger.error(f"Failed to store document {slug}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def main():
    """Main ingestion process"""
    logger.info("Starting document ingestion using existing working parser")
    
    try:
        # Use the existing working parser
        documents = parse_rss_feed()
        logger.info(f"Found {len(documents)} documents to process")
        
        # Process each document
        success_count = 0
        error_count = 0
        skip_count = 0
        
        for i, doc in enumerate(documents, 1):
            slug = doc["slug"]
            json_url = doc["json_url"]
            
            logger.info(f"[{i}/{len(documents)}] Processing {slug}")
            
            # Check if already exists
            if is_document_exists(slug):
                logger.debug(f"Document {slug} already exists, skipping")
                skip_count += 1
                continue
            
            # Fetch JSON using existing function
            document_json = fetch_document_json(json_url)
            if document_json is None:
                error_count += 1
                continue
            
            # Store in database
            if store_document(slug, document_json):
                success_count += 1
            
            # Rate limiting
            time.sleep(0.1)  # Be nice to the API
        
        logger.info(f"Ingestion completed: {success_count} new, {skip_count} skipped, {error_count} errors")
        
        # Final count check
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM processed_documents")
        total_count = cursor.fetchone()[0]
        conn.close()
        
        logger.info(f"Total documents in database: {total_count}")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()
