import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import re
import pytest
from fr_pipeline import parse_rss_feed

def test_parse_rss_feed():
    """Test parsing of Federal Register RSS feed."""
    documents = parse_rss_feed()
    assert len(documents) > 0, "No documents were parsed"
    first_doc = documents[0]
    assert "slug" in first_doc, "Missing slug in document"
    assert re.fullmatch(r"\d{4}-\d{5}", first_doc["slug"]), "Invalid slug format"
    assert "json_url" in first_doc, "Missing json_url in document"
    assert first_doc["json_url"].startswith("https://www.federalregister.gov/api/v1/documents/"), "Invalid JSON URL format"
    assert first_doc["json_url"].endswith(".json"), "JSON URL should end with .json"
