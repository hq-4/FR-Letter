"""
Configuration management for the Federal Register monitoring system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        protected_namespaces=(),
        extra='ignore'  # Ignore extra environment variables
    )
    
    # Federal Register API
    federal_register_api_key: Optional[str] = Field(None, alias="FEDERAL_REGISTER_API_KEY")
    
    # OpenRouter API
    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field("anthropic/claude-3-haiku", alias="OPENROUTER_MODEL")
    
    # Redis Configuration
    redis_host: str = Field("localhost", alias="REDIS_HOST")
    redis_port: int = Field(6380, alias="REDIS_PORT")  # Redis Stack container port
    redis_password: Optional[str] = Field(None, alias="REDIS_PASSWORD")
    redis_db: int = Field(0, alias="REDIS_DB")
    
    # Ollama Configuration
    ollama_host: str = Field("http://localhost:11434", alias="OLLAMA_HOST")
    embedding_model: str = Field("qwen3:1.7b", alias="EMBEDDING_MODEL")
    summary_model: str = Field("mistral:latest", alias="SUMMARY_MODEL")
    
    # Publishing APIs
    # Publishing configuration removed - using markdown file output
    
    # Pipeline Configuration
    max_daily_openrouter_calls: int = Field(5, alias="MAX_DAILY_OPENROUTER_CALLS")
    pipeline_timeout_minutes: int = Field(5, alias="PIPELINE_TIMEOUT_MINUTES")
    impact_score_threshold: float = Field(0.7, alias="IMPACT_SCORE_THRESHOLD")
    similarity_threshold: float = Field(0.8, alias="SIMILARITY_THRESHOLD")
    
    # Monitoring
    sentry_dsn: Optional[str] = Field(None, alias="SENTRY_DSN")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent
    
    @property
    def config_dir(self) -> Path:
        """Get the configuration directory."""
        return self.project_root / "config"
    
    @property
    def logs_dir(self) -> Path:
        """Get the logs directory."""
        return self.project_root / "logs"


class ScoringConfig:
    """Configuration for impact scoring weights."""
    
    def __init__(self):
        # Load from environment variables with defaults
        self.agency_epa = float(os.getenv('SCORING_AGENCY_EPA', 1.5))
        self.agency_dot = float(os.getenv('SCORING_AGENCY_DOT', 1.4))
        self.agency_hhs = float(os.getenv('SCORING_AGENCY_HHS', 1.4))
        self.agency_dol = float(os.getenv('SCORING_AGENCY_DOL', 1.3))
        self.agency_treasury = float(os.getenv('SCORING_AGENCY_TREASURY', 1.3))
        self.agency_other = float(os.getenv('SCORING_AGENCY_OTHER', 1.0))
        
        # Document characteristics
        self.document_length = float(os.getenv('SCORING_DOCUMENT_LENGTH', 0.01))
        self.is_final_rule = float(os.getenv('SCORING_IS_FINAL_RULE', 0.5))
        self.has_economic_impact = float(os.getenv('SCORING_HAS_ECONOMIC_IMPACT', 0.7))
        self.has_public_comments = float(os.getenv('SCORING_HAS_PUBLIC_COMMENTS', 0.3))
        
        # Document type modifiers
        self.rule = float(os.getenv('SCORING_RULE', 1.0))
        self.notice = float(os.getenv('SCORING_NOTICE', 0.7))
        self.proposed_rule = float(os.getenv('SCORING_PROPOSED_RULE', 0.8))
        self.presidential_document = float(os.getenv('SCORING_PRESIDENTIAL_DOCUMENT', 1.5))
    
    @property
    def weights(self) -> Dict[str, float]:
        """Get all weights as a dictionary."""
        return {
            'agency_epa': self.agency_epa,
            'agency_dot': self.agency_dot,
            'agency_hhs': self.agency_hhs,
            'agency_dol': self.agency_dol,
            'agency_treasury': self.agency_treasury,
            'agency_other': self.agency_other,
            'document_length': self.document_length,
            'is_final_rule': self.is_final_rule,
            'has_economic_impact': self.has_economic_impact,
            'has_public_comments': self.has_public_comments,
            'rule': self.rule,
            'notice': self.notice,
            'proposed_rule': self.proposed_rule,
            'presidential_document': self.presidential_document
        }


# Global settings instance
settings = Settings()
scoring_config = ScoringConfig()

# Ensure required directories exist
os.makedirs(settings.logs_dir, exist_ok=True)
os.makedirs(settings.config_dir, exist_ok=True)
