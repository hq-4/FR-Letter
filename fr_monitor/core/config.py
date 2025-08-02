"""
Configuration management for the Federal Register monitoring system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    # Federal Register API
    federal_register_api_key: Optional[str] = Field(None, alias="FEDERAL_REGISTER_API_KEY")
    
    # OpenRouter API
    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field("anthropic/claude-3-haiku", alias="OPENROUTER_MODEL")
    
    # Redis Configuration
    redis_host: str = Field("localhost", alias="REDIS_HOST")
    redis_port: int = Field(6379, alias="REDIS_PORT")
    redis_password: Optional[str] = Field(None, alias="REDIS_PASSWORD")
    redis_db: int = Field(0, alias="REDIS_DB")
    
    # Ollama Configuration
    ollama_host: str = Field("http://localhost:11434", alias="OLLAMA_HOST")
    embedding_model: str = Field("qwen2:1.5b", alias="EMBEDDING_MODEL")
    summary_model: str = Field("mistral:latest", alias="SUMMARY_MODEL")
    
    # Publishing APIs
    substack_api_key: Optional[str] = Field(None, alias="SUBSTACK_API_KEY")
    substack_publication_id: Optional[str] = Field(None, alias="SUBSTACK_PUBLICATION_ID")
    telegram_bot_token: Optional[str] = Field(None, alias="TELEGRAM_BOT_TOKEN")
    telegram_channel_id: Optional[str] = Field(None, alias="TELEGRAM_CHANNEL_ID")
    
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
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/scoring.yaml")
        self._weights = self._load_default_weights()
    
    def _load_default_weights(self) -> Dict[str, float]:
        """Load default scoring weights."""
        return {
            "agency_importance": {
                "EPA": 0.9,
                "FDA": 0.8,
                "SEC": 0.8,
                "FTC": 0.7,
                "DOT": 0.7,
                "HHS": 0.6,
                "default": 0.5
            },
            "document_length_weight": 0.3,
            "final_rule_bonus": 0.4,
            "executive_order_bonus": 0.8,
            "major_rule_bonus": 0.6
        }
    
    @property
    def weights(self) -> Dict[str, Any]:
        """Get scoring weights."""
        return self._weights


# Global settings instance
settings = Settings()
scoring_config = ScoringConfig()
