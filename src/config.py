"""Configuration management for Orion agent."""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional


class Config:
    """Configuration loader for Orion agent."""
    
    def __init__(self):
        # Load environment variables
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()
        
        # Google Cloud & BigQuery
        self.google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.google_application_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Gemini API
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_AI_KEY")
        
        # BigQuery settings
        self.bigquery_dataset = os.getenv(
            "BIGQUERY_DATASET", 
            "bigquery-public-data.thelook_ecommerce"
        )
        self.max_query_rows = int(os.getenv("MAX_QUERY_ROWS", "10000"))
        self.query_timeout = int(os.getenv("QUERY_TIMEOUT", "300"))
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of missing fields."""
        missing = []
        
        if not self.google_cloud_project:
            missing.append("GOOGLE_CLOUD_PROJECT")
        
        if not self.gemini_api_key:
            missing.append("GEMINI_API_KEY or GEMINI_AI_KEY")
        
        return missing

# Global config instance
config = Config()

