"""Application configuration."""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings."""

    # API Configuration
    app_name: str = "Pastiche API"
    app_version: str = "1.0.0"
    debug: bool = False

    # CORS Configuration
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",  # Vite default port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # AWS Configuration
    aws_s3_bucket: str = ""
    aws_region: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # FAISS Configuration
    faiss_top_k: int = 50
    procrustes_top_k: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = False

    save_sketch_debug: bool = True


settings = Settings()
