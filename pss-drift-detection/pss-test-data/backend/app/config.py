"""Application configuration from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    domain_id: str = "healthcare"
    backend_port: int = 8000
    frontend_port: int = 3000

    model_config = {"env_file": "../.env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
