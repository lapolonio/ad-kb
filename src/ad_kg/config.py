"""Configuration module: loads environment variables with typed defaults."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (three levels up from this file)
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_env_path, override=False)

# Neo4j
NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

# NCBI / PubMed
NCBI_API_KEY: str | None = os.getenv("NCBI_API_KEY") or None
NCBI_EMAIL: str = os.getenv("NCBI_EMAIL", "you@example.com")
MAX_PUBMED_PER_QUERY: int = int(os.getenv("MAX_PUBMED_PER_QUERY", "2000"))

# NCBI rate limit: 10/s with key, 3/s without
NCBI_RATE_LIMIT: float = 10.0 if NCBI_API_KEY else 3.0
NCBI_SLEEP: float = 1.0 / NCBI_RATE_LIMIT

# Anthropic
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY") or None

# Pipeline tuning
HDBSCAN_THRESHOLD: float = float(os.getenv("HDBSCAN_THRESHOLD", "0.88"))
RELATION_EXTRACTION_MAX_PAPERS: int = int(
    os.getenv("RELATION_EXTRACTION_MAX_PAPERS", "200")
)
GWAS_P_VALUE_THRESHOLD: float = float(os.getenv("GWAS_P_VALUE_THRESHOLD", "5e-8"))

# Embedding
EMBEDDING_MODEL: str = "allenai/specter2"

# Data directory
DATA_DIR: Path = Path(__file__).resolve().parents[3] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
