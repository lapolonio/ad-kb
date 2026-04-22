# AD Knowledge Graph — Claude Code Instructions

## Project goal
Find drug repurposing candidates for Alzheimer's disease by building a
knowledge graph over PubMed, ClinicalTrials.gov, GWAS Catalog, and FDA FAERS.
Focus: GLP-1 agonists and metabolic pathway drugs as AD candidates.

## Architecture
5 stages: ingest → NER extract → entity resolve → Neo4j load → Cypher query
Source files: src/ad_kg/ package

## Key constraints
- NEVER hardcode credentials. Use .env + python-dotenv.
- ALWAYS add retry + exponential backoff on every API call.
- ALWAYS deduplicate by stable ID before inserting to Neo4j.
- Rate limits: NCBI 3 req/s (no key), 10 req/s (with key). Sleep accordingly.
- HDBSCAN cosine threshold = 0.88. Do not change without running eval first.
- scispaCy model: en_core_sci_lg (not sm). UMLS linker required.
- Embedding model: allenai/specter2 for biomedical domain accuracy.

## Setup
  uv sync --dev                   # core + dev deps
  uv sync --dev --extra nlp       # + scispacy (requires macOS ARM or Linux)
  uv sync --dev --extra ml        # + sentence-transformers (requires py<=3.12 on Intel Mac)

## Run commands
  uv run python -m ad_kg ingest          # fetch all four sources
  uv run python -m ad_kg extract         # NER + relation extraction
  uv run python -m ad_kg resolve         # embed + canonicalize
  uv run python -m ad_kg load            # write to Neo4j
  uv run python -m ad_kg query           # run named Cypher queries
  uv run python -m ad_kg query --name whitespace_opportunity

## Neo4j setup (Docker)
  docker compose up -d

## Tests
  uv run pytest tests/ -v
