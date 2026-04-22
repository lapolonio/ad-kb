# Alzheimer's Drug Repurposing Knowledge Graph

A Python pipeline that builds a Neo4j knowledge graph from biomedical data sources
to identify drug repurposing candidates for Alzheimer's disease — with a focus on
GLP-1 agonists and metabolic pathway drugs.

## Quick Start

### 1. Clone and set up environment

```bash
cd ad_kg
cp .env.example .env
# Edit .env with your API keys
pip install -e ".[dev]"
```

### 2. Start Neo4j

```bash
docker compose up -d
```

### 3. Run the pipeline

```bash
# Fetch all data sources (add --limit 50 for a quick dev run)
python -m ad_kg ingest

# NER + relation extraction via Claude
python -m ad_kg extract

# Embed and canonicalize entities
python -m ad_kg resolve

# Load everything into Neo4j
python -m ad_kg load

# Run queries
python -m ad_kg query --name whitespace_opportunity
```

### 4. Run tests

```bash
pytest tests/ -v
```

## Architecture

```
PubMed ──┐
GWAS ────┤→ ingest/ → extract/ → resolve/ → graph/ → Cypher queries
FAERS ───┤   (data)   (NER/LLM)  (embed)   (Neo4j)
ClinTrials┘
```

**5 stages:**
1. **ingest** — Fetch papers (PubMed), trials (ClinicalTrials.gov), genetic hits (GWAS Catalog), pharmacovigilance signals (FDA FAERS)
2. **extract** — Named entity recognition (scispaCy en_core_sci_lg + UMLS) + LLM relation extraction (Claude claude-sonnet-4-6 with prompt caching)
3. **resolve** — Embed entities (allenai/specter2) and cluster synonyms (HDBSCAN, cosine threshold 0.88)
4. **load** — MERGE nodes and edges idempotently into Neo4j 5 with APOC
5. **query** — 11 named Cypher queries for drug repurposing analysis

## Named Queries

| Query | Purpose |
|-------|---------|
| `whitespace_opportunity` | Drugs with FAERS protective signal + GWAS gene support but **no active AD trial** — top repurposing targets |
| `triple_convergence` | Drugs with all three signals: FAERS protective + GWAS + literature |
| `bridge_genes_ranked` | Genes ranked by combined AD × metabolic GWAS p-value product |
| `repurposing_candidates` | Drugs near GLP1R and insulin pathway genes |
| `genetic_overlap` | Genes with GWAS hits in **both** T2D and AD |
| `pathway_bridges` | Pathways co-mentioned in metabolic AND AD literature |
| `trial_gaps` | Drugs with ≥5 AD literature mentions but no active trial |
| `gwas_snp_to_drug` | SNP → gene → approved drug paths |
| `open_trials_bridge_genes` | Active trials testing drugs that hit bridge genes |
| `protective_drugs_ranked` | All FAERS protective drugs ranked by ROR + literature support |
| `semaglutide_neighbors` | Everything connected to semaglutide within 2 hops |

## Data Sources

| Source | What | API |
|--------|------|-----|
| PubMed | Biomedical abstracts | NCBI E-utilities via pymed |
| ClinicalTrials.gov | Trial metadata | ClinicalTrials API v2 |
| GWAS Catalog | Genetic associations | EBI REST API |
| FDA FAERS | Adverse event reports | OpenFDA API |
| ChEMBL | Drug-target links | EBI ChEMBL REST API |

## Key Design Decisions

- **No hardcoded credentials** — all secrets via `.env` + python-dotenv
- **Retry + exponential backoff** on every external API call (tenacity)
- **Rate limiting** — 3 req/s (NCBI no key) or 10 req/s (with key)
- **Deduplication** by stable ID (PMID, NCT ID, rsID) before Neo4j insert
- **Prompt caching** on all Claude API calls (ephemeral cache on system prompt)
- **HDBSCAN threshold 0.88** — conservative to avoid false merges

## Environment Variables

See `.env.example` for all required variables. The most important:

```
ANTHROPIC_API_KEY=sk-ant-...     # Required for relation extraction
NCBI_API_KEY=...                 # Optional, increases rate limit 3x
NEO4J_PASSWORD=...               # Match docker-compose.yml
```
