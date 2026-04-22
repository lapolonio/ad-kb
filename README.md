# Alzheimer's Drug Repurposing Knowledge Graph

A Python pipeline that builds a Neo4j knowledge graph from biomedical data sources
to identify drug repurposing candidates for Alzheimer's disease — with a focus on
GLP-1 agonists and metabolic pathway drugs.

## Quick Start

### 1. Clone and set up environment

```bash
cd ad_kg
cp .env.example .env
# Edit .env with your API keys and Neo4j credentials
uv sync --dev --extra nlp --extra ml
```

### 2. Start Neo4j (local) or point to AuraDB

For local development:
```bash
docker compose up -d
```

For AuraDB, set `NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io` in `.env` — no Docker needed.

### 3. Run the pipeline

```bash
# Fetch all data sources
uv run python -m ad_kg ingest

# NER + relation extraction via Claude
uv run python -m ad_kg extract

# Embed and canonicalize entities
uv run python -m ad_kg resolve

# Load everything into Neo4j
uv run python -m ad_kg load

# Run queries
uv run python -m ad_kg query --name whitespace_opportunity
```

### 4. Run tests

```bash
uv run pytest tests/ -v
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
| `faers_sensitivity_cohorts` | ROR across all sub-populations (overall, T2DM-only, elderly, post-2020) for drugs with a baseline protective signal |
| `faers_cross_reaction_consistency` | Count of AD MedDRA reaction terms showing ROR < 1 per drug — high count = signal consistent across phenotypes |
| `faers_subpopulation_comparison` | Side-by-side overall vs T2DM vs elderly vs post-2020 ROR — robustness check against confounding |

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

## Live Instance

A public read-only instance of the knowledge graph is running on Neo4j AuraDB.
To request access for querying or exploration, open a [GitHub issue](https://github.com/leo-explainml/ad-kg/issues) or reach out by email (see below).

The graph is periodically refreshed as new PubMed, GWAS, and FAERS data becomes available.

## Collaboration

This project is maintained by Leonardo Apolonio. If you are a researcher, clinician, or engineer working on Alzheimer's disease, drug repurposing, or biomedical knowledge graphs and want to collaborate — whether to extend the pipeline, add new data sources, or use the graph in your own work — get in touch.

- **GitHub Issues:** feature requests, bug reports, data questions
- **Email:** apolonio.leonardo@gmail.com
- **LinkedIn:** https://www.linkedin.com/in/leonardo-apolonio/
