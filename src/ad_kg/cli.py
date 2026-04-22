"""Command-line interface for the AD Knowledge Graph pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ad_kg.cli")


# ── Sub-command handlers ──────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    """Run all four ingest functions and save results to data/."""
    from ad_kg.config import DATA_DIR  # noqa: PLC0415
    from ad_kg.ingest.pubmed import fetch_pubmed  # noqa: PLC0415
    from ad_kg.ingest.clinical_trials import fetch_clinical_trials  # noqa: PLC0415
    from ad_kg.ingest.gwas import ingest_gwas  # noqa: PLC0415
    from ad_kg.ingest.faers import ingest_faers  # noqa: PLC0415

    limit = args.limit

    # PubMed
    logger.info("=== PubMed ingest ===")
    queries = [
        "Alzheimer's disease GLP-1 agonist",
        "Alzheimer's disease semaglutide",
        "Alzheimer's disease metformin",
        "Alzheimer's disease SGLT2 inhibitor",
        "Alzheimer's disease type 2 diabetes drug repurposing",
    ]
    all_papers = {}
    for q in queries:
        papers = fetch_pubmed(q, limit=limit)
        for p in papers:
            all_papers[p.pmid] = p
    papers_list = list(all_papers.values())
    logger.info("Total unique papers: %d", len(papers_list))
    papers_path = DATA_DIR / "papers.json"
    papers_path.write_text(
        json.dumps([p.to_dict() for p in papers_list], indent=2)
    )
    logger.info("Saved papers to %s", papers_path)

    # Clinical Trials
    logger.info("=== Clinical Trials ingest ===")
    trials = fetch_clinical_trials(limit=limit)
    trials_path = DATA_DIR / "trials.json"
    trials_path.write_text(
        json.dumps([t.to_dict() for t in trials], indent=2)
    )
    logger.info("Saved %d trials to %s", len(trials), trials_path)

    # GWAS
    logger.info("=== GWAS ingest ===")
    gwas_hits = ingest_gwas(limit=limit)
    gwas_path = DATA_DIR / "gwas.json"
    gwas_path.write_text(
        json.dumps([g.to_dict() for g in gwas_hits], indent=2)
    )
    logger.info("Saved %d GWAS hits to %s", len(gwas_hits), gwas_path)

    # FAERS
    logger.info("=== FAERS ingest ===")
    faers_reports = ingest_faers()
    faers_path = DATA_DIR / "faers.json"
    faers_path.write_text(
        json.dumps([f.to_dict() for f in faers_reports], indent=2)
    )
    logger.info("Saved %d FAERS reports to %s", len(faers_reports), faers_path)

    logger.info("Ingest complete.")


def cmd_extract(args: argparse.Namespace) -> None:
    """Load ingested data, run NER + relation extraction, save to data/."""
    from ad_kg.config import DATA_DIR  # noqa: PLC0415
    from ad_kg.models import Paper  # noqa: PLC0415
    from ad_kg.extract.ner import extract_entities  # noqa: PLC0415
    from ad_kg.extract.relations import extract_relations_llm  # noqa: PLC0415

    papers_path = DATA_DIR / "papers.json"
    if not papers_path.exists():
        logger.error("papers.json not found. Run 'ingest' first.")
        sys.exit(1)

    papers_raw = json.loads(papers_path.read_text())
    papers = [Paper.from_dict(d) for d in papers_raw]
    logger.info("Loaded %d papers", len(papers))

    # NER
    logger.info("=== NER extraction ===")
    mentions = extract_entities(papers)
    entities_path = DATA_DIR / "entities.json"
    entities_path.write_text(
        json.dumps([m.to_dict() for m in mentions], indent=2)
    )
    logger.info("Saved %d mentions to %s", len(mentions), entities_path)

    # Relation extraction
    logger.info("=== LLM relation extraction ===")
    relations = extract_relations_llm(papers)
    relations_path = DATA_DIR / "relations.json"
    relations_path.write_text(json.dumps(relations, indent=2))
    logger.info("Saved %d relations to %s", len(relations), relations_path)

    logger.info("Extract complete.")


def cmd_resolve(args: argparse.Namespace) -> None:
    """Load entity mentions, embed, canonicalize, and save."""
    from ad_kg.config import DATA_DIR  # noqa: PLC0415
    from ad_kg.models import EntityMention  # noqa: PLC0415
    from ad_kg.resolve.embed import embed_mentions  # noqa: PLC0415
    from ad_kg.resolve.canonicalize import cluster_and_canonicalize  # noqa: PLC0415

    entities_path = DATA_DIR / "entities.json"
    if not entities_path.exists():
        logger.error("entities.json not found. Run 'extract' first.")
        sys.exit(1)

    ent_raw = json.loads(entities_path.read_text())
    mentions = [EntityMention.from_dict(d) for d in ent_raw]
    logger.info("Loaded %d mentions", len(mentions))

    logger.info("=== Embedding ===")
    embeddings = embed_mentions(mentions)

    logger.info("=== Clustering / canonicalization ===")
    resolved = cluster_and_canonicalize(mentions, embeddings)

    resolved_path = DATA_DIR / "entities_resolved.json"
    resolved_path.write_text(
        json.dumps([m.to_dict() for m in resolved], indent=2)
    )
    logger.info("Saved %d resolved mentions to %s", len(resolved), resolved_path)
    logger.info("Resolve complete.")


def cmd_load(args: argparse.Namespace) -> None:
    """Connect to Neo4j, apply schema, and load all data."""
    from neo4j import GraphDatabase  # noqa: PLC0415
    from ad_kg.config import DATA_DIR, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD  # noqa: PLC0415
    from ad_kg.models import Paper, Trial, EntityMention, GWASHit, FAERSReport  # noqa: PLC0415
    from ad_kg.graph.schema import apply_schema  # noqa: PLC0415
    from ad_kg.graph.loader import load_graph, load_gwas, load_faers, load_trials  # noqa: PLC0415

    logger.info("Connecting to Neo4j at %s", NEO4J_URI)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    logger.info("Applying schema...")
    apply_schema(driver)

    # Load papers + entities + relations
    papers_path = DATA_DIR / "papers.json"
    resolved_path = DATA_DIR / "entities_resolved.json"
    fallback_entities_path = DATA_DIR / "entities.json"
    relations_path = DATA_DIR / "relations.json"

    papers: list[Paper] = []
    if papers_path.exists():
        papers = [Paper.from_dict(d) for d in json.loads(papers_path.read_text())]

    mentions: list[EntityMention] = []
    ent_path = resolved_path if resolved_path.exists() else fallback_entities_path
    if ent_path.exists():
        mentions = [EntityMention.from_dict(d) for d in json.loads(ent_path.read_text())]

    relations: list[dict] = []
    if relations_path.exists():
        relations = json.loads(relations_path.read_text())

    load_graph(driver, papers, mentions, relations)

    # Load trials
    trials_path = DATA_DIR / "trials.json"
    if trials_path.exists():
        trials = [Trial.from_dict(d) for d in json.loads(trials_path.read_text())]
        load_trials(driver, trials)

    # Load GWAS
    gwas_path = DATA_DIR / "gwas.json"
    if gwas_path.exists():
        gwas_hits = [GWASHit.from_dict(d) for d in json.loads(gwas_path.read_text())]
        load_gwas(driver, gwas_hits)

    # Load FAERS
    faers_path = DATA_DIR / "faers.json"
    if faers_path.exists():
        faers = [FAERSReport.from_dict(d) for d in json.loads(faers_path.read_text())]
        load_faers(driver, faers)

    driver.close()
    logger.info("Load complete.")


def cmd_query(args: argparse.Namespace) -> None:
    """Run a named Cypher query and print results."""
    from neo4j import GraphDatabase  # noqa: PLC0415
    from ad_kg.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD  # noqa: PLC0415
    from ad_kg.graph.queries import QUERIES, run_query  # noqa: PLC0415

    name = args.name
    if name not in QUERIES:
        logger.error(
            "Unknown query: %r. Available queries: %s",
            name,
            ", ".join(QUERIES.keys()),
        )
        sys.exit(1)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    rows = run_query(driver, name, limit=args.limit)
    driver.close()

    if not rows:
        print("No results.")
        return

    # Pretty print as JSON
    print(json.dumps(rows, indent=2, default=str))
    print(f"\n({len(rows)} rows)")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ad_kg",
        description="Alzheimer's Drug Repurposing Knowledge Graph Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest_p = subparsers.add_parser("ingest", help="Fetch all data sources")
    ingest_p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max items per source (for dev runs)",
    )
    ingest_p.set_defaults(func=cmd_ingest)

    # extract
    extract_p = subparsers.add_parser("extract", help="NER + relation extraction")
    extract_p.set_defaults(func=cmd_extract)

    # resolve
    resolve_p = subparsers.add_parser("resolve", help="Embed + canonicalize entities")
    resolve_p.set_defaults(func=cmd_resolve)

    # load
    load_p = subparsers.add_parser("load", help="Load data into Neo4j")
    load_p.set_defaults(func=cmd_load)

    # query
    query_p = subparsers.add_parser("query", help="Run a named Cypher query")
    from ad_kg.graph.queries import QUERIES  # noqa: PLC0415
    query_p.add_argument(
        "--name",
        default="whitespace_opportunity",
        choices=list(QUERIES.keys()),
        help="Query name (default: whitespace_opportunity)",
    )
    query_p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="LIMIT override for dev runs",
    )
    query_p.set_defaults(func=cmd_query)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
