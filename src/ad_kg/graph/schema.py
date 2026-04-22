"""Neo4j schema: constraints and indexes for the AD Knowledge Graph."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_CYPHER: list[str] = [
    # ── Node uniqueness constraints ──────────────────────────────────────────
    "CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (n:Drug) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT gene_id IF NOT EXISTS FOR (n:Gene) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (n:Disease) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT pathway_id IF NOT EXISTS FOR (n:Pathway) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (n:Paper) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT trial_id IF NOT EXISTS FOR (n:Trial) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT snp_id IF NOT EXISTS FOR (n:SNP) REQUIRE n.id IS UNIQUE",
    # ── Supporting indexes ────────────────────────────────────────────────────
    "CREATE INDEX drug_name IF NOT EXISTS FOR (n:Drug) ON (n.name)",
    "CREATE INDEX gene_symbol IF NOT EXISTS FOR (n:Gene) ON (n.symbol)",
    "CREATE INDEX disease_name IF NOT EXISTS FOR (n:Disease) ON (n.name)",
    "CREATE INDEX paper_pmid IF NOT EXISTS FOR (n:Paper) ON (n.pmid)",
    "CREATE INDEX trial_status IF NOT EXISTS FOR (n:Trial) ON (n.status)",
    "CREATE INDEX snp_rsid IF NOT EXISTS FOR (n:SNP) ON (n.rsid)",
    # ── Full-text search index on Paper abstracts ─────────────────────────────
    (
        "CREATE FULLTEXT INDEX paper_abstract_ft IF NOT EXISTS "
        "FOR (n:Paper) ON EACH [n.abstract, n.title]"
    ),
]


def apply_schema(driver: Any) -> None:
    """Execute all schema Cypher statements against the Neo4j database.

    Args:
        driver: neo4j.Driver instance.
    """
    with driver.session() as session:
        for cypher in SCHEMA_CYPHER:
            try:
                session.run(cypher)
                logger.debug("Applied: %s", cypher[:80])
            except Exception as exc:
                logger.warning("Schema statement failed (may already exist): %s", exc)
    logger.info("Schema applied (%d statements).", len(SCHEMA_CYPHER))
