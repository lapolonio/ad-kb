"""Named Cypher queries for the AD Knowledge Graph."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

QUERIES: dict[str, str] = {
    # 1. Drugs with FAERS protective signal + bridge gene (AD or metabolic GWAS)
    #    but NO active AD trial — i.e. repurposing whitespace
    "whitespace_opportunity": """
MATCH (d:Drug)-[:PROTECTIVE_SIGNAL]->(f:FAERSReport)
WHERE f.ror < 1.0
WITH d
MATCH (d)-[:TARGETS]->(g:Gene)<-[:LINKED_TO]-(s:SNP)-[:ASSOCIATED_WITH]->(dis:Disease)
WHERE dis.name CONTAINS "Alzheimer"
   OR dis.name IN ["type 2 diabetes", "insulin resistance", "body mass index",
                   "glycated hemoglobin", "fasting glucose", "obesity"]
WITH DISTINCT d, g, dis
WHERE NOT EXISTS {
  MATCH (d)<-[:TESTS]-(t:Trial)-[:FOR]->(dis2:Disease)
  WHERE dis2.name CONTAINS "Alzheimer"
    AND t.status IN ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING"]
}
RETURN d.name AS drug, d.id AS drug_id,
       collect(DISTINCT g.symbol) AS bridge_genes,
       collect(DISTINCT dis.name) AS gwas_traits
ORDER BY d.name
""".strip(),

    # 2. Drugs with all three signals: FAERS protective + GWAS gene + literature
    "triple_convergence": """
MATCH (d:Drug)-[:PROTECTIVE_SIGNAL]->(f:FAERSReport)
WHERE f.ror < 1.0
WITH d

MATCH (d)-[:TARGETS|RELATED_TO*1..2]->(g:Gene)<-[:ASSOCIATED_WITH|LINKED_TO*1..2]-(s:SNP)
   -[:ASSOCIATED_WITH]->(dis:Disease)
WHERE dis.name CONTAINS "Alzheimer"
WITH d, count(DISTINCT s) AS gwas_snps

MATCH (d)-[:MENTIONS|RELATED_TO*1..2]-(p:Paper)
WITH d, gwas_snps, count(DISTINCT p) AS lit_count
WHERE lit_count >= 1

RETURN d.name AS drug,
       gwas_snps AS gwas_support,
       lit_count AS literature_count
ORDER BY gwas_snps DESC, lit_count DESC
""".strip(),

    # 3. Genes ranked by product of AD × metabolic GWAS association p-values
    "bridge_genes_ranked": """
MATCH (g:Gene)<-[:LINKED_TO]-(s_ad:SNP)-[:ASSOCIATED_WITH]->(d_ad:Disease)
WHERE d_ad.name CONTAINS "Alzheimer"
WITH g, min(toFloat(s_ad.p_value)) AS ad_pval

MATCH (g)<-[:LINKED_TO]-(s_met:SNP)-[:ASSOCIATED_WITH]->(d_met:Disease)
WHERE d_met.name IN ["type 2 diabetes", "insulin resistance", "body mass index"]
WITH g, ad_pval, min(toFloat(s_met.p_value)) AS met_pval

RETURN g.symbol AS gene,
       ad_pval AS min_ad_pval,
       met_pval AS min_metabolic_pval,
       ad_pval * met_pval AS combined_score
ORDER BY combined_score ASC
LIMIT 50
""".strip(),

    # 4. Drugs near GLP1R and cognitive biomarkers (repurposing candidates)
    "repurposing_candidates": """
MATCH (d:Drug)-[:TARGETS|RELATED_TO*1..2]->(g:Gene)
WHERE g.symbol IN ["GLP1R", "GIPR", "GCGR", "INSR", "IGF1R"]
WITH d, collect(DISTINCT g.symbol) AS target_genes

OPTIONAL MATCH (d)-[:PROTECTIVE_SIGNAL]->(f:FAERSReport)
WHERE f.ror < 1.0

OPTIONAL MATCH (d)-[:MENTIONS|RELATED_TO*1..2]-(p:Paper)

RETURN d.name AS drug,
       target_genes,
       count(DISTINCT f) AS protective_signals,
       count(DISTINCT p) AS literature_count
ORDER BY protective_signals DESC, literature_count DESC
LIMIT 50
""".strip(),

    # 5. Genes with GWAS hits in both T2D and AD
    "genetic_overlap": """
MATCH (g:Gene)<-[:LINKED_TO]-(s1:SNP)-[:ASSOCIATED_WITH]->(d1:Disease)
WHERE d1.name CONTAINS "Alzheimer"
WITH g, collect(DISTINCT s1.rsid) AS ad_snps

MATCH (g)<-[:LINKED_TO]-(s2:SNP)-[:ASSOCIATED_WITH]->(d2:Disease)
WHERE d2.name = "type 2 diabetes"
WITH g, ad_snps, collect(DISTINCT s2.rsid) AS t2d_snps

RETURN g.symbol AS gene,
       size(ad_snps) AS ad_hit_count,
       size(t2d_snps) AS t2d_hit_count,
       ad_snps AS ad_snp_ids,
       t2d_snps AS t2d_snp_ids
ORDER BY ad_hit_count + t2d_hit_count DESC
LIMIT 30
""".strip(),

    # 6. Pathways co-mentioned in metabolic AND AD literature
    "pathway_bridges": """
MATCH (path:Pathway)-[:PATHWAY_MEMBER|RELATED_TO*1..2]-(e1)-[:RELATED_TO|MENTIONS*1..2]-(p1:Paper)
WHERE p1.abstract CONTAINS "Alzheimer" OR p1.title CONTAINS "Alzheimer"
WITH path, count(DISTINCT p1) AS ad_papers

MATCH (path)-[:PATHWAY_MEMBER|RELATED_TO*1..2]-(e2)-[:RELATED_TO|MENTIONS*1..2]-(p2:Paper)
WHERE p2.abstract CONTAINS "diabetes"
   OR p2.abstract CONTAINS "insulin"
   OR p2.abstract CONTAINS "metabolic"
WITH path, ad_papers, count(DISTINCT p2) AS metabolic_papers
WHERE ad_papers > 0 AND metabolic_papers > 0

RETURN path.name AS pathway,
       ad_papers,
       metabolic_papers
ORDER BY ad_papers + metabolic_papers DESC
LIMIT 20
""".strip(),

    # 7. Drugs with ≥5 literature mentions in AD but no active trial
    "trial_gaps": """
MATCH (d:Drug)-[:MENTIONS|RELATED_TO*1..2]-(p:Paper)
WHERE p.abstract CONTAINS "Alzheimer" OR p.title CONTAINS "Alzheimer"
WITH d, count(DISTINCT p) AS lit_count
WHERE lit_count >= 5

WHERE NOT EXISTS {
  MATCH (d)<-[:TESTS]-(t:Trial)-[:FOR]->(dis:Disease)
  WHERE dis.name CONTAINS "Alzheimer"
    AND t.status IN ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING"]
}

RETURN d.name AS drug, lit_count
ORDER BY lit_count DESC
""".strip(),

    # 8. SNP → gene → approved drug paths
    "gwas_snp_to_drug": """
MATCH (s:SNP)-[:ASSOCIATED_WITH]->(dis:Disease)
WHERE dis.name CONTAINS "Alzheimer"
WITH s

MATCH path = (s)-[:LINKED_TO]->(g:Gene)<-[:TARGETS]-(d:Drug)
RETURN s.rsid AS snp,
       g.symbol AS gene,
       d.name AS drug,
       length(path) AS path_length
ORDER BY g.symbol, d.name
LIMIT 100
""".strip(),

    # 9. Active trials testing drugs that hit bridge genes
    "open_trials_bridge_genes": """
MATCH (g:Gene)<-[:LINKED_TO]-(s1:SNP)-[:ASSOCIATED_WITH]->(d1:Disease)
WHERE d1.name CONTAINS "Alzheimer"
WITH g

MATCH (g)<-[:LINKED_TO]-(s2:SNP)-[:ASSOCIATED_WITH]->(d2:Disease)
WHERE d2.name IN ["type 2 diabetes", "insulin resistance"]
WITH g

MATCH (t:Trial)-[:TESTS]->(d:Drug)-[:TARGETS]->(g)
WHERE t.status IN ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING"]

RETURN t.nct_id AS nct_id,
       t.title AS trial_title,
       t.status AS status,
       d.name AS drug,
       g.symbol AS bridge_gene
ORDER BY t.status, d.name
""".strip(),

    # 10. All FAERS protective drugs ranked by literature evidence
    "protective_drugs_ranked": """
MATCH (d:Drug)-[:PROTECTIVE_SIGNAL]->(f:FAERSReport)
WHERE f.ror < 1.0
WITH d, min(f.ror) AS min_ror, count(DISTINCT f) AS protective_reactions

OPTIONAL MATCH (d)-[:MENTIONS|RELATED_TO*1..2]-(p:Paper)
WITH d, min_ror, protective_reactions, count(DISTINCT p) AS lit_count

RETURN d.name AS drug,
       min_ror AS best_ror,
       protective_reactions,
       lit_count AS literature_mentions
ORDER BY min_ror ASC, lit_count DESC
""".strip(),

    # 11. Everything connected to semaglutide within 2 hops
    "semaglutide_neighbors": """
MATCH (sema:Drug)
WHERE toLower(sema.name) CONTAINS "semaglutide"
WITH sema

MATCH (sema)-[r*1..2]-(neighbor)
WHERE neighbor <> sema
WITH DISTINCT neighbor,
     labels(neighbor)[0] AS node_type,
     [rel IN r | type(rel)] AS relationship_types

RETURN neighbor.name AS name,
       node_type,
       relationship_types
ORDER BY node_type, name
LIMIT 200
""".strip(),
}


def run_query(
    driver: Any,
    name: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Execute a named query and return results as a list of dicts.

    Args:
        driver: neo4j.Driver instance.
        name: Query name from QUERIES dict.
        limit: Optional LIMIT override appended to the query.

    Returns:
        List of result row dicts.
    """
    if name not in QUERIES:
        raise ValueError(f"Unknown query: {name!r}. Available: {list(QUERIES.keys())}")

    cypher = QUERIES[name]
    if limit:
        cypher = cypher.rstrip() + f"\nLIMIT {limit}"

    logger.info("Running query: %s", name)
    with driver.session() as session:
        result = session.run(cypher)
        rows = [dict(record) for record in result]

    logger.info("Query '%s' returned %d rows", name, len(rows))
    return rows
