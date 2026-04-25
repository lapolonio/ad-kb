"""Named Cypher queries for the AD Knowledge Graph."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

QUERIES: dict[str, str] = {
    # 1. Drugs with FAERS protective signal + bridge gene (AD or metabolic GWAS)
    #    but NO active AD trial — i.e. repurposing whitespace.
    #    report_count >= 2 removes single case-report noise (n=1 RORs are unreliable).
    "whitespace_opportunity": """
MATCH (d:Drug)-[:PROTECTIVE_SIGNAL]->(f:FAERSReport)
WHERE f.cohort = 'all' AND f.ror < 1.0 AND f.report_count >= 2
WITH d,
     round(min(f.ror), 3)      AS best_ror,
     round(min(f.ci_upper), 3) AS best_ci_upper
MATCH (d)-[:TARGETS]->(g:Gene)<-[:LINKED_TO]-(s:SNP)-[:ASSOCIATED_WITH]->(dis:Disease)
WHERE dis.name CONTAINS "Alzheimer"
   OR dis.name IN ["type 2 diabetes", "insulin resistance", "body mass index",
                   "glycated hemoglobin", "fasting glucose", "obesity"]
WITH DISTINCT d, g, dis, best_ror, best_ci_upper
WHERE NOT EXISTS {
  MATCH (d)<-[:TESTS]-(t:Trial)-[:FOR]->(dis2:Disease)
  WHERE dis2.name CONTAINS "Alzheimer"
    AND t.status IN ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING"]
}
RETURN d.name AS drug, d.id AS drug_id,
       collect(DISTINCT g.symbol) AS bridge_genes,
       collect(DISTINCT dis.name) AS gwas_traits,
       best_ror,
       best_ci_upper < 1.0 AS signal_significant
ORDER BY best_ror ASC
""".strip(),

    # 2. Drugs with all three signals: FAERS protective + GWAS gene + literature.
    #    Includes metabolic GWAS traits (T2DM, BMI, insulin resistance) because
    #    GLP-1 drugs target GLP1R/GIPR which have metabolic — not direct AD — GWAS hits.
    #    Excludes drugs where significant overall adverse signals outnumber
    #    significant protective signals (net-adverse FAERS profile).
    #    This catches semaglutide (0 sig protective, 3 sig adverse) without
    #    excluding liraglutide (2 sig protective, 1 sig adverse = net-protective).
    "triple_convergence": """
MATCH (d:Drug)-[:PROTECTIVE_SIGNAL]->(f:FAERSReport)
WHERE f.cohort = 'all' AND f.ror < 1.0 AND f.report_count >= 2
WITH d,
     count { (d)-[:PROTECTIVE_SIGNAL]->(fp:FAERSReport)
             WHERE fp.cohort = 'all' AND fp.ci_upper < 1.0
               AND fp.report_count >= 10 } AS sig_protective,
     count { (d)-[:ADVERSE_SIGNAL]->(fa:FAERSReport)
             WHERE fa.cohort = 'all' AND fa.ci_lower > 1.0
               AND fa.report_count >= 10 } AS sig_adverse
WHERE sig_adverse <= sig_protective

MATCH (d)-[:TARGETS|RELATED_TO*1..2]->(g:Gene)<-[:LINKED_TO]-(s:SNP)
   -[:ASSOCIATED_WITH]->(dis:Disease)
WHERE dis.name CONTAINS "Alzheimer"
   OR dis.name IN ["type 2 diabetes", "insulin resistance", "body mass index",
                   "glycated hemoglobin", "fasting glucose", "obesity"]
WITH d, count(DISTINCT s) AS gwas_snps, collect(DISTINCT dis.name) AS gwas_traits

MATCH (d)-[:MENTIONS|RELATED_TO*1..2]-(p:Paper)
WITH d, gwas_snps, gwas_traits, count(DISTINCT p) AS lit_count
WHERE lit_count >= 1

RETURN d.name AS drug,
       gwas_snps AS gwas_support,
       gwas_traits,
       lit_count AS literature_count
ORDER BY gwas_snps DESC, lit_count DESC
""".strip(),

    # 3. Genes ranked by product of AD × metabolic GWAS association p-values.
    #    p_value lives on the ASSOCIATED_WITH relationship, not the SNP node.
    #    "NR" is excluded — it is the GWAS Catalog placeholder for unmapped variants,
    #    not a real gene symbol.
    "bridge_genes_ranked": """
MATCH (g:Gene)<-[:LINKED_TO]-(s_ad:SNP)-[r_ad:ASSOCIATED_WITH]->(d_ad:Disease)
WHERE d_ad.name CONTAINS "Alzheimer"
  AND g.symbol <> "NR"
WITH g, min(toFloat(r_ad.p_value)) AS ad_pval

MATCH (g)<-[:LINKED_TO]-(s_met:SNP)-[r_met:ASSOCIATED_WITH]->(d_met:Disease)
WHERE d_met.name IN ["type 2 diabetes", "insulin resistance", "body mass index"]
WITH g, ad_pval, min(toFloat(r_met.p_value)) AS met_pval

RETURN g.symbol AS gene,
       ad_pval AS min_ad_pval,
       met_pval AS min_metabolic_pval,
       ad_pval * met_pval AS combined_score
ORDER BY combined_score ASC
LIMIT 50
""".strip(),

    # 4. Drugs near GLP1R and cognitive biomarkers (repurposing candidates)
    #    SLC5A2 (SGLT2) included: SGLT2i have no direct AD GWAS hits via this gene,
    #    but their T2DM mechanism (glucose transport → insulin sensitivity) is relevant.
    "repurposing_candidates": """
MATCH (d:Drug)-[:TARGETS|RELATED_TO*1..2]->(g:Gene)
WHERE g.symbol IN ["GLP1R", "GIPR", "GCGR", "INSR", "IGF1R", "SLC5A2"]
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

    # 5. Genes with GWAS hits in both T2D and AD.
    #    "NR" excluded — GWAS Catalog placeholder for unmapped variants.
    "genetic_overlap": """
MATCH (g:Gene)<-[:LINKED_TO]-(s1:SNP)-[:ASSOCIATED_WITH]->(d1:Disease)
WHERE d1.name CONTAINS "Alzheimer"
  AND g.symbol <> "NR"
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

    # 8. SNP → gene → drug paths for AD and metabolic GWAS traits.
    #    Includes metabolic traits because GLP-1 drugs target GLP1R/GIPR (T2DM GWAS),
    #    not genes with direct AD GWAS hits. "NR" excluded.
    "gwas_snp_to_drug": """
MATCH (s:SNP)-[:ASSOCIATED_WITH]->(dis:Disease)
WHERE dis.name CONTAINS "Alzheimer"
   OR dis.name IN ["type 2 diabetes", "insulin resistance", "body mass index"]
WITH s, dis.name AS trait

MATCH path = (s)-[:LINKED_TO]->(g:Gene)<-[:TARGETS]-(d:Drug)
WHERE g.symbol <> "NR"
RETURN s.rsid AS snp,
       trait,
       g.symbol AS gene,
       d.name AS drug,
       length(path) AS path_length
ORDER BY trait, g.symbol, d.name
LIMIT 100
""".strip(),

    # 9. Active trials testing drugs that hit bridge genes.
    #    Bridge defined broadly: gene with GWAS hits in AD OR metabolic traits,
    #    since GLP-1 drugs target GLP1R/GIPR which have T2DM (not direct AD) GWAS hits.
    #    "NR" excluded — GWAS Catalog unmapped-variant placeholder.
    "open_trials_bridge_genes": """
MATCH (g:Gene)<-[:LINKED_TO]-(s:SNP)-[:ASSOCIATED_WITH]->(dis:Disease)
WHERE (dis.name CONTAINS "Alzheimer"
    OR dis.name IN ["type 2 diabetes", "insulin resistance", "body mass index"])
  AND g.symbol <> "NR"
WITH DISTINCT g

MATCH (t:Trial)-[:TESTS]->(d:Drug)-[:TARGETS]->(g)
WHERE t.status IN ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING"]
  AND toLower(t.title) CONTAINS "alzheimer"

RETURN t.nct_id AS nct_id,
       t.title AS trial_title,
       t.status AS status,
       d.name AS drug,
       g.symbol AS bridge_gene
ORDER BY t.status, d.name
LIMIT 50
""".strip(),

    # 10. FAERS protective drugs ranked by literature evidence.
    #     Anchored to cohort='all' so subpopulation-only protective signals
    #     (e.g. T2DM-only) do not inflate the ranking for drugs that are
    #     otherwise adverse in the general population.
    #     best_ci_upper and signal_significant expose statistical quality so
    #     small-n signals (tirzepatide Dementia n=2, ertugliflozin n=3) are
    #     visibly distinguished from robust signals (liraglutide CI entirely < 1).
    "protective_drugs_ranked": """
MATCH (d:Drug)-[:PROTECTIVE_SIGNAL]->(f:FAERSReport)
WHERE f.cohort = 'all' AND f.report_count >= 2
WITH d,
     round(min(f.ror), 3)      AS best_ror,
     round(min(f.ci_upper), 3) AS best_ci_upper,
     count(DISTINCT f)         AS protective_reactions

OPTIONAL MATCH (d)-[:MENTIONS|RELATED_TO*1..2]-(p:Paper)
WITH d, best_ror, best_ci_upper, protective_reactions, count(DISTINCT p) AS lit_count

RETURN d.name AS drug,
       best_ror,
       best_ci_upper,
       best_ci_upper < 1.0     AS signal_significant,
       protective_reactions,
       lit_count AS literature_mentions
ORDER BY best_ror ASC, lit_count DESC
""".strip(),

    # 12. FAERS ROR across all cohorts for drugs with a baseline protective signal.
    #     Use this to confirm the signal holds in sub-populations (T2DM-only,
    #     elderly, post-2020) and is not an artifact of the full reporting pool.
    "faers_sensitivity_cohorts": """
MATCH (d:Drug)-[:PROTECTIVE_SIGNAL]->(f_base:FAERSReport)
WHERE f_base.cohort = 'all' AND f_base.report_count >= 2
WITH DISTINCT d
MATCH (d)-[rel:PROTECTIVE_SIGNAL|ADVERSE_SIGNAL]->(f:FAERSReport)
WHERE f.report_count >= 2
RETURN d.name AS drug,
       f.cohort AS cohort,
       f.reaction AS reaction,
       round(f.ror, 3) AS ror,
       round(f.ci_lower, 3) AS ci_lower,
       round(f.ci_upper, 3) AS ci_upper,
       f.report_count AS n,
       type(rel) AS signal_type
ORDER BY d.name, f.reaction, f.cohort
""".strip(),

    # 13. For each drug with a baseline protective signal, count how many of the
    #     6 MedDRA AD reactions show ROR < 1 in the "all" cohort. A drug that is
    #     protective across multiple AD phenotypes is a stronger candidate.
    "faers_cross_reaction_consistency": """
MATCH (d:Drug)-[:PROTECTIVE_SIGNAL]->(f:FAERSReport)
WHERE f.cohort = 'all' AND f.report_count >= 2
WITH d,
     count(DISTINCT f.reaction) AS protective_reactions,
     round(avg(f.ror), 3) AS mean_ror,
     round(min(f.ror), 3) AS min_ror,
     max(f.ci_upper) AS max_ci_upper
RETURN d.name AS drug,
       protective_reactions,
       mean_ror,
       min_ror,
       max_ci_upper < 1.0 AS all_ci_below_1
ORDER BY protective_reactions DESC, mean_ror ASC
""".strip(),

    # 14. Side-by-side ROR comparison: overall vs T2DM-only vs elderly cohort.
    #     Drugs protective across all three sub-populations are robust against
    #     the healthy-user and T2DM-confounding critiques.
    "faers_subpopulation_comparison": """
MATCH (d:Drug)-[:PROTECTIVE_SIGNAL]->(f_all:FAERSReport)
WHERE f_all.cohort = 'all' AND f_all.report_count >= 2
WITH d, round(min(f_all.ror), 3) AS overall_ror

OPTIONAL MATCH (d)-[:PROTECTIVE_SIGNAL]->(f_t2dm:FAERSReport)
WHERE f_t2dm.cohort = 't2dm'
WITH d, overall_ror, round(min(f_t2dm.ror), 3) AS t2dm_ror

OPTIONAL MATCH (d)-[:PROTECTIVE_SIGNAL]->(f_eld:FAERSReport)
WHERE f_eld.cohort = 'elderly'
WITH d, overall_ror, t2dm_ror, round(min(f_eld.ror), 3) AS elderly_ror

OPTIONAL MATCH (d)-[:PROTECTIVE_SIGNAL]->(f_post:FAERSReport)
WHERE f_post.cohort = 'post_2020'
WITH d, overall_ror, t2dm_ror, elderly_ror,
     round(min(f_post.ror), 3) AS post_2020_ror

RETURN d.name AS drug,
       overall_ror,
       t2dm_ror,
       elderly_ror,
       post_2020_ror,
       [x IN [t2dm_ror, elderly_ror, post_2020_ror] WHERE x IS NOT NULL AND x < 1.0]
         AS cohorts_protective
ORDER BY overall_ror ASC
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
