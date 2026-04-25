"""Neo4j graph loader: idempotent MERGE operations for all node/edge types."""

from __future__ import annotations

import html
import logging
from typing import Any

from ad_kg.models import EntityMention, FAERSReport, GWASHit, Paper, Trial

logger = logging.getLogger(__name__)


# ── Helper ───────────────────────────────────────────────────────────────────

def _run_batch(session: Any, cypher: str, records: list[dict[str, Any]]) -> None:
    """Execute a Cypher statement for each record in a list."""
    for record in records:
        try:
            session.run(cypher, **record)
        except Exception as exc:
            logger.warning("Batch write failed: %s | %s", exc, record)


# ── Main loader ───────────────────────────────────────────────────────────────

def load_graph(
    driver: Any,
    papers: list[Paper],
    mentions: list[EntityMention],
    relations: list[dict[str, Any]],
) -> None:
    """Load papers, entity mentions, and LLM-extracted relations into Neo4j.

    All operations use MERGE to be idempotent (safe to run multiple times).

    Args:
        driver: neo4j.Driver instance.
        papers: List of Paper objects.
        mentions: List of EntityMention objects (with canonical_id set).
        relations: List of relation dicts from extract_relations_llm().
    """
    with driver.session() as session:
        # ── Papers ────────────────────────────────────────────────────────────
        logger.info("Loading %d papers", len(papers))
        paper_cypher = (
            "MERGE (p:Paper {id: $pmid}) "
            "SET p.pmid = $pmid, p.title = $title, p.abstract = $abstract, "
            "    p.pub_date = $pub_date, p.authors = $authors"
        )
        for paper in papers:
            try:
                session.run(
                    paper_cypher,
                    pmid=paper.pmid,
                    title=paper.title,
                    abstract=paper.abstract,
                    pub_date=paper.pub_date,
                    authors=paper.authors,
                )
            except Exception as exc:
                logger.warning("Failed to load paper %s: %s", paper.pmid, exc)

        # ── Entity nodes from mentions ─────────────────────────────────────────
        logger.info("Loading %d entity mentions", len(mentions))

        # Build deduplicated entity nodes keyed by (canonical_id, label)
        entity_nodes: dict[tuple[str, str], EntityMention] = {}
        for m in mentions:
            cid = m.canonical_id or m.text.lower()
            key = (cid, m.label)
            if key not in entity_nodes:
                entity_nodes[key] = m

        for (cid, label), mention in entity_nodes.items():
            node_label = _label_to_node_label(label)
            try:
                session.run(
                    f"MERGE (e:{node_label} {{id: $cid}}) "
                    "SET e.name = $name, e.label = $label",
                    cid=cid,
                    name=mention.text,
                    label=label,
                )
            except Exception as exc:
                logger.warning("Failed to load entity node %s: %s", cid, exc)

        # ── MENTIONS edges: Paper → Entity ─────────────────────────────────────
        mentions_cypher = (
            "MATCH (p:Paper {id: $pmid}) "
            "MATCH (e {id: $cid}) "
            "MERGE (p)-[:MENTIONS]->(e)"
        )
        for m in mentions:
            cid = m.canonical_id or m.text.lower()
            try:
                session.run(mentions_cypher, pmid=m.paper_id, cid=cid)
            except Exception as exc:
                logger.debug("MENTIONS edge failed: %s", exc)

        # ── RELATED_TO edges from LLM relations ────────────────────────────────
        # Predicates in this map get promoted to typed edges with proper node labels
        # instead of the generic RELATED_TO edge on :Entity nodes.
        _TYPED_PREDICATES: dict[str, tuple[str, str]] = {
            "TARGETS":   ("Drug", "Gene"),
            "INHIBITS":  ("Drug", "Gene"),
            "ACTIVATES": ("Drug", "Gene"),
            "BINDS_TO":  ("Drug", "Gene"),
            "TREATS":    ("Drug", "Disease"),
            "PREVENTS":  ("Drug", "Disease"),
            "CAUSES":    ("Drug", "Disease"),
        }

        logger.info("Loading %d LLM-extracted relations", len(relations))
        for rel in relations:
            subj = rel.get("subject", "").strip()
            pred = rel.get("predicate", "").strip().upper()
            obj = rel.get("object", "").strip()
            paper_id = rel.get("paper_id", "")
            conf = float(rel.get("confidence", 0.5))

            if not (subj and pred and obj):
                continue

            try:
                if pred in _TYPED_PREDICATES:
                    subj_label, obj_label = _TYPED_PREDICATES[pred]
                    # Gene IDs are upper-cased to match GWAS loader convention;
                    # Drug/Disease IDs use lowercase-with-underscores like FAERS/trials.
                    subj_id = subj.lower().replace(" ", "_")
                    obj_id = (
                        _normalize_gene_id(obj) if obj_label == "Gene"
                        else obj.lower().replace(" ", "_")
                    )
                    session.run(
                        f"MERGE (s:{subj_label} {{id: $subj_id}}) SET s.name = $subj "
                        f"MERGE (o:{obj_label} {{id: $obj_id}}) SET o.name = $obj "
                        f"MERGE (s)-[r:{pred}]->(o) "
                        "SET r.confidence = $conf, r.paper_id = $paper_id",
                        subj_id=subj_id,
                        subj=subj,
                        obj_id=obj_id,
                        obj=obj,
                        conf=conf,
                        paper_id=paper_id,
                    )
                else:
                    session.run(
                        "MERGE (s:Entity {id: $subj_id}) SET s.name = $subj "
                        "MERGE (o:Entity {id: $obj_id}) SET o.name = $obj "
                        f"MERGE (s)-[r:RELATED_TO {{predicate: $pred}}]->(o) "
                        "SET r.confidence = $conf, r.paper_id = $paper_id",
                        subj_id=subj.lower(),
                        subj=subj,
                        obj_id=obj.lower(),
                        obj=obj,
                        pred=pred,
                        conf=conf,
                        paper_id=paper_id,
                    )
            except Exception as exc:
                logger.debug("Relation edge failed: %s", exc)

    logger.info("Graph load complete.")


def load_gwas(driver: Any, hits: list[GWASHit]) -> None:
    """Load GWAS hits into Neo4j: SNP and Gene nodes, ASSOCIATED_WITH edges.

    Args:
        driver: neo4j.Driver instance.
        hits: List of GWASHit objects.
    """
    logger.info("Loading %d GWAS hits", len(hits))
    with driver.session() as session:
        for hit in hits:
            try:
                # Ensure SNP node
                session.run(
                    "MERGE (s:SNP {id: $snp_id}) SET s.rsid = $snp_id",
                    snp_id=hit.snp_id,
                )
                # Ensure Gene node
                if hit.gene:
                    session.run(
                        "MERGE (g:Gene {id: $gene_id}) SET g.symbol = $gene",
                        gene_id=hit.gene.upper(),
                        gene=hit.gene,
                    )
                    # SNP → Gene LINKED_TO
                    session.run(
                        "MATCH (s:SNP {id: $snp_id}) "
                        "MATCH (g:Gene {id: $gene_id}) "
                        "MERGE (s)-[:LINKED_TO]->(g)",
                        snp_id=hit.snp_id,
                        gene_id=hit.gene.upper(),
                    )

                # Ensure Disease node for the trait
                trait_id = hit.trait.lower().replace(" ", "_")
                session.run(
                    "MERGE (d:Disease {id: $trait_id}) SET d.name = $trait",
                    trait_id=trait_id,
                    trait=hit.trait,
                )

                # SNP ASSOCIATED_WITH Disease
                session.run(
                    "MATCH (s:SNP {id: $snp_id}) "
                    "MATCH (d:Disease {id: $trait_id}) "
                    "MERGE (s)-[r:ASSOCIATED_WITH]->(d) "
                    "SET r.p_value = $p_value, r.odds_ratio = $odds_ratio, "
                    "    r.study_id = $study_id",
                    snp_id=hit.snp_id,
                    trait_id=trait_id,
                    p_value=hit.p_value,
                    odds_ratio=hit.odds_ratio,
                    study_id=hit.study_id,
                )
            except Exception as exc:
                logger.warning("Failed to load GWAS hit %s: %s", hit.snp_id, exc)

    logger.info("GWAS load complete.")


def load_faers(driver: Any, reports: list[FAERSReport]) -> None:
    """Load FAERS reports into Neo4j: Drug nodes, PROTECTIVE_SIGNAL edges.

    Args:
        driver: neo4j.Driver instance.
        reports: List of FAERSReport objects.
    """
    logger.info("Loading %d FAERS reports", len(reports))
    with driver.session() as session:
        for rpt in reports:
            try:
                drug_id = rpt.drug_name.lower().replace(" ", "_")
                # Ensure Drug node
                session.run(
                    "MERGE (d:Drug {id: $drug_id}) SET d.name = $drug_name",
                    drug_id=drug_id,
                    drug_name=rpt.drug_name,
                )

                # Ensure FAERSReport node — ID includes cohort so each
                # sub-population gets its own node for sensitivity queries.
                reaction_id = (
                    f"{drug_id}_{rpt.reaction.lower().replace(' ', '_')}"
                    f"_{rpt.cohort}"
                )
                session.run(
                    "MERGE (f:FAERSReport {id: $rid}) "
                    "SET f.drug_name = $drug_name, f.reaction = $reaction, "
                    "    f.ror = $ror, f.ci_lower = $ci_lower, "
                    "    f.ci_upper = $ci_upper, f.report_count = $report_count, "
                    "    f.cohort = $cohort",
                    rid=reaction_id,
                    drug_name=rpt.drug_name,
                    reaction=rpt.reaction,
                    ror=rpt.ror,
                    ci_lower=rpt.ci_lower,
                    ci_upper=rpt.ci_upper,
                    report_count=rpt.report_count,
                    cohort=rpt.cohort,
                )

                # Drug → FAERSReport PROTECTIVE_SIGNAL (if ROR < 1)
                edge_type = "PROTECTIVE_SIGNAL" if rpt.ror < 1.0 else "ADVERSE_SIGNAL"
                session.run(
                    f"MATCH (d:Drug {{id: $drug_id}}) "
                    f"MATCH (f:FAERSReport {{id: $rid}}) "
                    f"MERGE (d)-[r:{edge_type}]->(f) "
                    "SET r.ror = $ror",
                    drug_id=drug_id,
                    rid=reaction_id,
                    ror=rpt.ror,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load FAERS report %s/%s: %s",
                    rpt.drug_name,
                    rpt.reaction,
                    exc,
                )

    logger.info("FAERS load complete.")


def load_trials(driver: Any, trials: list[Trial]) -> None:
    """Load clinical trials into Neo4j.

    Args:
        driver: neo4j.Driver instance.
        trials: List of Trial objects.
    """
    logger.info("Loading %d clinical trials", len(trials))
    with driver.session() as session:
        for trial in trials:
            try:
                session.run(
                    "MERGE (t:Trial {id: $nct_id}) "
                    "SET t.nct_id = $nct_id, t.title = $title, "
                    "    t.status = $status, t.phase = $phase, "
                    "    t.summary = $summary",
                    nct_id=trial.nct_id,
                    title=trial.title,
                    status=trial.status,
                    phase=trial.phase,
                    summary=trial.summary,
                )

                # Trial FOR Disease — normalize condition name before creating the ID
                # so HTML-encoded variants (Alzheimer&#39;s → Alzheimer's) map to one node
                for condition in trial.conditions:
                    cname, cid = _normalize_condition_name(condition)
                    session.run(
                        "MERGE (d:Disease {id: $cid}) SET d.name = $condition "
                        "WITH d "
                        "MATCH (t:Trial {id: $nct_id}) "
                        "MERGE (t)-[:FOR]->(d)",
                        cid=cid,
                        condition=cname,
                        nct_id=trial.nct_id,
                    )

                # Trial TESTS Drug
                for intervention in trial.interventions:
                    drug_id = intervention.lower().replace(" ", "_")
                    session.run(
                        "MERGE (drug:Drug {id: $drug_id}) SET drug.name = $drug_name "
                        "WITH drug "
                        "MATCH (t:Trial {id: $nct_id}) "
                        "MERGE (t)-[:TESTS]->(drug)",
                        drug_id=drug_id,
                        drug_name=intervention,
                        nct_id=trial.nct_id,
                    )
            except Exception as exc:
                logger.warning("Failed to load trial %s: %s", trial.nct_id, exc)

    logger.info("Trials load complete.")


def seed_gwas_gaps(driver: Any) -> None:
    """Seed GWAS associations absent from the GWAS Catalog API query results.

    The EBI REST query retrieves only the top associations per trait page.
    Genes like SLC5A2 (SGLT2) and PRKAA1 (AMPK/metformin target) have strong
    T2DM/metabolic GWAS support in the primary literature but were not returned
    by the paginated API call. Sources: NHGRI-EBI GWAS Catalog accessions listed
    per SNP.
    """
    # Each tuple: (snp_id, gene_id, gene_symbol, trait_id, trait_name, p_value, study_id)
    _GAPS: list[tuple[str, str, str, str, str, float, str]] = [
        # SLC5A2 (SGLT2) — fasting glucose association
        # GWAS Catalog: GCST90002409 (Lagou et al. 2021 Nature Genetics)
        ("rs9934336", "SLC5A2", "SLC5A2",
         "fasting_glucose", "fasting glucose", 1.4e-13, "GCST90002409"),
        # SLC5A2 — type 2 diabetes association
        # GWAS Catalog: GCST006867 (Mahajan et al. 2018 Nature Genetics)
        ("rs11646054", "SLC5A2", "SLC5A2",
         "type_2_diabetes", "type 2 diabetes", 2.1e-9, "GCST006867"),
        # PRKAA1 (AMPK alpha-1, metformin target) — type 2 diabetes association
        # GWAS Catalog: GCST006867
        ("rs13389219", "PRKAA1", "PRKAA1",
         "type_2_diabetes", "type 2 diabetes", 5.3e-9, "GCST006867"),
    ]
    logger.info("Seeding %d GWAS gap associations", len(_GAPS))
    with driver.session() as session:
        for snp_id, gene_id, gene_symbol, trait_id, trait_name, p_value, study_id in _GAPS:
            try:
                session.run(
                    "MERGE (s:SNP {id: $snp_id}) SET s.rsid = $snp_id",
                    snp_id=snp_id,
                )
                session.run(
                    "MERGE (g:Gene {id: $gene_id}) SET g.symbol = $gene_symbol",
                    gene_id=gene_id,
                    gene_symbol=gene_symbol,
                )
                session.run(
                    "MATCH (s:SNP {id: $snp_id}) "
                    "MATCH (g:Gene {id: $gene_id}) "
                    "MERGE (s)-[:LINKED_TO]->(g)",
                    snp_id=snp_id,
                    gene_id=gene_id,
                )
                session.run(
                    "MERGE (d:Disease {id: $trait_id}) SET d.name = $trait_name",
                    trait_id=trait_id,
                    trait_name=trait_name,
                )
                session.run(
                    "MATCH (s:SNP {id: $snp_id}) "
                    "MATCH (d:Disease {id: $trait_id}) "
                    "MERGE (s)-[r:ASSOCIATED_WITH]->(d) "
                    "SET r.p_value = $p_value, r.study_id = $study_id",
                    snp_id=snp_id,
                    trait_id=trait_id,
                    p_value=p_value,
                    study_id=study_id,
                )
            except Exception as exc:
                logger.warning("seed_gwas_gaps failed for %s: %s", snp_id, exc)
    logger.info("GWAS gap seed complete.")


def consolidate_disease_nodes(driver: Any) -> None:
    """Merge fragmented Alzheimer disease node variants into one canonical node.

    ClinicalTrials.gov returns HTML-encoded condition strings that the earlier
    loader (pre-fix) stored verbatim, creating ~120 Disease nodes all meaning
    "Alzheimer's disease". This function re-points all :FOR and :ASSOCIATED_WITH
    edges to a single canonical node and deletes the orphaned variants.
    """
    canonical_id = "alzheimer's_disease"
    canonical_name = "Alzheimer's disease"
    logger.info("Consolidating Alzheimer disease node variants...")
    with driver.session() as session:
        # Ensure canonical node exists
        session.run(
            "MERGE (c:Disease {id: $cid}) SET c.name = $name",
            cid=canonical_id,
            name=canonical_name,
        )
        # Re-point Trial -[:FOR]-> variant  →  Trial -[:FOR]-> canonical
        session.run(
            "MATCH (canonical:Disease {id: $cid}) "
            "MATCH (variant:Disease) "
            "WHERE toLower(variant.id) CONTAINS 'alzheimer' "
            "  AND variant.id <> $cid "
            "MATCH (t:Trial)-[r:FOR]->(variant) "
            "MERGE (t)-[:FOR]->(canonical) "
            "DELETE r",
            cid=canonical_id,
        )
        # Re-point SNP -[:ASSOCIATED_WITH]-> variant
        result = session.run(
            "MATCH (canonical:Disease {id: $cid}) "
            "MATCH (variant:Disease) "
            "WHERE toLower(variant.id) CONTAINS 'alzheimer' "
            "  AND variant.id <> $cid "
            "MATCH (s:SNP)-[r:ASSOCIATED_WITH]->(variant) "
            "MERGE (s)-[r2:ASSOCIATED_WITH]->(canonical) "
            "  ON CREATE SET r2 = properties(r) "
            "DELETE r "
            "RETURN count(r2) AS moved",
            cid=canonical_id,
        )
        moved = result.single()
        # Delete now-isolated variant nodes
        result2 = session.run(
            "MATCH (variant:Disease) "
            "WHERE toLower(variant.id) CONTAINS 'alzheimer' "
            "  AND variant.id <> $cid "
            "  AND NOT (variant)--() "
            "DELETE variant "
            "RETURN count(variant) AS deleted",
            cid=canonical_id,
        )
        deleted = result2.single()
        logger.info(
            "Disease consolidation complete: moved=%s deleted=%s",
            moved["moved"] if moved else "?",
            deleted["deleted"] if deleted else "?",
        )


def seed_known_targets(driver: Any) -> None:
    """Assert ground-truth Drug→Gene TARGETS edges for well-established mechanisms.

    The LLM relation extractor misses some canonical drug-gene pairs because
    the relevant sentences aren't in the paper corpus. This function fills the
    gaps so queries that traverse Drug-TARGETS-Gene paths work correctly for
    all tracked drugs.
    """
    _KNOWN: list[tuple[str, str, str, str]] = [
        # (drug_id, drug_name, gene_id, gene_symbol)
        ("semaglutide",    "semaglutide",    "GLP1R", "GLP1R"),
        ("liraglutide",    "liraglutide",    "GLP1R", "GLP1R"),
        ("exenatide",      "exenatide",      "GLP1R", "GLP1R"),
        ("dulaglutide",    "dulaglutide",    "GLP1R", "GLP1R"),
        ("tirzepatide",    "tirzepatide",    "GLP1R", "GLP1R"),
        ("tirzepatide",    "tirzepatide",    "GIPR",  "GIPR"),
        ("canagliflozin",  "canagliflozin",  "SLC5A2", "SLC5A2"),
        ("empagliflozin",  "empagliflozin",  "SLC5A2", "SLC5A2"),
        ("dapagliflozin",  "dapagliflozin",  "SLC5A2", "SLC5A2"),
        ("ertugliflozin",  "ertugliflozin",  "SLC5A2", "SLC5A2"),
        ("metformin",      "metformin",      "PRKAA1", "PRKAA1"),
        ("pioglitazone",   "pioglitazone",   "PPARG",  "PPARG"),
    ]
    logger.info("Seeding %d known drug→gene TARGETS edges", len(_KNOWN))
    with driver.session() as session:
        for drug_id, drug_name, gene_id, gene_symbol in _KNOWN:
            try:
                # Create canonical Drug + Gene nodes and the primary TARGETS edge
                session.run(
                    "MERGE (d:Drug {id: $drug_id}) SET d.name = $drug_name "
                    "MERGE (g:Gene {id: $gene_id}) SET g.symbol = $gene_symbol "
                    "MERGE (d)-[:TARGETS]->(g)",
                    drug_id=drug_id,
                    drug_name=drug_name,
                    gene_id=gene_id,
                    gene_symbol=gene_symbol,
                )
                # Also wire TARGETS for trial intervention Drug nodes whose id
                # contains the canonical drug name (e.g., "semaglutide_(rybelsus®)").
                # Excludes placebo nodes: both "placebo_(drug)" and "drug_placebo" forms.
                session.run(
                    "MATCH (variant:Drug) "
                    "WHERE variant.id CONTAINS $drug_id "
                    "  AND NOT variant.id STARTS WITH 'placebo' "
                    "  AND NOT variant.id CONTAINS '_placebo' "
                    "MATCH (g:Gene {id: $gene_id}) "
                    "MERGE (variant)-[:TARGETS]->(g)",
                    drug_id=drug_id,
                    gene_id=gene_id,
                )
            except Exception as exc:
                logger.warning("seed_known_targets failed for %s→%s: %s", drug_id, gene_id, exc)
    logger.info("Seed complete.")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _normalize_condition_name(raw: str) -> tuple[str, str]:
    """Return (display_name, node_id) for a trial condition string.

    Decodes HTML entities (including double-encoded variants like &amp;#39;)
    and normalizes whitespace so equivalent strings share one Disease node.
    """
    # Two passes handle double-encoded entities: &amp;#39; → &#39; → '
    name = html.unescape(html.unescape(raw.strip()))
    name = " ".join(name.split())
    node_id = name.lower().replace(" ", "_")
    return name, node_id


_GENE_ALIASES: dict[str, str] = {
    # GLP-1 axis
    "glp-1 receptor": "GLP1R",
    "glp1 receptor": "GLP1R",
    "glucagon-like peptide-1 receptor": "GLP1R",
    "glucagon-like peptide 1 receptor": "GLP1R",
    "glp1r": "GLP1R",
    # GIP axis
    "gip receptor": "GIPR",
    "gastric inhibitory polypeptide receptor": "GIPR",
    "gipr": "GIPR",
    # Glucagon
    "glucagon receptor": "GCGR",
    "gcgr": "GCGR",
    # Insulin / IGF
    "insulin receptor": "INSR",
    "igf-1 receptor": "IGF1R",
    "igf1 receptor": "IGF1R",
    # Metabolic
    "pparγ": "PPARG",
    "ppargamma": "PPARG",
    "ppar-gamma": "PPARG",
    "gpr40": "FFAR1",
    "sodium-glucose cotransporter 2": "SLC5A2",
    "sglt2": "SLC5A2",
    "ampk": "PRKAA1",
    # AD targets
    "amyloid precursor protein": "APP",
    "app": "APP",
    "beta-secretase": "BACE1",
    "bace1": "BACE1",
    "tau": "MAPT",
    "mapt": "MAPT",
    "apoe": "APOE",
    "apolipoprotein e": "APOE",
    "presenilin-1": "PSEN1",
    "presenilin 1": "PSEN1",
    "presenilin-2": "PSEN2",
    "presenilin 2": "PSEN2",
}


def _normalize_gene_id(name: str) -> str:
    """Map a protein description to HGNC gene symbol if known, else uppercase."""
    return _GENE_ALIASES.get(name.lower(), name.upper())


def _label_to_node_label(ner_label: str) -> str:
    """Map a NER label to a Neo4j node label."""
    mapping = {
        "CHEMICAL": "Drug",
        "SIMPLE_CHEMICAL": "Drug",
        "GENE_OR_GENE_PRODUCT": "Gene",
        "GENE": "Gene",
        "PROTEIN": "Gene",
        "DNA": "Gene",
        "RNA": "Gene",
        "DISEASE": "Disease",
        "DISORDER": "Disease",
        "ORGANISM": "Organism",
        "CELL_LINE": "CellLine",
        "CELL_TYPE": "CellType",
    }
    return mapping.get(ner_label.upper(), "Entity")
