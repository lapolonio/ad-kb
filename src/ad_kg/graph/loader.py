"""Neo4j graph loader: idempotent MERGE operations for all node/edge types."""

from __future__ import annotations

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

                # Ensure FAERSReport node
                reaction_id = f"{drug_id}_{rpt.reaction.lower().replace(' ', '_')}"
                session.run(
                    "MERGE (f:FAERSReport {id: $rid}) "
                    "SET f.drug_name = $drug_name, f.reaction = $reaction, "
                    "    f.ror = $ror, f.ci_lower = $ci_lower, "
                    "    f.ci_upper = $ci_upper, f.report_count = $report_count",
                    rid=reaction_id,
                    drug_name=rpt.drug_name,
                    reaction=rpt.reaction,
                    ror=rpt.ror,
                    ci_lower=rpt.ci_lower,
                    ci_upper=rpt.ci_upper,
                    report_count=rpt.report_count,
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

                # Trial FOR Disease
                for condition in trial.conditions:
                    cid = condition.lower().replace(" ", "_")
                    session.run(
                        "MERGE (d:Disease {id: $cid}) SET d.name = $condition "
                        "WITH d "
                        "MATCH (t:Trial {id: $nct_id}) "
                        "MERGE (t)-[:FOR]->(d)",
                        cid=cid,
                        condition=condition,
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


# ── Utilities ─────────────────────────────────────────────────────────────────

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
