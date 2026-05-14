# BOSC 2026 Abstract — Draft
**Track:** Abstracts - BOSC: Bioinformatics Open Source Conference
**Submitted by:** Leo (apolonio.leonardo@gmail.com)
**Status:** Draft — April 24, 2026

---

## Title
**ad-kg: An Open-Source Knowledge Graph Pipeline for Alzheimer's Disease Drug Repurposing**

---

## Abstract

Drug repurposing for Alzheimer's disease (AD) is hindered by the fragmentation of biomedical evidence across pharmacovigilance databases, clinical trial registries, genetic association catalogs, and the primary literature. We present **ad-kg**, a fully open-source Python pipeline that integrates four public data sources into a queryable Neo4j knowledge graph to surface drug repurposing candidates — with an initial focus on GLP-1 receptor agonists and metabolic pathway drugs.

The pipeline runs in five reproducible stages. The **ingest** stage fetches 873 biomedical abstracts (PubMed / NCBI E-utilities), 4,106 trial records (ClinicalTrials.gov API v2), 12,782 genetic associations (GWAS Catalog / EBI REST), and 119 adverse event reports across 4 subpopulation cohorts (FDA FAERS / OpenFDA). The **extract** stage applies scispaCy (`en_core_sci_lg` + UMLS linker) for named entity recognition followed by Claude-powered relation extraction with prompt caching, yielding 9,993 subject–predicate–object triples. The **resolve** stage embeds entities with `allenai/specter2` and canonicalizes synonyms via HDBSCAN (cosine threshold 0.88). The **load** stage merges nodes and edges idempotently into Neo4j 5. The **query** stage exposes 14 named Cypher patterns for drug discovery analysis.

The entire stack runs locally via Docker Compose with no proprietary dependencies beyond an optional NCBI API key. All data sources are openly licensed. The pipeline is tested, type-annotated, and designed for extension to other disease areas.

Applied to AD, the `whitespace_opportunity` query — drugs with a protective FAERS signal, GWAS genetic support, and no active AD trial — returns four GLP-1 agonists: liraglutide, exenatide, dulaglutide, and tirzepatide. Tirzepatide is the strongest candidate: it uniquely targets both GIP and GLP-1 receptors (GIPR + GLP1R), giving it 11 metabolic GWAS-linked SNPs versus 1 for the other three, and it is the only drug in the set with no registered Phase 2+ AD trial as of April 2026. The `triple_convergence` query — requiring concordant evidence across FAERS, GWAS, and literature simultaneously — confirms all four candidates, with tirzepatide again leading (11 GWAS SNPs, 11 AD-context papers). A pharmacovigilance sensitivity analysis across T2DM-enriched, elderly, and post-2020 FAERS cohorts finds that T2DM enrichment consistently strengthens the protective signal (liraglutide: overall ROR 0.26, T2DM ROR 0.15; canagliflozin: overall 0.21, T2DM 0.15), supporting a metabolic mechanism rather than a reporting artifact. Bridge gene analysis identifies APOE (combined AD × T2DM GWAS score 1.6×10⁻¹³⁷), TOMM40 (2×10⁻¹⁶⁵), and IGF1R (6×10⁻¹⁷) as the mechanistic spine connecting insulin resistance to neurodegeneration.

ad-kg is available at https://github.com/lapolonio/ad-kb under the MIT license.

---

## Keywords
knowledge graph, drug repurposing, Alzheimer's disease, pharmacovigilance, FAERS, GWAS, GLP-1, tirzepatide

---

## Word count: ~350 words

---

## Notes for revision
- BOSC 2026 deadline (April 9) has passed — **target BOSC 2027 or ISMB 2026 poster track**
- Consider trimming to 300 words if the venue requires it (cut the GWAS counts in the ingest sentence)
- The FAERS sensitivity analysis paragraph is the most novel addition vs. prior draft — keep it
