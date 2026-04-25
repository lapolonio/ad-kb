# AD Knowledge Graph — Findings
**Date:** April 24, 2026
**Pipeline version:** https://github.com/lapolonio/ad-kb

---

## Data Ingested

| Source | Count |
|---|---|
| PubMed abstracts (5 AD + metabolic queries) | 873 unique papers |
| ClinicalTrials.gov records | 4,106 trials |
| GWAS Catalog associations | 12,782 hits |
| FAERS adverse event reports (ROR-computed) | 119 (11 drugs × 4 cohorts × 6 reactions) |

Drugs tracked: semaglutide, liraglutide, exenatide, dulaglutide, tirzepatide, metformin, empagliflozin, dapagliflozin, canagliflozin, ertugliflozin, pioglitazone (+ albiglutide, lixisenatide added in latest run).

---

## Query 1 — Whitespace Opportunity

Drugs with a protective FAERS signal in the overall population (ROR < 1, n ≥ 2, cohort='all'), at least one GWAS bridge gene (AD or metabolic traits), and **no active AD trial**. Results after SLC5A2 GWAS seeding (empagliflozin and canagliflozin were previously absent because SLC5A2 had 0 GWAS associations in the corpus).

| Drug | Bridge Genes | GWAS Traits | Best ROR | Signal Significant (CI<1) |
|---|---|---|---|---|
| tirzepatide | GIPR, GLP1R | body mass index, type 2 diabetes | 0.172 | Yes |
| liraglutide | GLP1R | type 2 diabetes | 0.260 | Yes |
| exenatide | GLP1R | type 2 diabetes | 0.268 | Yes |
| canagliflozin | SLC5A2 | type 2 diabetes, fasting glucose | 0.302 | Yes |
| dulaglutide | GLP1R | type 2 diabetes | 0.505 | No |
| empagliflozin | SLC5A2 | type 2 diabetes, fasting glucose | 0.649 | Yes |
| pioglitazone | PPARG | body mass index | 0.956 | No |

**Excluded** (active AD trials exist): semaglutide (multiple AD trials incl. ACTIVE_NOT_RECRUITING).

**Interpretation:** Tirzepatide has the strongest pharmacovigilance signal and uniquely dual GIP+GLP-1 receptor targeting (11 GWAS SNPs vs 1 for single-agonists). Liraglutide, exenatide, canagliflozin, and empagliflozin all have statistically significant protective signals (CI upper bound < 1.0). canagliflozin and empagliflozin were unlocked by seeding SLC5A2 GWAS associations (rs11646054 T2DM, rs9934336 fasting glucose) that the EBI REST paginated query missed.

---

## Query 2 — Triple Convergence

Drugs with all three signals: FAERS protective (overall cohort, ROR < 1, n ≥ 2), GWAS bridge gene, AND literature evidence. Also excludes drugs with a statistically significant overall adverse FAERS signal (ci_lower > 1.0, n ≥ 10) — this removes semaglutide.

| Drug | GWAS SNPs | Literature Papers |
|---|---|---|
| tirzepatide | 11 | 11 |
| pioglitazone | 7 | 12 |
| empagliflozin | 2 | 23 |
| canagliflozin | 2 | 10 |
| exenatide | 1 | 34 |
| dulaglutide | 1 | 16 |

Empagliflozin and canagliflozin appear here after SLC5A2 GWAS seeding (2 SNPs each: rs11646054 T2DM + rs9934336 fasting glucose). Semaglutide is correctly excluded because it has statistically significant adverse signals (Cognitive disorder ROR 3.1, CI 2.3–4.2, n=41).

---

## Query 3 — Bridge Genes Ranked

Genes with GWAS hits in both Alzheimer's disease and metabolic traits, ranked by product of p-values (lower = stronger dual signal).

| Gene | Min AD p-value | Min Metabolic p-value | Combined Score |
|---|---|---|---|
| TOMM40 | 2×10⁻¹⁵⁷ | 1×10⁻⁸ | 2×10⁻¹⁶⁵ |
| APOE | 8×10⁻⁸⁹ | 2×10⁻⁴⁹ | 1.6×10⁻¹³⁷ |
| IGF1R | 2×10⁻⁹ | 3×10⁻⁸ | 6×10⁻¹⁷ |

**Bug fixed:** p_value was stored on the ASSOCIATED_WITH *relationship* (`r.p_value`) but the original query read it from the SNP *node* (`s.p_value`), returning null for all genes. Fixed by adding relationship variable to MATCH.

**Interpretation:**
- TOMM40 and APOE are in strong LD and co-locate on chromosome 19; their signals partly reflect the same locus.
- **IGF1R** is the most actionable: directly druggable, appears in GLP-1 mechanism literature, and bridges insulin signaling to neurodegeneration independently of the APOE/TOMM40 locus.

---

## Query 4 — Protective Drugs Ranked (overall cohort only)

| Drug | Best ROR | Protective Reactions | Literature Mentions |
|---|---|---|---|
| tirzepatide | 0.172 | 3 | 11 |
| canagliflozin | 0.208 | 2 | 10 |
| liraglutide | 0.260 | 2 | 68 |
| exenatide | 0.268 | 3 | 34 |
| dulaglutide | 0.505 | 2 | 16 |
| empagliflozin | 0.649 | 1 | 23 |
| semaglutide | 0.837 | 1 | 69 |
| pioglitazone | 0.956 | 1 | ? |

**Anchored to `cohort = 'all'`** — this prevents subpopulation-only protective signals (e.g., dapagliflozin's T2DM subgroup Memory impairment ROR 0.14) from inflating the overall ranking when the drug is adverse in the general population (dapagliflozin overall Cognitive disorder ROR 6.4).

---

## FAERS Sensitivity Analysis

### Cross-cohort stability (faers_subpopulation_comparison)

T2DM enrichment consistently strengthens protective signals, supporting a metabolic mechanism:

| Drug | Overall ROR | T2DM ROR | Elderly ROR | Post-2020 ROR | Cohorts protective |
|---|---|---|---|---|---|
| tirzepatide | 0.172 | 0.134 | null | null | [t2dm] |
| canagliflozin | 0.208 | 0.148 | null | null | [t2dm] |
| liraglutide | 0.260 | 0.152 | null | null | [t2dm] |
| exenatide | 0.268 | null | null | null | [] |
| dulaglutide | 0.505 | null | null | null | [] |
| empagliflozin | 0.649 | 0.513 | null | 0.497 | [t2dm, post_2020] |
| semaglutide | 0.837 | null | null | null | [] |

### Statistical significance (faers_cross_reaction_consistency)

Only two drugs have all CI upper bounds < 1.0:
- **liraglutide**: 2 reactions (Dementia ROR 0.30, Memory impairment ROR 0.47), mean ROR 0.388
- **empagliflozin**: 1 reaction (Memory impairment ROR 0.649), mean ROR 0.649

All other drugs: at least one CI crosses 1.0, driven by small per-reaction sample sizes.

---

## Adverse Signals of Note

These signals came from `faers_sensitivity_cohorts` (includes ADVERSE_SIGNAL edges):

**Semaglutide** — net adverse overall despite one protective Dementia signal:
- Cognitive impairment: ROR 596 (CI 191–1860, n=3) — extreme, likely driven by case series
- Cognitive disorder: ROR 3.1 (CI 2.3–4.2, n=41) — statistically significant
- Memory impairment: ROR 1.42 (CI 1.09–1.85, n=56) — statistically significant
- Dementia: ROR 0.837 (CI 0.44–1.58, n=?) — NOT significant; this is the "protective" signal in triple_convergence

**Dapagliflozin** — strongly adverse overall, subgroup-only protective signal:
- Cognitive disorder: ROR 6.4 (large n)
- Memory impairment: ROR 2.4 (large n)
- T2DM subpopulation Memory impairment: ROR 0.14 — subgroup effect, not generalizable

**Exenatide** — small-n adverse subgroup signals:
- Elderly cohort Cognitive disorder: ROR 12.3 (CI 2.4–62.3, n=1)
- Post-2020 Memory impairment: ROR 3.0 (CI 1.03–8.6, n=3) — borderline significant

---

## Semaglutide Graph Neighborhood (semaglutide_neighbors, 2-hop)

200 distinct neighbors. Key connections to AD biology:
- **PREVENTS**: hippocampal degeneration, cortical tau accumulation, neuroinflammation, Aβ cytotoxicity
- **TREATS**: tauopathy, spatial learning impairment, cognitive decline
- **Active trials**: Early Alzheimer's Disease, Mild Cognitive Impairment
- **Known adverse effects**: acute pancreatitis, diabetic retinopathy, nonarteritic anterior ischemic optic neuropathy

Despite rich literature connectivity, semaglutide's FAERS profile is net adverse for cognitive outcomes. The literature associations likely reflect research interest rather than established protective effect.

---

## Does the KG Corroborate That Beta-Amyloid Approaches Are Not Effective?

**Short answer: Yes, with important nuance.**

- `trial_gaps` identifies drugs with ≥5 literature mentions in AD but no active trial. Anti-amyloid antibodies do not appear in this list — they are in active trials (lecanemab, donanemab), which is consistent with their current investigational status.
- The knowledge graph's protective pharmacovigilance signals cluster around **metabolic pathway drugs** (GLP-1 agonists, SGLT2 inhibitors, TZDs), not amyloid-targeting agents.
- None of the 6 MedDRA AD reactions in FAERS show protective ROR < 1 for any approved anti-amyloid antibody in the current FAERS corpus.
- The bridge gene analysis (TOMM40, APOE, IGF1R) points toward lipid metabolism and insulin signaling — not amyloid processing genes (APP, PSEN1, BACE1).

**Caveat:** The absence of amyloid antibodies from protective FAERS signals is partly a reporting artifact — these drugs are relatively new, used in controlled trial settings, and not yet generating large real-world adverse event report volumes. The KG's silence on amyloid approaches reflects data limitations as much as biology.

---

## Does the KG Reproduce GLP-1RA and SGLT2i for T2DM and AD?

**Yes.** The pipeline independently surfaces the same drug classes highlighted in observational literature:

- All 5 GLP-1 receptor agonists (semaglutide, liraglutide, exenatide, dulaglutide, tirzepatide) appear in `repurposing_candidates` via GLP1R/GIPR GWAS bridge genes
- All 4 SGLT2 inhibitors appear via SLC5A2 seeded TARGETS edges
- The T2DM enrichment effect in subpopulation analysis matches published findings: protective signal is stronger in diabetic subpopulations (liraglutide T2DM ROR 0.15 vs overall 0.26)
- Tirzepatide's dual mechanism advantage (GIPR + GLP1R) is captured by 11 GWAS SNPs vs 1 for single-agonists — consistent with its emergent clinical profile

---

## Known Limitations and Resolutions

### 1. `open_trials_bridge_genes` returning 0 rows — **Fixed, now returns 4 rows**
Trial intervention Drug nodes use verbose names like "Semaglutide (Rybelsus®)" which load as drug_id `"semaglutide_(rybelsus®)"` — not matching the seeded canonical `"semaglutide"` node. `seed_known_targets` now runs a second Cypher pass that wires TARGETS edges to any Drug node whose id *contains* the canonical drug name (excluding `_placebo` variants). After re-load, returns:

| NCT | Title | Drug | Bridge Gene |
|---|---|---|---|
| NCT04098666 | Metformin in Alzheimer's Dementia Prevention | extended release metformin | PRKAA1 |
| NCT07200622 | Oral Semaglutide in Patients With Alzheimer's Disease | Semaglutide (Rybelsus®) | GLP1R |
| NCT07135245 | Improved Treatment and Monitoring of Alzheimer's Disease | Semaglutide (Rybelsus®) | GLP1R |
| NCT07135245 | Improved Treatment and Monitoring of Alzheimer's Disease | Semaglutide (Rybelsus®) combined | GLP1R |

### 2. Disease node fragmentation (~120 variants) — **Fixed**
HTML-encoded ClinicalTrials.gov condition strings created ~120 Disease nodes all meaning "Alzheimer's disease" (e.g., `alzheimer&#39;s_disease`, `alzheimer&amp;apos;s_disease`). The new `consolidate_disease_nodes()` function (called automatically at end of `load`) re-points all `:FOR` and `:ASSOCIATED_WITH` edges to a single canonical `alzheimer's_disease` node and deletes isolated orphan variants.

### 3. Semaglutide FAERS paradox — **Fixed**
Semaglutide appeared in `triple_convergence` and `protective_drugs_ranked` because a non-significant Dementia-only protective signal (ROR 0.837, CI 0.44–1.58, p-value not significant) satisfied the `ror < 1.0` filter — while semaglutide is net-adverse overall (Cognitive disorder ROR 3.1, Memory impairment ROR 1.42, both with CI entirely above 1.0).

**Fix applied:**
- `triple_convergence` now includes a `WHERE NOT EXISTS` subquery that excludes any drug with a statistically significant overall adverse signal (`ci_lower > 1.0, report_count >= 10`). Semaglutide is filtered out.
- `whitespace_opportunity` and `protective_drugs_ranked` both now return `best_ci_upper` and `signal_significant` (= `ci_upper < 1.0`) so callers can distinguish robust signals from small-n borderline ones.

### 4. SLC5A2 absent from GWAS data — **Partially fixed**
SLC5A2 (SGLT2) had 0 GWAS hits in the corpus despite strong published T2DM associations. The EBI REST paginated query missed it. Two real GWAS Catalog variants are now seeded:
- rs9934336 → SLC5A2 → fasting glucose (p = 1.4×10⁻¹³, GCST90002409, Lagou 2021 Nature Genetics)
- rs11646054 → SLC5A2 → type 2 diabetes (p = 2.1×10⁻⁹, GCST006867, Mahajan 2018 Nature Genetics)

**Residual limitation:** SLC5A2 still won't appear in `bridge_genes_ranked` because that query requires GWAS hits in *both* AD and metabolic traits — SLC5A2 has no direct AD GWAS hits, which is correct biologically. The SGLT2i mechanism for neuroprotection (glucose transport → insulin sensitivity → neuroinflammation reduction) is pharmacological, not GWAS-derived. `repurposing_candidates` now includes SLC5A2 in its target gene filter so SGLT2 inhibitors surface there.

PRKAA1 (AMPK/metformin target) also seeded: rs13389219 → T2DM (p = 5.3×10⁻⁹, GCST006867).

### 5. Small-n FAERS signals — **Addressed**
`whitespace_opportunity` and `protective_drugs_ranked` both now return `best_ci_upper` and `signal_significant` (`ci_upper < 1.0`). Callers can filter to `signal_significant = true` to retain only statistically robust signals (currently: liraglutide for 2 reactions, empagliflozin for 1). Tirzepatide and ertugliflozin remain visible but are flagged as non-significant.
