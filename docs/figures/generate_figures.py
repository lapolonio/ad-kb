"""Generate knowledge graph figures for ad-kg paper/abstract.

Run with: uv run python docs/figures/generate_figures.py
Outputs PNGs to docs/figures/.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from neo4j import GraphDatabase

OUT = Path(__file__).parent
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

DRUG_CLASS = {
    "tirzepatide":   "GLP-1/GIP",
    "liraglutide":   "GLP-1",
    "exenatide":     "GLP-1",
    "dulaglutide":   "GLP-1",
    "semaglutide":   "GLP-1",
    "canagliflozin": "SGLT2i",
    "empagliflozin": "SGLT2i",
    "dapagliflozin": "SGLT2i",
    "ertugliflozin": "SGLT2i",
    "pioglitazone":  "TZD",
    "metformin":     "Biguanide",
}

CLASS_COLOR = {
    "GLP-1/GIP": "#2563EB",  # blue
    "GLP-1":     "#60A5FA",  # light blue
    "SGLT2i":    "#16A34A",  # green
    "TZD":       "#D97706",  # amber
    "Biguanide": "#7C3AED",  # purple
    "other":     "#6B7280",  # gray
}


def connect():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ── Figure 1: Whitespace Opportunity Network ──────────────────────────────────

def fig_whitespace_network(driver):
    with driver.session() as s:
        rows = s.run("""
            MATCH (d:Drug)-[:PROTECTIVE_SIGNAL]->(f:FAERSReport)
            WHERE f.cohort = 'all' AND f.ror < 1.0 AND f.report_count >= 2
            WITH d,
                 round(min(f.ror), 3) AS best_ror,
                 round(min(f.ci_upper), 3) AS best_ci_upper
            MATCH (d)-[:TARGETS]->(g:Gene)<-[:LINKED_TO]-(s:SNP)
                  -[:ASSOCIATED_WITH]->(dis:Disease)
            WHERE dis.name CONTAINS 'Alzheimer'
               OR dis.name IN ['type 2 diabetes', 'insulin resistance',
                               'body mass index', 'fasting glucose']
            WITH DISTINCT d, g, dis, best_ror, best_ci_upper
            WHERE NOT EXISTS {
              MATCH (d)<-[:TESTS]-(t:Trial)-[:FOR]->(dis2:Disease)
              WHERE dis2.name CONTAINS 'Alzheimer'
                AND t.status IN ['RECRUITING','ACTIVE_NOT_RECRUITING','NOT_YET_RECRUITING']
            }
            RETURN d.name AS drug, g.symbol AS gene,
                   dis.name AS trait, best_ror, best_ci_upper
        """).data()

    G = nx.Graph()
    drug_set, gene_set, trait_set = set(), set(), set()
    edge_data: list[tuple] = []

    for row in rows:
        drug = row["drug"].lower()
        gene = row["gene"]
        trait = row["trait"]
        drug_set.add(drug)
        gene_set.add(gene)
        trait_set.add(trait)
        edge_data.append((drug, gene, row["best_ror"], row["best_ci_upper"]))
        G.add_edge(drug, gene)
        G.add_edge(gene, trait)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Layered layout: drugs left, genes centre, traits right
    pos = {}
    drugs  = sorted(drug_set)
    genes  = sorted(gene_set)
    traits = sorted(trait_set)
    for i, d in enumerate(drugs):
        pos[d] = (-2, i - len(drugs) / 2)
    for i, g in enumerate(genes):
        pos[g] = (0, i - len(genes) / 2)
    for i, t in enumerate(traits):
        pos[t] = (2.5, i - len(traits) / 2)

    # Draw edges
    for d, g in [(r["drug"].lower(), r["gene"]) for r in rows]:
        sig = any(e[0] == d and e[1] == g and e[3] < 1.0 for e in edge_data)
        nx.draw_networkx_edges(G, pos, edgelist=[(d, g)], ax=ax,
                               edge_color="#2563EB" if sig else "#93C5FD",
                               width=2.0 if sig else 1.0, alpha=0.7)
    for r in rows:
        g, t = r["gene"], r["trait"]
        nx.draw_networkx_edges(G, pos, edgelist=[(g, t)], ax=ax,
                               edge_color="#6B7280", width=1.0, alpha=0.5,
                               style="dashed")

    # Drug nodes
    for d in drugs:
        cls = DRUG_CLASS.get(d, "other")
        col = CLASS_COLOR.get(cls, CLASS_COLOR["other"])
        nx.draw_networkx_nodes(G, pos, nodelist=[d], ax=ax,
                               node_color=col, node_size=1200, node_shape="o")
    # Gene nodes
    nx.draw_networkx_nodes(G, pos, nodelist=list(genes), ax=ax,
                           node_color="#F59E0B", node_size=900, node_shape="s")
    # Trait nodes
    nx.draw_networkx_nodes(G, pos, nodelist=list(traits), ax=ax,
                           node_color="#E5E7EB", node_size=700, node_shape="D",
                           edgecolors="#6B7280", linewidths=1)

    labels = {n: n if n in gene_set | trait_set else n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight="bold")

    # Legend
    legend_handles = [
        mpatches.Patch(color=CLASS_COLOR["GLP-1/GIP"], label="GLP-1/GIP agonist"),
        mpatches.Patch(color=CLASS_COLOR["GLP-1"],     label="GLP-1 agonist"),
        mpatches.Patch(color=CLASS_COLOR["SGLT2i"],    label="SGLT2 inhibitor"),
        mpatches.Patch(color=CLASS_COLOR["TZD"],       label="TZD (PPARG agonist)"),
        mpatches.Patch(color="#F59E0B",                label="Bridge gene"),
        mpatches.Patch(color="#E5E7EB", ec="#6B7280",  label="GWAS trait"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    ax.set_title("Whitespace Opportunity: Drug → Bridge Gene → GWAS Trait\n"
                 "(solid blue = statistically significant FAERS protective signal)",
                 fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    out = OUT / "fig1_whitespace_network.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Figure 2: FAERS ROR Forest Plot ──────────────────────────────────────────

def fig_faers_forest(driver):
    with driver.session() as s:
        all_rows = s.run("""
            MATCH (d:Drug)-[:PROTECTIVE_SIGNAL]->(f:FAERSReport)
            WHERE f.cohort = 'all' AND f.report_count >= 2
            RETURN d.name AS drug,
                   f.ror AS ror, f.ci_lower AS ci_lower,
                   f.ci_upper AS ci_upper,
                   f.reaction AS reaction, f.report_count AS n
        """).data()

    # Pick the best (lowest ROR) signal per drug
    best: dict[str, dict] = {}
    for r in all_rows:
        drug = r["drug"].lower()
        if drug not in best or r["ror"] < best[drug]["ror"]:
            best[drug] = r

    rows = sorted(best.values(), key=lambda r: r["ror"])
    drugs  = [r["drug"].lower() for r in rows]
    rors   = [r["ror"]      for r in rows]
    lowers = [r["ci_lower"] for r in rows]
    uppers = [r["ci_upper"] for r in rows]
    ns     = [r["n"]        for r in rows]
    rxns   = [r["reaction"] for r in rows]

    y = list(range(len(drugs)))
    xerr_lo = [ror - lo for ror, lo in zip(rors, lowers)]
    xerr_hi = [hi - ror for ror, hi in zip(rors, uppers)]

    fig, ax = plt.subplots(figsize=(10, max(5, len(drugs) * 0.6 + 1)))

    for i, (ror, lo, hi, n, rxn, drug) in enumerate(
            zip(rors, lowers, uppers, ns, rxns, drugs)):
        sig = hi < 1.0
        cls = DRUG_CLASS.get(drug, "other")
        col = CLASS_COLOR.get(cls, CLASS_COLOR["other"])
        ax.errorbar(ror, i, xerr=[[ror - lo], [hi - ror]],
                    fmt="o", color=col, ecolor=col,
                    elinewidth=2 if sig else 1,
                    capsize=4, markersize=8 if sig else 6,
                    alpha=1.0 if sig else 0.6)
        ax.text(max(hi, 1.05) + 0.05, i,
                f"n={n}, {rxn}", va="center", fontsize=7, color="#374151")

    ax.axvline(1.0, color="black", linewidth=1.2, linestyle="--", label="ROR = 1 (null)")
    ax.set_yticks(y)
    ax.set_yticklabels([d.capitalize() for d in drugs], fontsize=9)
    ax.set_xlabel("Reporting Odds Ratio (ROR) with 95% CI", fontsize=10)
    ax.set_title("FAERS Protective Signals — Best Overall-Cohort Signal per Drug\n"
                 "(filled = CI entirely below 1.0; faded = CI crosses 1.0)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(left=0)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    legend_handles = [
        mpatches.Patch(color=CLASS_COLOR["GLP-1/GIP"], label="GLP-1/GIP"),
        mpatches.Patch(color=CLASS_COLOR["GLP-1"],     label="GLP-1"),
        mpatches.Patch(color=CLASS_COLOR["SGLT2i"],    label="SGLT2i"),
        mpatches.Patch(color=CLASS_COLOR["TZD"],       label="TZD"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    plt.tight_layout()
    out = OUT / "fig2_faers_forest.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Figure 3: Triple Convergence Bubble Chart ─────────────────────────────────

def fig_triple_convergence(driver):
    with driver.session() as s:
        rows = s.run("""
            MATCH (d:Drug)-[:PROTECTIVE_SIGNAL]->(f:FAERSReport)
            WHERE f.cohort = 'all' AND f.ror < 1.0 AND f.report_count >= 2
            WITH d,
                 count { (d)-[:PROTECTIVE_SIGNAL]->(fp:FAERSReport)
                         WHERE fp.cohort='all' AND fp.ci_upper < 1.0
                           AND fp.report_count >= 10 } AS sig_protective,
                 count { (d)-[:ADVERSE_SIGNAL]->(fa:FAERSReport)
                         WHERE fa.cohort='all' AND fa.ci_lower > 1.0
                           AND fa.report_count >= 10 } AS sig_adverse
            WHERE sig_adverse <= sig_protective
            MATCH (d)-[:TARGETS|RELATED_TO*1..2]->(g:Gene)<-[:LINKED_TO]-(s:SNP)
                  -[:ASSOCIATED_WITH]->(dis:Disease)
            WHERE dis.name CONTAINS 'Alzheimer'
               OR dis.name IN ['type 2 diabetes','insulin resistance',
                               'body mass index','glycated hemoglobin',
                               'fasting glucose','obesity']
            WITH d, count(DISTINCT s) AS gwas_snps
            MATCH (d)-[:MENTIONS|RELATED_TO*1..2]-(p:Paper)
            WITH d, gwas_snps, count(DISTINCT p) AS lit_count
            WHERE lit_count >= 1
            MATCH (d)-[:PROTECTIVE_SIGNAL]->(f2:FAERSReport)
            WHERE f2.cohort='all' AND f2.report_count >= 2
            RETURN d.name AS drug, gwas_snps, lit_count,
                   round(min(f2.ror), 3) AS best_ror
            ORDER BY gwas_snps DESC, lit_count DESC
        """).data()

    fig, ax = plt.subplots(figsize=(9, 7))

    for row in rows:
        drug = row["drug"].lower()
        cls  = DRUG_CLASS.get(drug, "other")
        col  = CLASS_COLOR.get(cls, CLASS_COLOR["other"])
        size = max(100, (1.0 - row["best_ror"]) * 2000)  # bigger = better ROR
        ax.scatter(row["gwas_snps"], row["lit_count"],
                   s=size, color=col, alpha=0.75, edgecolors="white", linewidth=1.5)
        ax.annotate(drug.capitalize(),
                    (row["gwas_snps"], row["lit_count"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, fontweight="bold")

    ax.set_xlabel("GWAS Bridge SNPs (AD + metabolic traits)", fontsize=10)
    ax.set_ylabel("AD-context Literature Papers (2-hop)", fontsize=10)
    ax.set_title("Triple Convergence: FAERS × GWAS × Literature\n"
                 "(bubble size ∝ pharmacovigilance strength; 1 − best ROR)",
                 fontsize=11, fontweight="bold")
    ax.set_xscale("symlog", linthresh=1)
    ax.grid(alpha=0.3)

    legend_handles = [
        mpatches.Patch(color=CLASS_COLOR["GLP-1/GIP"], label="GLP-1/GIP"),
        mpatches.Patch(color=CLASS_COLOR["GLP-1"],     label="GLP-1"),
        mpatches.Patch(color=CLASS_COLOR["SGLT2i"],    label="SGLT2i"),
        mpatches.Patch(color=CLASS_COLOR["TZD"],       label="TZD"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    plt.tight_layout()
    out = OUT / "fig3_triple_convergence.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Figure 4: Bridge Genes Dual-Signal Scatter ────────────────────────────────

def fig_bridge_genes(driver):
    with driver.session() as s:
        rows = s.run("""
            MATCH (g:Gene)<-[:LINKED_TO]-(s_ad:SNP)-[r_ad:ASSOCIATED_WITH]->(d_ad:Disease)
            WHERE d_ad.name CONTAINS 'Alzheimer' AND g.symbol <> 'NR'
            WITH g, min(toFloat(r_ad.p_value)) AS ad_pval
            MATCH (g)<-[:LINKED_TO]-(s_met:SNP)-[r_met:ASSOCIATED_WITH]->(d_met:Disease)
            WHERE d_met.name IN ['type 2 diabetes','insulin resistance',
                                 'body mass index','fasting glucose']
            WITH g, ad_pval, min(toFloat(r_met.p_value)) AS met_pval
            RETURN g.symbol AS gene, ad_pval, met_pval,
                   ad_pval * met_pval AS combined
            ORDER BY combined ASC
            LIMIT 20
        """).data()

    genes    = [r["gene"]     for r in rows]
    ad_pvals = [r["ad_pval"]  for r in rows]
    me_pvals = [r["met_pval"] for r in rows]

    # safe log10
    log_ad = [-math.log10(max(p, 1e-320)) for p in ad_pvals]
    log_me = [-math.log10(max(p, 1e-320)) for p in me_pvals]

    # Clip extreme values so the plot isn't dominated by TOMM40/APOE
    MAX_LOG = 100
    log_ad_clip = [min(v, MAX_LOG) for v in log_ad]
    log_me_clip = [min(v, MAX_LOG) for v in log_me]

    fig, ax = plt.subplots(figsize=(9, 7))

    sizes = [120 if (a >= 7.3 and m >= 7.3) else 60
             for a, m in zip(log_ad, log_me)]
    ax.scatter(log_me_clip, log_ad_clip, s=sizes, color="#6366F1", alpha=0.75,
               edgecolors="white", linewidth=0.8)

    for g, x, y, xa, ya in zip(genes, log_me_clip, log_ad_clip, log_me, log_ad):
        label = g
        # Mark clipped values
        if xa > MAX_LOG:
            label = f"{g} (>{MAX_LOG})"
        elif ya > MAX_LOG:
            label = f"{g} (>{MAX_LOG})"
        ax.annotate(label, (x, y), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, fontweight="bold")

    gwas_sig = -math.log10(5e-8)  # ≈ 7.3
    ax.axhline(gwas_sig, color="#EF4444", linestyle="--", linewidth=1,
               label="GWAS significance (p=5×10⁻⁸)")
    ax.axvline(gwas_sig, color="#EF4444", linestyle="--", linewidth=1)

    ax.set_xlabel("−log₁₀(best metabolic GWAS p-value)", fontsize=10)
    ax.set_ylabel("−log₁₀(best Alzheimer's GWAS p-value)", fontsize=10)
    ax.set_title("Bridge Genes: Dual GWAS Signal (AD × Metabolic Traits)\n"
                 f"Top 20 by combined p-value product (axes capped at {MAX_LOG})",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, MAX_LOG + 5)
    ax.set_ylim(0, MAX_LOG + 5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT / "fig4_bridge_genes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Figure 5: FAERS Subpopulation Heatmap ────────────────────────────────────

def fig_subpopulation_heatmap(driver):
    with driver.session() as s:
        rows = s.run("""
            MATCH (d:Drug)-[:PROTECTIVE_SIGNAL]->(f_all:FAERSReport)
            WHERE f_all.cohort = 'all' AND f_all.report_count >= 2
            WITH DISTINCT d, round(min(f_all.ror), 3) AS overall_ror
            OPTIONAL MATCH (d)-[:PROTECTIVE_SIGNAL|ADVERSE_SIGNAL]->(f:FAERSReport)
            WHERE f.cohort IN ['t2dm','elderly','post_2020']
              AND f.report_count >= 2
            RETURN d.name AS drug, overall_ror,
                   f.cohort AS cohort, round(min(f.ror), 3) AS cohort_ror
            ORDER BY overall_ror ASC
        """).data()

    from collections import defaultdict
    drug_overall: dict[str, float] = {}
    drug_cohort:  dict[str, dict[str, float]] = defaultdict(dict)

    for r in rows:
        drug = r["drug"].lower()
        drug_overall[drug] = r["overall_ror"]
        if r["cohort"] and r["cohort_ror"] is not None:
            drug_cohort[drug][r["cohort"]] = r["cohort_ror"]

    drugs   = sorted(drug_overall, key=lambda d: drug_overall[d])
    cohorts = ["all", "t2dm", "elderly", "post_2020"]
    labels  = ["Overall", "T2DM", "Elderly", "Post-2020"]

    matrix = np.full((len(drugs), len(cohorts)), np.nan)
    for i, drug in enumerate(drugs):
        matrix[i, 0] = drug_overall[drug]
        for j, coh in enumerate(cohorts[1:], 1):
            if coh in drug_cohort[drug]:
                matrix[i, j] = drug_cohort[drug][coh]

    fig, ax = plt.subplots(figsize=(9, max(5, len(drugs) * 0.55 + 1.5)))
    # Diverging colormap centred at 1.0
    vmax = 2.0
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    for i in range(len(drugs)):
        for j in range(len(cohorts)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if 0.4 < val < 1.6 else "white",
                        fontweight="bold" if val < 1.0 else "normal")
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="#9CA3AF")

    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(drugs)))
    ax.set_yticklabels([d.capitalize() for d in drugs], fontsize=9)
    ax.set_title("FAERS ROR Sensitivity Across Subpopulations\n"
                 "(green < 1.0 = protective; red > 1.0 = adverse; — = no data)",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Reporting Odds Ratio (ROR)", shrink=0.6)
    plt.tight_layout()
    out = OUT / "fig5_subpopulation_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    driver = connect()
    print("Generating figures...")
    fig_whitespace_network(driver)
    fig_faers_forest(driver)
    fig_triple_convergence(driver)
    fig_bridge_genes(driver)
    fig_subpopulation_heatmap(driver)
    driver.close()
    print("Done.")
