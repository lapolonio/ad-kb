"""GWAS Catalog ingest and ChEMBL drug lookup."""

from __future__ import annotations

import logging
import math
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ad_kg.config import GWAS_P_VALUE_THRESHOLD
from ad_kg.models import GWASHit

logger = logging.getLogger(__name__)

_GWAS_BASE = "https://www.ebi.ac.uk/gwas/rest/api"
_CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

_DEFAULT_TRAITS = [
    "Alzheimer's disease",
    "type 2 diabetes",
    "insulin resistance",
    "body mass index",
]


@retry(
    retry=retry_if_exception_type((requests.RequestException, IOError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _gwas_get(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


@retry(
    retry=retry_if_exception_type((requests.RequestException, IOError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _chembl_get(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_association(
    assoc: dict[str, Any], trait: str, study_id: str
) -> GWASHit | None:
    """Parse a GWAS Catalog association into a GWASHit."""
    try:
        # API returns mantissa + exponent separately, not a combined pvalue field
        mantissa = assoc.get("pvalueMantissa")
        exponent = assoc.get("pvalueExponent")
        if mantissa is None or exponent is None:
            return None
        p_value = float(mantissa) * (10 ** float(exponent))
        if p_value >= GWAS_P_VALUE_THRESHOLD:
            return None

        loci = assoc.get("loci", [])
        snp_id = ""
        gene = ""
        for locus in loci:
            for sv in locus.get("strongestRiskAlleles", []):
                snp_id = sv.get("riskAlleleName", "").split("-")[0]
            for author_gene in locus.get("authorReportedGenes", []):
                gene = author_gene.get("geneName", "")

        if not snp_id:
            return None

        or_val = 1.0
        or_raw = assoc.get("orPerCopyNum")
        beta_raw = assoc.get("betaNum")
        if or_raw is not None:
            try:
                or_val = float(or_raw)
            except (TypeError, ValueError):
                or_val = 1.0
        elif beta_raw is not None:
            try:
                or_val = math.exp(float(beta_raw))
            except (TypeError, ValueError):
                or_val = 1.0

        return GWASHit(
            snp_id=snp_id,
            gene=gene,
            trait=trait,
            p_value=p_value,
            odds_ratio=or_val,
            study_id=study_id,
        )
    except Exception as exc:
        logger.debug("Failed to parse GWAS association: %s", exc)
        return None


def _fetch_study_accessions_for_trait(trait: str) -> list[str]:
    """Return GWAS Catalog study accession IDs for a disease trait name."""
    url = f"{_GWAS_BASE}/studies/search/findByDiseaseTrait"
    page, page_size = 0, 100
    accessions: list[str] = []

    while True:
        try:
            data = _gwas_get(url, {"diseaseTrait": trait, "page": page, "size": page_size})
        except Exception as exc:
            logger.error("GWAS studies search error for trait %r: %s", trait, exc)
            break

        for study in data.get("_embedded", {}).get("studies", []):
            acc = study.get("accessionId")
            if acc:
                accessions.append(acc)

        page_info = data.get("page", {})
        if page >= page_info.get("totalPages", 1) - 1:
            break
        page += 1

    return accessions


def _fetch_associations_for_study(accession_id: str) -> list[tuple[dict[str, Any], str]]:
    """Fetch all associations for one study, returning (assoc, study_id) pairs."""
    url = f"{_GWAS_BASE}/associations/search/findByStudyAccessionId"
    page, page_size = 0, 200
    results: list[tuple[dict[str, Any], str]] = []

    while True:
        try:
            data = _gwas_get(url, {"accessionId": accession_id, "page": page, "size": page_size})
        except Exception as exc:
            logger.warning("GWAS associations fetch failed for %s: %s", accession_id, exc)
            break

        for assoc in data.get("_embedded", {}).get("associations", []):
            results.append((assoc, accession_id))

        page_info = data.get("page", {})
        if page >= page_info.get("totalPages", 1) - 1:
            break
        page += 1

    return results


def ingest_gwas(
    traits: list[str] | None = None,
    limit: int | None = None,
) -> list[GWASHit]:
    """Ingest GWAS Catalog hits for specified traits.

    Args:
        traits: List of trait names. Defaults to AD + metabolic traits.
        limit: Maximum total hits to return.

    Returns:
        List of GWASHit objects passing p-value threshold.
    """
    if traits is None:
        traits = _DEFAULT_TRAITS

    seen: set[tuple[str, str]] = set()
    hits: list[GWASHit] = []

    for trait in traits:
        logger.info("Fetching GWAS studies for trait=%r", trait)
        accessions = _fetch_study_accessions_for_trait(trait)
        logger.info("  Found %d studies", len(accessions))

        for accession_id in accessions:
            assoc_pairs = _fetch_associations_for_study(accession_id)
            for assoc, study_id in assoc_pairs:
                hit = _parse_association(assoc, trait, study_id)
                if hit is None:
                    continue
                key = (hit.snp_id, hit.trait)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(hit)
                if limit and len(hits) >= limit:
                    logger.info("Reached limit of %d GWAS hits", limit)
                    return hits

    logger.info("Total GWAS hits: %d", len(hits))
    return hits


def fetch_drugs_for_gene(gene_symbol: str) -> list[str]:
    """Use ChEMBL to find approved drugs targeting a gene.

    Args:
        gene_symbol: HGNC gene symbol (e.g. "GLP1R").

    Returns:
        List of approved drug names targeting this gene.
    """
    logger.info("Fetching ChEMBL drugs for gene=%r", gene_symbol)
    drug_names: list[str] = []

    # Step 1: find target ChEMBL ID for gene
    target_url = f"{_CHEMBL_BASE}/target.json"
    try:
        data = _chembl_get(
            target_url,
            {"gene_name": gene_symbol, "target_type": "SINGLE PROTEIN", "limit": 10},
        )
    except Exception as exc:
        logger.error("ChEMBL target search failed for %r: %s", gene_symbol, exc)
        return []

    targets = data.get("targets", [])
    if not targets:
        return []

    # Use the first exact match
    target_chembl_id = None
    for t in targets:
        components = t.get("target_components", [])
        for comp in components:
            for syn in comp.get("target_component_synonyms", []):
                if syn.get("component_synonym", "").upper() == gene_symbol.upper():
                    target_chembl_id = t.get("target_chembl_id")
                    break
            if target_chembl_id:
                break
        if target_chembl_id:
            break
    if not target_chembl_id:
        target_chembl_id = targets[0].get("target_chembl_id")

    if not target_chembl_id:
        return []

    # Step 2: find approved drugs (max_phase == 4) for this target
    activity_url = f"{_CHEMBL_BASE}/drug_indication.json"
    try:
        ind_data = _chembl_get(
            activity_url,
            {"target_chembl_id": target_chembl_id, "max_phase_for_ind": 4, "limit": 50},
        )
    except Exception as exc:
        logger.warning("ChEMBL drug indication lookup failed: %s", exc)
        ind_data = {}

    for ind in ind_data.get("drug_indications", []):
        mol_id = ind.get("molecule_chembl_id", "")
        if not mol_id:
            continue
        # Fetch molecule name
        try:
            mol_data = _chembl_get(
                f"{_CHEMBL_BASE}/molecule/{mol_id}.json"
            )
            pref_name = mol_data.get("pref_name", "") or mol_id
            if pref_name and pref_name not in drug_names:
                drug_names.append(pref_name)
        except Exception:
            continue

    # Fallback: search activities directly
    if not drug_names:
        act_url = f"{_CHEMBL_BASE}/activity.json"
        try:
            act_data = _chembl_get(
                act_url,
                {
                    "target_chembl_id": target_chembl_id,
                    "pchembl_value__isnull": False,
                    "limit": 50,
                },
            )
            for act in act_data.get("activities", []):
                name = act.get("molecule_pref_name", "") or act.get(
                    "molecule_chembl_id", ""
                )
                if name and name not in drug_names:
                    drug_names.append(name)
        except Exception as exc:
            logger.warning("ChEMBL activity lookup fallback failed: %s", exc)

    logger.info("Found %d drugs for gene %s", len(drug_names), gene_symbol)
    return drug_names
