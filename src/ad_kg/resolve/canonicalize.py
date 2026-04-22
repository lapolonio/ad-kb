"""Entity canonicalization: HDBSCAN clustering + external ID lookup."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ad_kg.config import HDBSCAN_THRESHOLD
from ad_kg.models import EntityMention

logger = logging.getLogger(__name__)


def cluster_and_canonicalize(
    mentions: list[EntityMention],
    embeddings: np.ndarray,
    threshold: float = HDBSCAN_THRESHOLD,
) -> list[EntityMention]:
    """Cluster entity mentions by cosine similarity and assign canonical IDs.

    Uses HDBSCAN on cosine distance (1 - cosine_similarity). Mentions with
    cosine_similarity >= threshold end up in the same cluster and share a
    canonical_id of the form 'cluster_{n}'.

    Noise points (label == -1) get their UMLS canonical_id if available, or
    a unique fallback ID.

    Args:
        mentions: List of EntityMention objects.
        embeddings: (N, D) normalized embedding matrix (output of embed_mentions).
        threshold: Cluster merge threshold expressed as min cosine similarity.
                   Converted to max cosine distance = 1 - threshold.

    Returns:
        Updated list of EntityMention objects with canonical_id set.
    """
    if not mentions:
        return mentions

    try:
        import hdbscan  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "hdbscan is required. Install via: pip install hdbscan"
        ) from exc

    from sklearn.metrics.pairwise import cosine_distances  # noqa: PLC0415

    logger.info(
        "Clustering %d mentions with HDBSCAN (threshold=%.2f)", len(mentions), threshold
    )

    if len(mentions) < 2:
        for i, m in enumerate(mentions):
            if m.canonical_id is None:
                m.canonical_id = f"singleton_{i}"
        return mentions

    # Compute pairwise cosine distance matrix
    dist_matrix = cosine_distances(embeddings).astype(np.float64)

    # HDBSCAN with precomputed distances
    # cluster_selection_epsilon = 1 - threshold means clusters merge when
    # distance <= 1 - threshold, i.e. similarity >= threshold
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        metric="precomputed",
        cluster_selection_epsilon=float(1.0 - threshold),
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(dist_matrix)

    # Assign canonical IDs
    updated: list[EntityMention] = []
    for mention, label in zip(mentions, labels):
        m = EntityMention(
            text=mention.text,
            label=mention.label,
            start=mention.start,
            end=mention.end,
            paper_id=mention.paper_id,
            canonical_id=mention.canonical_id,
        )
        if label >= 0:
            m.canonical_id = f"cluster_{label}"
        elif m.canonical_id is None:
            # Noise point with no UMLS CUI — assign text-based fallback
            safe_text = m.text.lower().replace(" ", "_")[:40]
            m.canonical_id = f"noise_{safe_text}"
        updated.append(m)

    n_clusters = len(set(l for l in labels if l >= 0))
    n_noise = sum(1 for l in labels if l < 0)
    logger.info("HDBSCAN: %d clusters, %d noise points", n_clusters, n_noise)
    return updated


@retry(
    retry=retry_if_exception_type((requests.RequestException, IOError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=False,
)
def _chembl_lookup(entity_text: str) -> str | None:
    """Look up a drug name in ChEMBL and return its CHEMBL ID."""
    url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
    params = {"pref_name__iexact": entity_text, "limit": 1}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        mols = data.get("molecules", [])
        if mols:
            return mols[0].get("molecule_chembl_id")
    except Exception as exc:
        logger.debug("ChEMBL lookup failed for %r: %s", entity_text, exc)
    return None


@retry(
    retry=retry_if_exception_type((requests.RequestException, IOError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=False,
)
def _hgnc_lookup(entity_text: str) -> str | None:
    """Look up a gene symbol in HGNC and return the HGNC ID."""
    url = "https://rest.genenames.org/fetch/symbol/" + entity_text.upper()
    headers = {"Accept": "application/json"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        docs = (
            data.get("response", {})
            .get("docs", [])
        )
        if docs:
            return docs[0].get("hgnc_id")
    except Exception as exc:
        logger.debug("HGNC lookup failed for %r: %s", entity_text, exc)
    return None


@retry(
    retry=retry_if_exception_type((requests.RequestException, IOError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=False,
)
def _mesh_lookup(entity_text: str) -> str | None:
    """Look up a disease/term in MeSH and return the MeSH ID."""
    url = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
    params = {"label": entity_text, "match": "exact", "limit": 1}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            return data[0].get("resource", "").split("/")[-1]
    except Exception as exc:
        logger.debug("MeSH lookup failed for %r: %s", entity_text, exc)
    return None


def lookup_canonical_id(entity_text: str, entity_type: str) -> str | None:
    """Try to find a canonical ID for an entity from external sources.

    - Drugs / chemicals: ChEMBL
    - Genes / proteins: HGNC
    - Diseases: MeSH

    Args:
        entity_text: Entity name/text string.
        entity_type: NER label (e.g. CHEMICAL, GENE_OR_GENE_PRODUCT, DISEASE).

    Returns:
        Canonical ID string (e.g. 'CHEMBL12345', 'HGNC:123', 'D000544') or None.
    """
    etype = entity_type.upper()

    if etype in ("CHEMICAL", "SIMPLE_CHEMICAL"):
        return _chembl_lookup(entity_text)
    elif etype in ("GENE_OR_GENE_PRODUCT", "GENE", "PROTEIN", "DNA", "RNA"):
        return _hgnc_lookup(entity_text)
    elif etype in ("DISEASE", "DISORDER"):
        return _mesh_lookup(entity_text)

    return None
