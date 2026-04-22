"""Named Entity Recognition using scispaCy en_core_sci_lg + UMLS linker."""

from __future__ import annotations

import logging
from typing import Any

from ad_kg.models import EntityMention, Paper

logger = logging.getLogger(__name__)

# Global NLP cache to avoid reloading on every call
_NLP: Any = None

# Labels of interest
_LABELS_OF_INTEREST = {
    "CHEMICAL",
    "GENE_OR_GENE_PRODUCT",
    "DISEASE",
    "CELL_LINE",
    "CELL_TYPE",
    "DNA",
    "RNA",
    "PROTEIN",
    "ORGANISM",
    "SIMPLE_CHEMICAL",
}


def load_nlp() -> Any:
    """Load scispaCy en_core_sci_lg with UMLS entity linker.

    Caches the pipeline globally after first load.

    Returns:
        Loaded spaCy Language object with UMLS linker.
    """
    global _NLP
    if _NLP is not None:
        return _NLP

    logger.info("Loading scispaCy en_core_sci_lg pipeline...")
    try:
        import spacy  # noqa: PLC0415
        from scispacy.linking import EntityLinker  # noqa: PLC0415, F401
    except ImportError as exc:
        raise ImportError(
            "scispacy and spacy are required. Install via: "
            "pip install scispacy && pip install https://s3-us-west-2.amazonaws.com"
            "/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
        ) from exc

    try:
        nlp = spacy.load("en_core_sci_lg")
    except OSError as exc:
        raise OSError(
            "Model en_core_sci_lg not found. Download with:\n"
            "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/"
            "v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
        ) from exc

    # Add UMLS entity linker
    nlp.add_pipe(
        "scispacy_linker",
        config={
            "resolve_abbreviations": True,
            "linker_name": "umls",
            "threshold": 0.85,
        },
    )
    _NLP = nlp
    logger.info("scispaCy pipeline loaded with UMLS linker.")
    return _NLP


def extract_entities(papers: list[Paper]) -> list[EntityMention]:
    """Run NER on each paper's abstract and return deduplicated mentions.

    Args:
        papers: List of Paper objects to process.

    Returns:
        List of EntityMention objects, deduplicated per (paper_id, text, label).
    """
    nlp = load_nlp()
    mentions: list[EntityMention] = []

    for paper in papers:
        text = paper.abstract or ""
        if not text.strip():
            continue

        try:
            doc = nlp(text)
        except Exception as exc:
            logger.warning("NER failed for paper %s: %s", paper.pmid, exc)
            continue

        seen_in_paper: set[tuple[str, str]] = set()

        for ent in doc.ents:
            label = ent.label_
            ent_text = ent.text.strip()
            if not ent_text:
                continue

            # Attempt to get canonical UMLS CUI
            canonical_id: str | None = None
            if hasattr(ent, "_.kb_ents") and ent._.kb_ents:
                top_concept = ent._.kb_ents[0]
                canonical_id = top_concept[0]  # UMLS CUI

            dedup_key = (ent_text.lower(), label)
            if dedup_key in seen_in_paper:
                continue
            seen_in_paper.add(dedup_key)

            mentions.append(
                EntityMention(
                    text=ent_text,
                    label=label,
                    start=ent.start_char,
                    end=ent.end_char,
                    paper_id=paper.pmid,
                    canonical_id=canonical_id,
                )
            )

    logger.info("Extracted %d entity mentions from %d papers", len(mentions), len(papers))
    return mentions
