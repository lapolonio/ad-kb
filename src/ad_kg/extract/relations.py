"""LLM-based relation extraction using Claude with prompt caching."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ad_kg.config import ANTHROPIC_API_KEY, RELATION_EXTRACTION_MAX_PAPERS
from ad_kg.models import Paper

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a biomedical knowledge extraction expert specializing in
Alzheimer's disease drug repurposing research.

Your task is to extract structured (subject, predicate, object) triples from
biomedical paper abstracts.  Focus on relationships between:
- Drugs / small molecules
- Genes / proteins
- Diseases / conditions
- Biological pathways
- Biomarkers

For each triple, assign a confidence score (0.0–1.0) based on how explicitly
the relationship is stated in the text.

Common predicates to extract:
- TREATS, INHIBITS, ACTIVATES, UPREGULATES, DOWNREGULATES
- ASSOCIATED_WITH, CAUSES, PREVENTS, BIOMARKER_OF
- TARGETS, BINDS_TO, PATHWAY_MEMBER
- REPURPOSED_FOR, PROTECTIVE_AGAINST

Output ONLY a valid JSON array with objects of this shape:
{
  "subject": "entity name",
  "predicate": "PREDICATE",
  "object": "entity name",
  "confidence": 0.85
}

If no clear triples can be extracted, return an empty array: []
Do not include any text outside the JSON array."""


def _get_client() -> anthropic.Anthropic:
    """Return an Anthropic client, using env key if config not set."""
    api_key = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return anthropic.Anthropic(api_key=api_key)
    return anthropic.Anthropic()


@retry(
    retry=retry_if_exception_type(
        (anthropic.APIConnectionError, anthropic.InternalServerError)
    ),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call_claude(client: anthropic.Anthropic, user_message: str) -> str:
    """Call Claude with prompt caching on the system prompt."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    # Log cache usage
    usage = response.usage
    logger.debug(
        "Cache stats: creation=%d read=%d input=%d",
        getattr(usage, "cache_creation_input_tokens", 0),
        getattr(usage, "cache_read_input_tokens", 0),
        usage.input_tokens,
    )

    for block in response.content:
        if block.type == "text":
            return block.text
    return "[]"


def _parse_triples(raw: str, paper_ids: list[str]) -> list[dict[str, Any]]:
    """Parse Claude's JSON output into a list of triple dicts."""
    raw = raw.strip()
    # Sometimes the model wraps JSON in markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(
            line for line in lines if not line.startswith("```")
        )

    try:
        triples = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse triples JSON: %s\nRaw: %s", exc, raw[:200])
        return []

    if not isinstance(triples, list):
        return []

    results: list[dict[str, Any]] = []
    for t in triples:
        if not isinstance(t, dict):
            continue
        subj = t.get("subject", "").strip()
        pred = t.get("predicate", "").strip().upper()
        obj = t.get("object", "").strip()
        conf = float(t.get("confidence", 0.5))

        if not (subj and pred and obj):
            continue

        for pid in paper_ids:
            results.append(
                {
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "paper_id": pid,
                    "confidence": conf,
                }
            )
    return results


def extract_relations_llm(
    papers: list[Paper],
    max_papers: int = RELATION_EXTRACTION_MAX_PAPERS,
) -> list[dict[str, Any]]:
    """Extract (subject, predicate, object) triples from paper abstracts.

    Uses claude-sonnet-4-6 with prompt caching on the system prompt to
    minimize API cost on repeated calls.

    Args:
        papers: List of Paper objects to process.
        max_papers: Maximum number of papers to process.

    Returns:
        List of dicts with keys: subject, predicate, object, paper_id, confidence.
    """
    client = _get_client()
    papers_to_process = papers[:max_papers]
    logger.info(
        "Extracting relations from %d papers (limit=%d)",
        len(papers_to_process),
        max_papers,
    )

    all_relations: list[dict[str, Any]] = []
    batch_size = 5  # Process papers in small batches to reduce API calls

    for batch_start in range(0, len(papers_to_process), batch_size):
        batch = papers_to_process[batch_start : batch_start + batch_size]

        # Build user message with batch of abstracts
        abstract_parts: list[str] = []
        paper_ids: list[str] = []

        for paper in batch:
            abstract = paper.abstract or ""
            if not abstract.strip():
                continue
            abstract_parts.append(
                f"[Paper PMID: {paper.pmid}]\n"
                f"Title: {paper.title}\n"
                f"Abstract: {abstract}"
            )
            paper_ids.append(paper.pmid)

        if not abstract_parts:
            continue

        user_message = (
            "Extract biomedical relation triples from the following paper abstracts. "
            "Return a single JSON array containing all triples from all papers:\n\n"
            + "\n\n---\n\n".join(abstract_parts)
        )

        logger.debug(
            "Processing batch %d-%d (papers: %s)",
            batch_start,
            batch_start + len(batch),
            paper_ids,
        )

        try:
            raw_output = _call_claude(client, user_message)
            triples = _parse_triples(raw_output, paper_ids)
            all_relations.extend(triples)
            logger.debug("Extracted %d triples from batch", len(triples))
        except anthropic.RateLimitError as exc:
            logger.warning("Rate limited on batch %d: %s", batch_start, exc)
            continue
        except Exception as exc:
            logger.error("Error processing batch %d: %s", batch_start, exc)
            continue

    logger.info("Total relations extracted: %d", len(all_relations))
    return all_relations
