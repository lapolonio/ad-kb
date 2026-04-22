"""PubMed ingest: fetch papers via pymed with retry, dedup, and rate limiting."""

from __future__ import annotations

import logging
import time
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ad_kg.config import MAX_PUBMED_PER_QUERY, NCBI_API_KEY, NCBI_EMAIL, NCBI_SLEEP
from ad_kg.models import Paper

logger = logging.getLogger(__name__)


def _build_pymed_client() -> Any:
    """Return a configured PubMed client."""
    try:
        from pymed import PubMed  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pymed is required. Install via: pip install pymed"
        ) from exc
    tool = "ad_kg_pipeline"
    return PubMed(tool=tool, email=NCBI_EMAIL)


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _query_pubmed(pubmed: PubMed, query: str, max_results: int) -> list[Any]:
    """Execute a PubMed query with retry/backoff."""
    results = list(pubmed.query(query, max_results=max_results))
    return results


def _article_to_paper(article: Any) -> Paper | None:
    """Convert a pymed Article to a Paper dataclass."""
    try:
        pmid = str(article.pubmed_id).strip().split("\n")[0]
        title = article.title or ""
        abstract = article.abstract or ""
        pub_date = ""
        if article.publication_date:
            try:
                pub_date = str(article.publication_date)
            except Exception:
                pub_date = ""
        authors: list[str] = []
        if article.authors:
            for a in article.authors:
                if isinstance(a, dict):
                    last = a.get("lastname", "") or ""
                    first = a.get("firstname", "") or ""
                    authors.append(f"{last}, {first}".strip(", "))
                else:
                    authors.append(str(a))
        return Paper(
            pmid=pmid,
            title=title,
            abstract=abstract,
            pub_date=pub_date,
            authors=authors,
        )
    except Exception as exc:
        logger.warning("Failed to parse article: %s", exc)
        return None


def fetch_pubmed(
    query: str,
    max_results: int = MAX_PUBMED_PER_QUERY,
    limit: int | None = None,
) -> list[Paper]:
    """Fetch papers from PubMed, deduplicated by PMID.

    Args:
        query: PubMed search query string.
        max_results: Maximum number of results to fetch.
        limit: Optional override for max_results (used for dev runs).

    Returns:
        Deduplicated list of Paper objects.
    """
    effective_max = limit if limit is not None else max_results
    logger.info("Fetching PubMed: query=%r max=%d", query, effective_max)

    pubmed = _build_pymed_client()
    time.sleep(NCBI_SLEEP)  # respect rate limit before first call

    try:
        articles = _query_pubmed(pubmed, query, effective_max)
    except Exception as exc:
        logger.error("PubMed query failed after retries: %s", exc)
        return []

    seen_pmids: set[str] = set()
    papers: list[Paper] = []

    for article in articles:
        paper = _article_to_paper(article)
        if paper is None:
            continue
        if paper.pmid in seen_pmids:
            logger.debug("Skipping duplicate PMID %s", paper.pmid)
            continue
        seen_pmids.add(paper.pmid)
        papers.append(paper)
        time.sleep(NCBI_SLEEP)  # throttle between record fetches

    logger.info("Fetched %d unique papers", len(papers))
    return papers


def chunk_text(text: str, chunk_size: int = 512, stride: int = 128) -> list[str]:
    """Split text into overlapping windows of tokens (whitespace-split words).

    Args:
        text: Input text to chunk.
        chunk_size: Number of words per chunk.
        stride: Step size between chunk starts (produces overlap of
                chunk_size - stride words).

    Returns:
        List of text chunks. Returns [""] for empty input.
    """
    if not text or not text.strip():
        return [""]

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += stride

    return chunks
