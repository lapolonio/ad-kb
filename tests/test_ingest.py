"""Tests for ingest modules."""

from __future__ import annotations

import math
import os
from unittest.mock import MagicMock, patch

import pytest

# VCR cassette directory (vcrpy is an optional dev dependency)
CASSETTE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "cassettes")
try:
    import vcr as _vcr  # noqa: F401
    HAS_VCR = True
except ImportError:
    HAS_VCR = False


# ── chunk_text tests ──────────────────────────────────────────────────────────

def test_chunk_text_short_text():
    """A text shorter than chunk_size returns a single chunk."""
    from ad_kg.ingest.pubmed import chunk_text

    text = "Semaglutide reduces amyloid burden in mouse models."
    chunks = chunk_text(text, chunk_size=512, stride=128)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_overlap():
    """Chunked windows overlap by (chunk_size - stride) words."""
    from ad_kg.ingest.pubmed import chunk_text

    # Create a text with 100 words
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)

    chunk_size = 20
    stride = 10  # overlap = chunk_size - stride = 10

    chunks = chunk_text(text, chunk_size=chunk_size, stride=stride)

    # Should have multiple chunks
    assert len(chunks) > 1

    # Verify overlap: last `overlap` words of chunk[n] == first `overlap` words of chunk[n+1]
    overlap = chunk_size - stride
    for i in range(len(chunks) - 1):
        tail = chunks[i].split()[-overlap:]
        head = chunks[i + 1].split()[:overlap]
        assert tail == head, f"Overlap mismatch at chunk {i}"


def test_chunk_text_empty():
    """Empty input returns ['']."""
    from ad_kg.ingest.pubmed import chunk_text

    result = chunk_text("", chunk_size=512, stride=128)
    assert result == [""]


def test_chunk_text_exact_stride():
    """Verify first chunk starts at position 0 and second at stride."""
    from ad_kg.ingest.pubmed import chunk_text

    words = [f"w{i}" for i in range(30)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=10, stride=5)

    first_words = chunks[0].split()[:5]
    second_words = chunks[1].split()[:5]

    # First chunk starts with w0..w4
    assert first_words == [f"w{i}" for i in range(5)]
    # Second chunk starts with w5..w9
    assert second_words == [f"w{i}" for i in range(5, 10)]


# ── compute_ror tests ─────────────────────────────────────────────────────────

def test_faers_ror_unit():
    """ROR ≈ 1.0 when proportions are equal (with continuity correction)."""
    from ad_kg.ingest.faers import compute_ror

    ror, ci_lower, ci_upper = compute_ror(10, 90, 100, 900)
    # 10/100 = 10%, 100/1000 = 10% → ROR ≈ 1.0
    assert abs(ror - 1.0) < 0.05, f"Expected ROR ≈ 1.0, got {ror}"
    assert ci_lower < ror < ci_upper
    assert ci_lower > 0
    assert ci_upper > ci_lower


def test_faers_ror_protective():
    """ROR < 1.0 when drug has proportionally fewer AD reports."""
    from ad_kg.ingest.faers import compute_ror

    # Drug has 5/100 AD rate vs background 100/1000 = 10%
    ror, ci_lower, ci_upper = compute_ror(5, 95, 100, 900)
    assert ror < 1.0, f"Expected protective ROR < 1.0, got {ror}"
    assert ci_lower > 0
    assert ci_upper > ci_lower


def test_faers_ror_elevated():
    """ROR > 1.0 when drug has proportionally more AD reports."""
    from ad_kg.ingest.faers import compute_ror

    # Drug has 20/100 = 20% AD rate vs background 100/1000 = 10%
    ror, ci_lower, ci_upper = compute_ror(20, 80, 100, 900)
    assert ror > 1.0, f"Expected elevated ROR > 1.0, got {ror}"


def test_faers_ror_zero_cells():
    """compute_ror handles zero counts via continuity correction."""
    from ad_kg.ingest.faers import compute_ror

    # a=0: drug never reported with AD
    ror, ci_lower, ci_upper = compute_ror(0, 100, 50, 850)
    assert math.isfinite(ror)
    assert math.isfinite(ci_lower)
    assert math.isfinite(ci_upper)
    assert ci_lower > 0


def test_faers_ror_returns_tuple():
    """compute_ror returns a 3-tuple of floats."""
    from ad_kg.ingest.faers import compute_ror

    result = compute_ror(10, 90, 100, 900)
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


# ── PubMed dedup test (mocked) ─────────────────────────────────────────────────

def _make_mock_article(pmid: str, title: str = "Test", abstract: str = "Abstract"):
    """Create a minimal mock pymed article."""
    art = MagicMock()
    art.pubmed_id = pmid
    art.title = title
    art.abstract = abstract
    art.publication_date = "2024-01-01"
    art.authors = []
    return art


def test_pubmed_dedup():
    """fetch_pubmed deduplicates articles with the same PMID."""
    from ad_kg.ingest import pubmed as pubmed_module  # noqa: PLC0415

    # Return two articles with duplicate PMIDs
    articles = [
        _make_mock_article("12345678"),
        _make_mock_article("12345678"),  # duplicate
        _make_mock_article("99999999"),
    ]
    mock_instance = MagicMock()
    mock_instance.query.return_value = articles

    with patch.object(pubmed_module, "_build_pymed_client", return_value=mock_instance):
        papers = pubmed_module.fetch_pubmed("Alzheimer semaglutide", max_results=10)

    pmids = [p.pmid for p in papers]

    assert len(pmids) == len(set(pmids)), "Duplicate PMIDs found"
    assert "12345678" in pmids
    assert "99999999" in pmids
    assert len(papers) == 2  # 3 articles but only 2 unique PMIDs


def test_pubmed_two_calls_same_result():
    """Calling fetch_pubmed twice returns identical PMID sets."""
    from ad_kg.ingest import pubmed as pubmed_module  # noqa: PLC0415

    articles = [
        _make_mock_article("11111111"),
        _make_mock_article("22222222"),
    ]
    mock_instance = MagicMock()
    mock_instance.query.return_value = articles

    with patch.object(pubmed_module, "_build_pymed_client", return_value=mock_instance):
        papers1 = pubmed_module.fetch_pubmed("Alzheimer", max_results=5)

    with patch.object(pubmed_module, "_build_pymed_client", return_value=mock_instance):
        papers2 = pubmed_module.fetch_pubmed("Alzheimer", max_results=5)

    pmids1 = {p.pmid for p in papers1}
    pmids2 = {p.pmid for p in papers2}
    assert pmids1 == pmids2
