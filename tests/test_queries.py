"""Tests for the named Cypher queries."""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from ad_kg.graph.queries import QUERIES, run_query


# ── Query validity tests ───────────────────────────────────────────────────────

def test_all_queries_present():
    """All 11 named queries exist in the QUERIES dict."""
    expected = {
        "whitespace_opportunity",
        "triple_convergence",
        "bridge_genes_ranked",
        "repurposing_candidates",
        "genetic_overlap",
        "pathway_bridges",
        "trial_gaps",
        "gwas_snp_to_drug",
        "open_trials_bridge_genes",
        "protective_drugs_ranked",
        "semaglutide_neighbors",
    }
    assert set(QUERIES.keys()) == expected


def test_all_queries_valid_syntax():
    """All 11 queries are non-empty strings containing valid Cypher keywords."""
    required_keywords = {"MATCH", "RETURN"}
    for name, cypher in QUERIES.items():
        assert isinstance(cypher, str), f"Query {name!r} is not a string"
        assert cypher.strip(), f"Query {name!r} is empty"
        upper = cypher.upper()
        for kw in required_keywords:
            assert kw in upper, f"Query {name!r} missing keyword {kw!r}"


def test_all_queries_non_empty():
    """QUERIES dict has 11 entries, all non-None."""
    assert len(QUERIES) == 11
    for name, cypher in QUERIES.items():
        assert cypher is not None
        assert len(cypher) > 10


# ── Whitespace opportunity structure ─────────────────────────────────────────

def test_whitespace_opportunity_structure():
    """whitespace_opportunity query contains expected Cypher patterns."""
    q = QUERIES["whitespace_opportunity"]
    upper = q.upper()

    # Must filter for protective signal
    assert "ROR" in upper or "ror" in q
    assert "< 1" in q or "<1" in q or "< 1.0" in q

    # Must check for absence of trials
    assert "NOT EXISTS" in upper or "WHERE NOT" in upper

    # Must return drug info
    assert "RETURN" in upper
    assert "drug" in q.lower()


def test_whitespace_opportunity_no_active_trial_filter():
    """whitespace_opportunity excludes drugs with active AD trials."""
    q = QUERIES["whitespace_opportunity"]
    # Should reference trial status
    assert "RECRUITING" in q or "status" in q.lower()


def test_triple_convergence_three_signals():
    """triple_convergence checks for FAERS + GWAS + literature signals."""
    q = QUERIES["triple_convergence"]
    upper = q.upper()
    assert "PROTECTIVE_SIGNAL" in upper or "ror" in q.lower()
    assert "ASSOCIATED_WITH" in upper or "GWAS" in upper or "SNP" in upper
    assert "PAPER" in upper or "lit" in q.lower()


def test_bridge_genes_ranked_p_value():
    """bridge_genes_ranked uses p-value scoring."""
    q = QUERIES["bridge_genes_ranked"]
    assert "p_value" in q.lower() or "pval" in q.lower()
    assert "RETURN" in q.upper()


def test_semaglutide_neighbors_two_hops():
    """semaglutide_neighbors traverses up to 2 hops."""
    q = QUERIES["semaglutide_neighbors"]
    assert "semaglutide" in q.lower()
    # 2-hop traversal
    assert "1..2" in q or "2" in q
    assert "RETURN" in q.upper()


# ── run_query with mock driver ─────────────────────────────────────────────────

def _make_mock_driver(rows: list[dict]) -> MagicMock:
    """Create a mock Neo4j driver that returns specified rows."""
    mock_record = MagicMock()
    mock_record.__iter__ = lambda s: iter(rows[0].items()) if rows else iter([])

    mock_result = MagicMock()
    mock_result.__iter__ = lambda s: iter([
        _dict_to_record(r) for r in rows
    ])

    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = mock_result

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session
    return mock_driver


def _dict_to_record(d: dict) -> MagicMock:
    """Convert a dict to a mock Neo4j Record."""
    record = MagicMock()
    record.__iter__ = lambda s: iter(d.items())
    # Make dict() work on the record
    record.data.return_value = d
    record.keys.return_value = list(d.keys())
    record.values.return_value = list(d.values())
    record.items.return_value = list(d.items())
    # Support dict(record) via __getitem__
    record.__getitem__ = lambda s, k: d[k]
    return record


def test_run_query_unknown_name():
    """run_query raises ValueError for unknown query names."""
    mock_driver = MagicMock()
    with pytest.raises(ValueError, match="Unknown query"):
        run_query(mock_driver, "nonexistent_query")


def test_run_query_calls_session():
    """run_query calls driver.session() and session.run()."""
    # Patch run_query to use a simple mock
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    # Mock result that returns empty list on iteration
    mock_result = MagicMock()
    mock_result.__iter__ = lambda s: iter([])
    mock_session.run.return_value = mock_result

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    result = run_query(mock_driver, "whitespace_opportunity")

    mock_driver.session.assert_called_once()
    mock_session.run.assert_called_once()
    assert isinstance(result, list)


def test_run_query_with_limit():
    """run_query appends LIMIT clause when limit is provided."""
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_result = MagicMock()
    mock_result.__iter__ = lambda s: iter([])
    mock_session.run.return_value = mock_result

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    run_query(mock_driver, "whitespace_opportunity", limit=10)

    call_args = mock_session.run.call_args
    cypher = call_args[0][0]
    assert "LIMIT 10" in cypher


# ── Query content checks ───────────────────────────────────────────────────────

def test_gwas_snp_to_drug_returns_path():
    """gwas_snp_to_drug returns snp, gene, drug fields."""
    q = QUERIES["gwas_snp_to_drug"]
    assert "snp" in q.lower()
    assert "gene" in q.lower()
    assert "drug" in q.lower()


def test_protective_drugs_ranked_orders_by_ror():
    """protective_drugs_ranked orders by ROR ascending."""
    q = QUERIES["protective_drugs_ranked"]
    upper = q.upper()
    assert "ORDER BY" in upper
    assert "ror" in q.lower()


def test_open_trials_bridge_genes_active_status():
    """open_trials_bridge_genes filters on active trial statuses."""
    q = QUERIES["open_trials_bridge_genes"]
    assert "RECRUITING" in q
