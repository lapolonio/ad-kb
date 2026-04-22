"""Tests for resolve modules (embedding and canonicalization)."""

from __future__ import annotations

import numpy as np
import pytest

from ad_kg.models import EntityMention

try:
    import sentence_transformers as _st  # noqa: F401
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

requires_st = pytest.mark.skipif(
    not _HAS_ST,
    reason="sentence-transformers not installed; install with: uv sync --extra ml",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mention(text: str, label: str = "CHEMICAL", paper_id: str = "pmid1") -> EntityMention:
    return EntityMention(text=text, label=label, start=0, end=len(text), paper_id=paper_id)


# ── Embedding tests ───────────────────────────────────────────────────────────

@pytest.mark.slow
@requires_st
def test_embed_returns_normalized():
    """embed_mentions returns unit vectors (L2 norm ≈ 1.0)."""
    from ad_kg.resolve.embed import embed_mentions

    mentions = [
        _make_mention("semaglutide"),
        _make_mention("Ozempic"),
        _make_mention("liraglutide"),
    ]
    embeddings = embed_mentions(mentions)

    assert embeddings.shape[0] == 3
    assert embeddings.ndim == 2

    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5), \
        f"Expected unit vectors, got norms: {norms}"


@pytest.mark.slow
@requires_st
def test_embed_empty_mentions():
    """embed_mentions handles an empty list."""
    from ad_kg.resolve.embed import embed_mentions

    result = embed_mentions([])
    assert result.shape[0] == 0


@pytest.mark.slow
@requires_st
def test_embed_single_mention():
    """embed_mentions works for a single mention."""
    from ad_kg.resolve.embed import embed_mentions

    mentions = [_make_mention("metformin")]
    embeddings = embed_mentions(mentions)
    assert embeddings.shape == (1, embeddings.shape[1])
    norm = np.linalg.norm(embeddings[0])
    assert abs(norm - 1.0) < 1e-5


# ── Canonicalization tests ────────────────────────────────────────────────────

@pytest.mark.slow
@requires_st
def test_cluster_synonyms():
    """semaglutide and Ozempic (brand name) cluster together."""
    from ad_kg.resolve.embed import embed_mentions
    from ad_kg.resolve.canonicalize import cluster_and_canonicalize

    mentions = [
        _make_mention("semaglutide"),
        _make_mention("Ozempic"),          # brand name for semaglutide
        _make_mention("liraglutide"),      # different GLP-1 agonist (similar but different)
    ]
    embeddings = embed_mentions(mentions)
    resolved = cluster_and_canonicalize(mentions, embeddings, threshold=0.88)

    assert len(resolved) == len(mentions)

    # semaglutide and Ozempic should share the same canonical_id
    sema_id = next(m.canonical_id for m in resolved if m.text == "semaglutide")
    ozem_id = next(m.canonical_id for m in resolved if m.text == "Ozempic")

    assert sema_id == ozem_id, (
        f"Expected semaglutide ({sema_id}) and Ozempic ({ozem_id}) to cluster together"
    )


@pytest.mark.slow
@requires_st
def test_cluster_different_drugs():
    """semaglutide and metformin have different canonical IDs (different drug classes)."""
    from ad_kg.resolve.embed import embed_mentions
    from ad_kg.resolve.canonicalize import cluster_and_canonicalize

    mentions = [
        _make_mention("semaglutide"),
        _make_mention("metformin"),
    ]
    embeddings = embed_mentions(mentions)
    resolved = cluster_and_canonicalize(mentions, embeddings, threshold=0.88)

    sema_id = next(m.canonical_id for m in resolved if m.text == "semaglutide")
    met_id = next(m.canonical_id for m in resolved if m.text == "metformin")

    assert sema_id != met_id, (
        f"semaglutide and metformin should NOT merge, but both got id={sema_id}"
    )


def test_cluster_assigns_all_canonical_ids():
    """All mentions get a non-None canonical_id after clustering."""
    from ad_kg.resolve.canonicalize import cluster_and_canonicalize

    mentions = [
        _make_mention("aspirin"),
        _make_mention("ibuprofen"),
        _make_mention("naproxen"),
    ]
    # Use precomputed fake embeddings (unit vectors)
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((3, 768)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    embeddings = raw / norms

    resolved = cluster_and_canonicalize(mentions, embeddings, threshold=0.88)

    for m in resolved:
        assert m.canonical_id is not None, f"Mention {m.text!r} has no canonical_id"


def test_cluster_single_mention():
    """cluster_and_canonicalize handles a single mention."""
    from ad_kg.resolve.canonicalize import cluster_and_canonicalize

    mentions = [_make_mention("semaglutide")]
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    resolved = cluster_and_canonicalize(mentions, embeddings)

    assert len(resolved) == 1
    assert resolved[0].canonical_id is not None
