"""Embedding of entity mentions using allenai/specter2."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ad_kg.models import EntityMention

logger = logging.getLogger(__name__)

_BASE_MODEL = "allenai/specter2_base"

_MODEL: Any = None
_TOKENIZER: Any = None


def _get_model() -> tuple[Any, Any]:
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    logger.info("Loading embedding model: %s", _BASE_MODEL)
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers and torch are required for embedding.") from exc

    _TOKENIZER = AutoTokenizer.from_pretrained(_BASE_MODEL)
    _MODEL = AutoModel.from_pretrained(_BASE_MODEL, use_safetensors=True)
    _MODEL.eval()

    if torch.backends.mps.is_available():
        _MODEL = _MODEL.to("mps")
    logger.info("Embedding model loaded on %s.", next(_MODEL.parameters()).device)
    return _MODEL, _TOKENIZER


def embed_mentions(
    mentions: list[EntityMention],
    batch_size: int = 64,
) -> np.ndarray:
    """Embed entity mention texts using SPECTER2 base with mean pooling.

    Returns L2-normalized (N, D) float32 matrix.
    """
    if not mentions:
        return np.zeros((0, 1), dtype=np.float32)

    import torch
    import torch.nn.functional as F

    model, tokenizer = _get_model()
    device = next(model.parameters()).device
    texts = [m.text for m in mentions]

    logger.info("Embedding %d entity mentions (batch_size=%d)", len(texts), batch_size)

    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)

        # Mean pool over non-padding tokens
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vecs = F.normalize(vecs, p=2, dim=1)
        chunks.append(vecs.cpu().numpy())

        if (start // batch_size) % 10 == 0:
            logger.info("  embedded %d / %d", min(start + batch_size, len(texts)), len(texts))

    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    logger.info("Embeddings shape: %s", embeddings.shape)
    return embeddings
