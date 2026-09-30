"""Turning text into vectors for search by meaning.

Two embedders share one small interface:

- ``glove`` ships inside the package (GloVe word vectors, public domain, 64 numbers per word). It needs
  no download and no GPU, so tests and CI are fast and repeatable.
- ``bge-small`` is BAAI's bge-small-en-v1.5 through fastembed (ONNX on CPU). It is much better at
  meaning and downloads about 70 MB once. Install it with ``uv sync --extra embeddings``.

An index remembers which embedder built it, and search refuses to mix them.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class Embedder(Protocol):
    id: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Unit-length vectors, one row per text."""
        ...


_STOP = frozenset(
    "i im ive id me my mine we our you your he she it its they them their his her him the a an that this "
    "those these to of in on at for with and or but is are was were be been am do did does have has had "
    "will would can could should from by about as into up down out over under again then than so very "
    "just also not no yes what which who whom where when why how all any some each few more most other "
    "such only own same too now there here get got let lets".split()
)


class GloveEmbedder:
    """Weighted average of GloVe word vectors (smooth inverse frequency), the same model Memoir uses."""

    id = "glove-64-v1"

    def __init__(self, path: Path | None = None):
        path = path or Path(__file__).parent / "models" / "words-64.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        words = raw["words"].split(" ")
        self.dim = int(raw["dims"])
        self.index = {w: i for i, w in enumerate(words)}
        vec = np.frombuffer(base64.b64decode(raw["vectors"]), dtype=np.int8).astype(np.float32)
        self.vectors = vec.reshape(len(words), self.dim)
        harmonic = math.log(400000) + 0.5772
        ranks = np.arange(len(words), dtype=np.float64)
        p = np.where(ranks < raw["common"], 1.0 / ((ranks + 1) * harmonic), 0.0)
        self.weights = (1e-3 / (1e-3 + p)).astype(np.float32)

    def _lookup(self, word: str) -> int | None:
        found = self.index.get(word)
        if found is not None:
            return found
        for ending in ("'s", "s", "es", "ing", "ed"):
            if len(word) > len(ending) + 2 and word.endswith(ending):
                stem = self.index.get(word[: -len(ending)])
                if stem is not None:
                    return stem
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()):
                if word in _STOP:
                    continue
                i = self._lookup(word)
                if i is not None:
                    out[row] += self.weights[i] * self.vectors[i]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class FastEmbedEmbedder:
    """bge-small-en-v1.5 on CPU through fastembed. Queries get the model's search prefix."""

    id = "bge-small-en-v1.5"
    dim = 384

    def __init__(self):
        from fastembed import TextEmbedding  # optional dependency

        self.model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self.model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


@lru_cache
def get_embedder(name: str = "glove") -> Embedder:
    if name == "bge-small":
        try:
            return FastEmbedEmbedder()
        except Exception as exc:  # missing package or no network for the first download
            log.warning("bge-small is not available (%s). Falling back to the built-in GloVe embedder.", exc)
    return GloveEmbedder()
