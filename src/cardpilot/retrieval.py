"""Building the index and searching it. One Knowledge object per country."""

from __future__ import annotations

import argparse
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Literal

from .catalog import Catalog
from .config import Settings, get_settings
from .countries import COUNTRIES
from .embeddings import Embedder, get_embedder
from .schemas import Chunk
from .store import Store, open_store
from .vocab import expand

log = logging.getLogger(__name__)

Mode = Literal["hybrid", "vector", "keyword"]


@dataclass
class Knowledge:
    """Everything CardPilot knows about one country: the catalog and its search index."""

    catalog: Catalog
    store: Store
    embedder: Embedder

    @property
    def country(self) -> str:
        return self.catalog.country.code

    def search(
        self,
        query: str,
        k: int = 6,
        card_ids: list[str] | None = None,
        mode: Mode = "hybrid",
        expand_query: bool = True,
    ) -> list[Chunk]:
        if expand_query:
            query = expand(query)
        if mode == "keyword":
            return self.store.keyword_search(query, k, card_ids)
        vector = self.embedder.embed([query])[0]
        if mode == "vector":
            return self.store.vector_search(vector, k, card_ids)
        return self.store.hybrid_search(query, vector, k, card_ids)


def index_version(embedder: Embedder, chunks: list[Chunk]) -> str:
    """The embedder plus a fingerprint of every chunk. Any change to the card data gives a new version."""
    digest = hashlib.sha256("\n".join(f"{c.id}\t{c.text}\t{c.url}" for c in chunks).encode()).hexdigest()[:12]
    return f"{embedder.id}+{digest}"


def build_index(settings: Settings, code: str, embedder: Embedder | None = None) -> Knowledge:
    catalog = Catalog.load(settings.cards_dir, code)
    embedder = embedder or get_embedder(settings.embedder)
    store = open_store(settings, code, embedder.dim)
    chunks = catalog.chunks()
    started = time.perf_counter()
    vectors = embedder.embed([c.text for c in chunks])
    store.rebuild(chunks, vectors, index_version(embedder, chunks))
    log.info("Indexed %d chunks for %s with %s in %.1fs", len(chunks), code, embedder.id, time.perf_counter() - started)
    return Knowledge(catalog, store, embedder)


def load_knowledge(settings: Settings, code: str) -> Knowledge:
    """Opens the country's index, rebuilding it when it is missing, was built with another embedder, or the card
    data changed since it was built. So editing a card file never leaves a stale index behind."""
    embedder = get_embedder(settings.embedder)
    catalog = Catalog.load(settings.cards_dir, code)
    store = open_store(settings, code, embedder.dim)
    try:
        ready = store.embedder_id == index_version(embedder, catalog.chunks()) and store.count() > 0
    except Exception:
        ready = False
    if not ready:
        return build_index(settings, code, embedder)
    return Knowledge(catalog, store, embedder)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CardPilot's search index for each country.")
    parser.add_argument("--country", choices=sorted(COUNTRIES), action="append", help="Only this country (repeatable).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    for code in args.country or sorted(COUNTRIES):
        k = build_index(settings, code)
        print(f"{code}: {k.store.count()} chunks indexed with {k.embedder.id} in the {settings.store} store")


if __name__ == "__main__":
    main()
