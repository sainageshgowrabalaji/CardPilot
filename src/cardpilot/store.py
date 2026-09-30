"""The search index, one per country, with keyword search and vector search.

Country isolation is structural. A store is opened for exactly one country (its own SQLite file, or
its own Postgres schema), and its search methods take no country argument. There is no query that
can reach another country's documents.

- ``SQLiteStore`` needs no setup. FTS5 with BM25 for keywords, numpy for vectors. Good for a laptop,
  CI and small deployments.
- ``PostgresStore`` is the production store. pgvector for vectors, full-text search for keywords,
  and both fused in one SQL query.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Protocol

import numpy as np

from .countries import get_country
from .schemas import Chunk


class Store(Protocol):
    country: str
    embedder_id: str | None

    def rebuild(self, chunks: list[Chunk], vectors: np.ndarray, embedder_id: str) -> None: ...
    def keyword_search(self, query: str, k: int, card_ids: list[str] | None = None) -> list[Chunk]: ...
    def vector_search(self, vector: np.ndarray, k: int, card_ids: list[str] | None = None) -> list[Chunk]: ...
    def hybrid_search(
        self, query: str, vector: np.ndarray, k: int, card_ids: list[str] | None = None
    ) -> list[Chunk]: ...
    def count(self) -> int: ...


RRF_K = 60  # the usual constant for reciprocal rank fusion


def fts_query(text: str) -> str:
    """A safe FTS5 query: plain words joined with OR, so user text can never inject FTS syntax."""
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1]
    return " OR ".join(f'"{w}"' for w in words[:24])


def rrf(lists: list[list[Chunk]], k: int) -> list[Chunk]:
    """Reciprocal rank fusion. A chunk that ranks well in both lists wins."""
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for results in lists:
        for rank, chunk in enumerate(results, 1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank)
            by_id.setdefault(chunk.id, chunk)
    ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k]
    return [by_id[cid].model_copy(update={"score": round(scores[cid], 5)}) for cid in ranked]


class SQLiteStore:
    def __init__(self, index_dir: Path, country: str):
        self.country = get_country(country).code  # validates against the allowlist
        index_dir.mkdir(parents=True, exist_ok=True)
        self.path = index_dir / f"{self.country}.sqlite"
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS chunks (
              rowid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, card_id TEXT, card_name TEXT, kind TEXT,
              text TEXT NOT NULL, url TEXT, title TEXT, as_of TEXT, vec BLOB NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
              text, card_name, content='chunks', content_rowid='rowid', tokenize='porter unicode61'
            );
            """
        )
        row = self.db.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
        self.embedder_id = row[0] if row else None
        self._matrix: np.ndarray | None = None
        self._rows: list[Chunk] = []

    def rebuild(self, chunks: list[Chunk], vectors: np.ndarray, embedder_id: str) -> None:
        if any(c.country != self.country for c in chunks):
            raise ValueError("Refusing to index a chunk from another country.")
        with self.db:
            self.db.execute("DELETE FROM chunks")
            self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('delete-all')")
            for chunk, vec in zip(chunks, vectors, strict=True):
                cur = self.db.execute(
                    "INSERT INTO chunks (id, card_id, card_name, kind, text, url, title, as_of, vec) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        chunk.id,
                        chunk.card_id,
                        chunk.card_name,
                        chunk.kind,
                        chunk.text,
                        chunk.url,
                        chunk.title,
                        chunk.as_of,
                        np.asarray(vec, dtype=np.float32).tobytes(),
                    ),
                )
                self.db.execute(
                    "INSERT INTO chunks_fts(rowid, text, card_name) VALUES (?, ?, ?)",
                    (cur.lastrowid, chunk.text, chunk.card_name or ""),
                )
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('embedder', ?)", (embedder_id,))
        self.embedder_id = embedder_id
        self._matrix = None

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def _row(self, r: tuple) -> Chunk:
        return Chunk(
            id=r[0],
            country=self.country,
            card_id=r[1],
            card_name=r[2],
            kind=r[3],
            text=r[4],
            url=r[5],
            title=r[6],
            as_of=r[7],
        )

    def _load(self) -> None:
        rows = self.db.execute(
            "SELECT id, card_id, card_name, kind, text, url, title, as_of, vec FROM chunks ORDER BY rowid"
        ).fetchall()
        self._rows = [self._row(r) for r in rows]
        self._matrix = (
            np.stack([np.frombuffer(r[8], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 1), np.float32)
        )

    def keyword_search(self, query: str, k: int, card_ids: list[str] | None = None) -> list[Chunk]:
        q = fts_query(query)
        if not q:
            return []
        sql = (
            "SELECT c.id, c.card_id, c.card_name, c.kind, c.text, c.url, c.title, c.as_of, bm25(chunks_fts) AS s "
            "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid WHERE chunks_fts MATCH ?"
        )
        params: list = [q]
        if card_ids:
            sql += f" AND (c.card_id IN ({','.join('?' * len(card_ids))}) OR c.card_id IS NULL)"
            params += card_ids
        sql += " ORDER BY s LIMIT ?"
        params.append(k)
        return [
            self._row(r).model_copy(update={"score": -float(r[8])}) for r in self.db.execute(sql, params).fetchall()
        ]

    def vector_search(self, vector: np.ndarray, k: int, card_ids: list[str] | None = None) -> list[Chunk]:
        if self._matrix is None:
            self._load()
        assert self._matrix is not None
        if not len(self._rows):
            return []
        sims = self._matrix @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-sims)
        out: list[Chunk] = []
        for i in order:
            chunk = self._rows[int(i)]
            if card_ids and chunk.card_id is not None and chunk.card_id not in card_ids:
                continue
            out.append(chunk.model_copy(update={"score": float(sims[int(i)])}))
            if len(out) >= k:
                break
        return out

    def hybrid_search(self, query: str, vector: np.ndarray, k: int, card_ids: list[str] | None = None) -> list[Chunk]:
        return rrf([self.vector_search(vector, 30, card_ids), self.keyword_search(query, 30, card_ids)], k)


class PostgresStore:
    """pgvector plus Postgres full-text search, fused with RRF in one query. One schema per country."""

    def __init__(self, database_url: str, country: str, dim: int):
        import psycopg
        from pgvector.psycopg import register_vector

        self.country = get_country(country).code
        self.schema = f"cp_{self.country}"  # from the allowlist, never from user input
        self.dim = dim
        self.conn = psycopg.connect(database_url, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        s = self.schema
        self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS {s}.meta (key text PRIMARY KEY, value text)")
        row = self.conn.execute(f"SELECT value FROM {s}.meta WHERE key = 'embedder'").fetchone()
        self.embedder_id = row[0] if row else None

    def rebuild(self, chunks: list[Chunk], vectors: np.ndarray, embedder_id: str) -> None:
        if any(c.country != self.country for c in chunks):
            raise ValueError("Refusing to index a chunk from another country.")
        s = self.schema
        with self.conn.transaction():
            self.conn.execute(f"DROP TABLE IF EXISTS {s}.chunks")
            self.conn.execute(
                f"""CREATE TABLE {s}.chunks (
                  id text PRIMARY KEY, card_id text, card_name text, kind text, text text NOT NULL,
                  url text, title text, as_of text, embedding vector({vectors.shape[1]}) NOT NULL,
                  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(card_name, '') || ' ' || text)) STORED
                )"""
            )
            self.conn.execute(f"CREATE INDEX ON {s}.chunks USING gin (tsv)")
            with self.conn.cursor() as cur:
                for chunk, vec in zip(chunks, vectors, strict=True):
                    cur.execute(
                        f"INSERT INTO {s}.chunks (id, card_id, card_name, kind, text, url, title, as_of, embedding) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (
                            chunk.id,
                            chunk.card_id,
                            chunk.card_name,
                            chunk.kind,
                            chunk.text,
                            chunk.url,
                            chunk.title,
                            chunk.as_of,
                            np.asarray(vec, dtype=np.float32),
                        ),
                    )
            self.conn.execute(f"CREATE INDEX ON {s}.chunks USING hnsw (embedding vector_cosine_ops)")
            self.conn.execute(
                f"INSERT INTO {s}.meta (key, value) VALUES ('embedder', %s) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (embedder_id,),
            )
        self.embedder_id = embedder_id

    def count(self) -> int:
        return self.conn.execute(f"SELECT COUNT(*) FROM {self.schema}.chunks").fetchone()[0]

    def _chunks(self, rows) -> list[Chunk]:
        return [
            Chunk(
                id=r[0],
                country=self.country,
                card_id=r[1],
                card_name=r[2],
                kind=r[3],
                text=r[4],
                url=r[5],
                title=r[6],
                as_of=r[7],
                score=float(r[8]),
            )
            for r in rows
        ]

    def _filter(self, card_ids: list[str] | None) -> tuple[str, list]:
        if not card_ids:
            return "", []
        return " AND (card_id = ANY(%s) OR card_id IS NULL)", [card_ids]

    def keyword_search(self, query: str, k: int, card_ids: list[str] | None = None) -> list[Chunk]:
        where, params = self._filter(card_ids)
        q = " | ".join(re.findall(r"[a-z0-9]+", query.lower())[:24]) or "nothing"
        rows = self.conn.execute(
            f"""SELECT id, card_id, card_name, kind, text, url, title, as_of, ts_rank_cd(tsv, to_tsquery('english', %s)) AS s
                FROM {self.schema}.chunks WHERE tsv @@ to_tsquery('english', %s){where} ORDER BY s DESC LIMIT %s""",
            [q, q, *params, k],
        ).fetchall()
        return self._chunks(rows)

    def vector_search(self, vector: np.ndarray, k: int, card_ids: list[str] | None = None) -> list[Chunk]:
        where, params = self._filter(card_ids)
        rows = self.conn.execute(
            f"""SELECT id, card_id, card_name, kind, text, url, title, as_of, 1 - (embedding <=> %s) AS s
                FROM {self.schema}.chunks WHERE true{where} ORDER BY embedding <=> %s LIMIT %s""",
            [np.asarray(vector, dtype=np.float32), *params, np.asarray(vector, dtype=np.float32), k],
        ).fetchall()
        return self._chunks(rows)

    def hybrid_search(self, query: str, vector: np.ndarray, k: int, card_ids: list[str] | None = None) -> list[Chunk]:
        """Vector and keyword ranks fused with RRF inside Postgres, in one round trip."""
        where, params = self._filter(card_ids)
        q = " | ".join(re.findall(r"[a-z0-9]+", query.lower())[:24]) or "nothing"
        vec = np.asarray(vector, dtype=np.float32)
        rows = self.conn.execute(
            f"""
            WITH vec AS (
              SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %s) AS rank
              FROM {self.schema}.chunks WHERE true{where} ORDER BY embedding <=> %s LIMIT 30
            ),
            kw AS (
              SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, to_tsquery('english', %s)) DESC) AS rank
              FROM {self.schema}.chunks WHERE tsv @@ to_tsquery('english', %s){where} LIMIT 30
            ),
            fused AS (
              SELECT id, SUM(1.0 / ({RRF_K} + rank)) AS score FROM (SELECT * FROM vec UNION ALL SELECT * FROM kw) AS both_lists
              GROUP BY id
            )
            SELECT c.id, c.card_id, c.card_name, c.kind, c.text, c.url, c.title, c.as_of, f.score
            FROM fused f JOIN {self.schema}.chunks c USING (id) ORDER BY f.score DESC LIMIT %s
            """,
            [vec, *params, vec, q, q, *params, k],
        ).fetchall()
        return self._chunks(rows)


def open_store(settings, country: str, dim: int) -> Store:
    if settings.store == "postgres":
        if not settings.database_url:
            raise ValueError("CARDPILOT_STORE=postgres needs CARDPILOT_DATABASE_URL.")
        return PostgresStore(settings.database_url, country, dim)
    return SQLiteStore(Path(settings.index_dir), country)


def describe(store: Store) -> str:
    return "postgres+pgvector" if isinstance(store, PostgresStore) else "sqlite+fts5"
