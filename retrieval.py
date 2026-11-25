"""Knowledge base: dense, keyword and hybrid search plus SQL over tabular files."""
import re
import sqlite3
import time
import unicodedata

import pandas as pd
from rank_bm25 import BM25Okapi

from .ingest.chunk import Chunk
from .ingest.embed import embed_query
from .ingest.extract import TABLE_TYPES, describe_table, load_tables
from .ingest.pipeline import ingest, source_files
from .ingest.store import VectorStore

MODES = ("hybrid", "semantic", "keyword")
RRF_K = 60
STOPWORDS = set("a an and are as at be by for from how in is it of on or the to was were what when which who with "
                "does do did can much many".split())


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))  # accents and Arabic diacritics/hamza
    words = re.findall(r"\w+", text.replace("ى", "ي").replace("ة", "ه"))
    return [w[:-1] if len(w) > 4 and w.endswith("s") and w.isascii() else w for w in words if w not in STOPWORDS]


class KnowledgeBase:
    def __init__(self, store: VectorStore | None = None, files: dict | None = None):
        self.store = store or VectorStore()
        self._files = files
        self.tables: dict[str, pd.DataFrame] = {}
        self.table_docs: dict[str, str] = {}
        self.reload()

    def refresh(self, log=print) -> dict:
        report = ingest(self.store, log, self._files)
        self.reload()
        return report

    def reload(self) -> None:
        self.chunks: list[Chunk] = self.store.all_chunks()
        self.by_id = {c.id: c for c in self.chunks}
        self.docs = {c.doc: c.title for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c.search_text) or ["_"] for c in self.chunks]) if self.chunks else None
        self.tables, self.table_docs = {}, {}
        files = source_files() if self._files is None else self._files
        for doc, path in files.items():
            if path.suffix.lower() in TABLE_TYPES and doc in self.docs:
                for name, df in load_tables(path).items():
                    self.tables[name], self.table_docs[name] = df, doc

    # --- text search ---------------------------------------------------------
    def search(self, query: str, mode: str = "hybrid", k: int = 4, docs: list[str] | None = None) -> list[Chunk]:
        if not self.chunks:
            return []
        pool = max(20, k)
        rankings = []
        if mode != "keyword":
            rankings.append([i for i, _ in self.store.query(embed_query(query), pool, docs)])
        if mode != "semantic":
            rankings.append(self._keyword(query, pool, docs))
        fused: dict[str, float] = {}
        for ranking in rankings:  # reciprocal rank fusion
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0) + 1 / (RRF_K + rank + 1)
        best = sorted(fused, key=fused.get, reverse=True)
        return [self.by_id[i] for i in best if i in self.by_id][:k]

    def _keyword(self, query: str, k: int, docs: list[str] | None) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(scores, self.chunks), key=lambda x: -x[0])
        return [c.id for s, c in ranked if s > 0 and (not docs or c.doc in docs)][:k]

    def resolve_doc(self, name: str) -> str | None:
        """Accept 'hr_handbook', 'HR Handbook' or 'hr_handbook.pdf'."""
        key = _norm(name)
        for doc, title in self.docs.items():
            if key in (_norm(doc), _norm(doc.rsplit(".", 1)[0]), _norm(title)):
                return doc
        partial = [d for d, t in self.docs.items() if key and (key in _norm(d) or key in _norm(t))]
        return partial[0] if len(partial) == 1 else None

    def document_chunks(self, doc: str) -> list[Chunk]:
        return [c for c in self.chunks if c.doc == doc]

    # --- tables --------------------------------------------------------------
    def schema(self, limit: int = 2400) -> str:
        return "\n".join(describe_table(n, df) for n, df in self.tables.items())[:limit]

    def sql(self, query: str, max_rows: int = 30) -> pd.DataFrame:
        """Run one read-only SELECT on an in-memory copy of the tables."""
        con = sqlite3.connect(":memory:")
        try:
            for name, df in self.tables.items():
                df.to_sql(name, con, index=False)
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                       getattr(sqlite3, "SQLITE_RECURSIVE", 33)}
            con.set_authorizer(lambda op, *_: sqlite3.SQLITE_OK if op in allowed else sqlite3.SQLITE_DENY)
            deadline = time.monotonic() + 5
            con.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
            cur = con.execute(query)
            if cur.description is None:
                raise ValueError("Only SELECT queries are allowed.")
            return pd.DataFrame(cur.fetchmany(max_rows), columns=[d[0] for d in cur.description])
        finally:
            con.close()


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()
