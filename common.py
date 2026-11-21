"""Briques communes : embeddings, base vectorielle, recherche hybride."""
import os
import re
import unicodedata
from pathlib import Path

try:  # charge la configuration LLM (clé API, modèle...) depuis un fichier .env s'il existe
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DB_DIR = Path(os.environ.get("RAG_DB", "db"))
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

_embedder = None
_reranker = None
_bm25_cache = {}


def embed(texts):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder.encode(texts, normalize_embeddings=True, batch_size=16).tolist()


def collection():
    import chromadb
    client = chromadb.PersistentClient(str(DB_DIR))
    return client.get_or_create_collection("site", metadata={"hnsw:space": "cosine"})


def tokenize(text):
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"\w+", text)


def _bm25(col):
    from rank_bm25 import BM25Okapi
    n = col.count()
    if _bm25_cache.get("n") != n:
        data = col.get(include=["documents", "metadatas"])
        _bm25_cache.update(
            n=n, ids=data["ids"], docs=data["documents"], metas=data["metadatas"],
            index=BM25Okapi([tokenize(d) for d in data["documents"]]),
        )
    return _bm25_cache


def search(query, k=6, rerank=False, pool=30):
    """Recherche hybride (vecteurs + BM25, fusion RRF), reranker optionnel."""
    col = collection()
    if col.count() == 0:
        raise SystemExit("Index vide : lancez d'abord index.py")

    vec = col.query(query_embeddings=[embed([query])[0]], n_results=min(pool, col.count()))
    items = {}
    scores = {}
    for rank, (i, d, m) in enumerate(zip(vec["ids"][0], vec["documents"][0], vec["metadatas"][0])):
        items[i] = {"id": i, "text": d, "meta": m}
        scores[i] = scores.get(i, 0) + 1 / (60 + rank)

    bm = _bm25(col)
    raw = bm["index"].get_scores(tokenize(query))
    top = sorted(range(len(raw)), key=lambda j: raw[j], reverse=True)[:pool]
    for rank, j in enumerate(top):
        if raw[j] <= 0:
            break
        i = bm["ids"][j]
        items.setdefault(i, {"id": i, "text": bm["docs"][j], "meta": bm["metas"][j]})
        scores[i] = scores.get(i, 0) + 1 / (60 + rank)

    ranked = [items[i] for i in sorted(scores, key=scores.get, reverse=True)]

    if rerank:
        global _reranker
        if _reranker is None:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(RERANK_MODEL)
        cand = ranked[:pool]
        ce = _reranker.predict([(query, c["text"]) for c in cand])
        ranked = [c for _, c in sorted(zip(ce, cand), key=lambda x: x[0], reverse=True)]

    return ranked[:k]
