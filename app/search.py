"""Search saved notes with three interchangeable methods.

keyword    - baseline: count shared words (stop words removed). Cannot match different words.
tfidf      - word TF-IDF cosine similarity.
tfidf_char - character 3-5 gram TF-IDF cosine (tolerates word-form differences).
embedding  - sentence embeddings (all-MiniLM-L6-v2): matches MEANING. Needs sentence-transformers.

Every function returns notes ranked best-first with a score; a cutoff (min_score) hides weak matches.
"""
import re
from functools import lru_cache

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

METHODS = ["keyword", "tfidf", "tfidf_char", "embedding"]
# Minimum score for a note to count as a match. Chosen with eval_search.py (see RESULTS.md).
MIN_SCORE = {"keyword": 1.0, "tfidf": 0.10, "tfidf_char": 0.15, "embedding": 0.27}
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if w not in ENGLISH_STOP_WORDS}


def _scores_keyword(query: str, notes: list[dict]) -> list[float]:
    q = tokens(query)
    return [float(len(q & tokens(n["text"]))) for n in notes]


def without_stop_words(text: str) -> str:
    return " ".join(w for w in re.findall(r"\w+", text.lower()) if w not in ENGLISH_STOP_WORDS)


def _scores_tfidf(query: str, notes: list[dict], clean=False, **kw) -> list[float]:
    prep = without_stop_words if clean else (lambda t: t)
    vec = TfidfVectorizer(sublinear_tf=True, **kw)
    try:
        matrix = vec.fit_transform([prep(n["text"]) for n in notes])
    except ValueError:
        return [0.0] * len(notes)
    return list(cosine_similarity(vec.transform([prep(query)]), matrix)[0])


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer  # imported only when needed
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=256)
def _embed_note(text: str):
    return _embedder().encode([text], normalize_embeddings=True)[0]


def _scores_embedding(query: str, notes: list[dict]) -> list[float]:
    q = _embedder().encode([query], normalize_embeddings=True)[0]
    return [float(q @ _embed_note(n["text"])) for n in notes]  # notes embedded once, cached


def scores(method: str, query: str, notes: list[dict]) -> list[float]:
    if method == "keyword":
        return _scores_keyword(query, notes)
    if method == "tfidf":
        return _scores_tfidf(query, notes, stop_words="english", ngram_range=(1, 2))
    if method == "tfidf_char":
        return _scores_tfidf(query, notes, clean=True, analyzer="char_wb", ngram_range=(3, 5))
    if method == "embedding":
        return _scores_embedding(query, notes)
    raise ValueError(f"unknown method {method}")


def rank(method: str, query: str, notes: list[dict]) -> list[dict]:
    """All notes, best first, each with its score (no cutoff)."""
    if not notes or not query.strip():
        return []
    sc = scores(method, query, notes)
    order = sorted(range(len(notes)), key=lambda i: (-sc[i], notes[i]["id"]))
    return [{**notes[i], "score": round(float(sc[i]), 3)} for i in order]


def search(query: str, notes: list[dict], method: str | None = None, top_k: int = 5):
    """Used by the app. Returns (method_used, matches_above_cutoff). Falls back to character TF-IDF if the
    embedding model cannot be loaded (for example no internet on first run)."""
    method = method or DEFAULT_METHOD
    try:
        ranked = rank(method, query, notes)
    except Exception:
        method = "tfidf_char"
        ranked = rank(method, query, notes)
    return method, [r for r in ranked if r["score"] >= MIN_SCORE[method]][:top_k]


DEFAULT_METHOD = "embedding"  # best hit rate at 3; falls back to tfidf_char if it cannot load
