"""Search saved notes. Phase 4 version: TF-IDF cosine similarity with a low-relevance cutoff."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MIN_SCORE = 0.10  # below this a note is not shown as a match


def search_tfidf(query: str, notes: list[dict], top_k: int = 5, min_score: float = MIN_SCORE) -> list[dict]:
    if not notes or not query.strip():
        return []
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        matrix = vec.fit_transform([n["text"] for n in notes])
    except ValueError:  # no usable words in the notes
        return []
    scores = cosine_similarity(vec.transform([query]), matrix)[0]
    order = scores.argsort()[::-1][:top_k]
    return [{**notes[i], "score": round(float(scores[i]), 3)} for i in order if scores[i] >= min_score]
