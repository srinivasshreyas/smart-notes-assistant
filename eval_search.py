"""Measure search quality. One command: python eval_search.py

Loads data/search_queries.json and the 12 starter notes, ranks the notes for every query with each
method, and prints hit rate @3 = share of queries where an expected note is in the top 3.
Also checks the low-relevance cutoff on queries that should match nothing.
"""
import json
from pathlib import Path

from app.search import METHODS, MIN_SCORE, rank, tokens

ROOT = Path(__file__).parent
# Extra probes, NOT part of the hit rate: written after seeing the first results, to look for failures.
PROBES = [("remember to call mom", "nothing"), ("something about money", "N002/N006/N010"),
          ("what is due this week?", "N005/N009"), ("how do I get to the airport?", "N003"),
          ("cheap flights", "N003"), ("computer accessories", "N004")]
NEGATIVE = ["bake a chocolate cake", "football match scores", "weather forecast tomorrow"]


def main() -> None:
    notes = json.loads((ROOT / "data" / "starter_notes.json").read_text(encoding="utf-8"))
    queries = json.loads((ROOT / "data" / "search_queries.json").read_text(encoding="utf-8"))
    by_id = {n["id"]: n["text"] for n in notes}

    no_share = [q for q in queries if not any(tokens(q["query"]) & tokens(by_id[i]) for i in q["expected"])]
    print(f"{len(queries)} queries over {len(notes)} starter notes; "
          f"{len(no_share)} share no content words with their target:")
    print("  " + " | ".join(q["query"] for q in no_share) + "\n")

    methods = []
    for m in METHODS:
        try:
            rank(m, "test", notes[:2])
            methods.append(m)
        except Exception as e:  # embedding model not installed / not downloadable
            print(f"[skipped] {m}: {type(e).__name__}: {str(e)[:80]}")

    results = {}
    for m in methods:
        hits, fails = 0, []
        hard_hits = 0
        for q in queries:
            top3 = [r["id"] for r in rank(m, q["query"], notes)[:3] if r["score"] > 0]  # zero score = not a match
            ok = any(i in top3 for i in q["expected"])
            hits += ok
            hard_hits += ok and q in no_share
            if not ok:
                fails.append((q["query"], q["expected"], top3))
        results[m] = (hits, hard_hits, fails)

    print(f"{'method':<12}{'hit@3':>8}{'(hits)':>9}{'hard-query hits':>18}")
    for m, (h, hh, _) in results.items():
        print(f"{m:<12}{h / len(queries):>8.2f}{f'({h}/{len(queries)})':>9}{f'{hh}/{len(no_share)}':>18}")

    print("\nFailing queries per method (query -> expected, got top 3):")
    for m, (_, _, fails) in results.items():
        print(f" {m}:")
        for q, exp, got in fails:
            print(f"   {q!r} -> expected {exp}, got {got}")

    print("\nLow-relevance cutoff check (unrelated queries should return nothing):")
    print(f"{'method':<12}{'cutoff':>8}{'unrelated queries with 0 results':>36}{'real queries still answered':>30}")
    for m in methods:
        cut = MIN_SCORE[m]
        silent = sum(1 for q in NEGATIVE if not [r for r in rank(m, q, notes) if r["score"] >= cut])
        answered = sum(1 for q in queries if [r for r in rank(m, q["query"], notes) if r["score"] >= cut])
        print(f"{m:<12}{cut:>8.2f}{f'{silent}/{len(NEGATIVE)}':>36}{f'{answered}/{len(queries)}':>30}")
    margins(methods, notes, queries)
    probes(methods, notes)


def probes(methods, notes) -> None:
    print("\nExtra probes (not counted in hit rate). Top-2 results above each method's cutoff:")
    for q, want in PROBES:
        print(f"  {q!r}  (a person would expect: {want})")
        for m in methods:
            got = [(r["id"], r["score"]) for r in rank(m, q, notes) if r["score"] >= MIN_SCORE[m]][:2]
            print(f"      {m:<11}{got if got else 'no results'}")


def margins(methods, notes, queries) -> None:
    """Help choose a cutoff: highest score of any unrelated query vs lowest top-1 score of a correct answer."""
    print("\nScore margins (cutoff should sit between the two numbers):")
    for m in methods:
        top_neg = max(rank(m, q, notes)[0]["score"] for q in NEGATIVE)
        good = [rank(m, q["query"], notes)[0] for q in queries]
        good = [g["score"] for g, q in zip(good, queries) if g["id"] in q["expected"] and g["score"] > 0]
        print(f"  {m:<12} highest unrelated-query score {top_neg:.2f} | lowest correct top-1 score {min(good, default=0):.2f}")


if __name__ == "__main__":
    main()
