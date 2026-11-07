"""Retrieval check: recall@k on the sample questions + distances for threshold calibration.

Usage: python eval.py [--top-k 4]
Prints, per question, whether the expected page is in the top-k and the best distance.
Out-of-corpus questions should have a best distance ABOVE the threshold.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from query import search, THRESHOLD

QUESTIONS = json.loads((Path(__file__).parent / "eval_questions.json").read_text())


def main(top_k: int):
    hits_ok, in_scope, worst_in, best_out = 0, 0, 0.0, 9.0
    for q in QUESTIONS:
        results = search(q["question"], top_k)
        best = min(d for _, _, d in results)
        pages = [(m["filename"], m["page"]) for _, m, _ in results]
        if q.get("expected"):
            in_scope += 1
            ok = (q["expected"]["filename"], q["expected"]["page"]) in pages
            hits_ok += ok
            worst_in = max(worst_in, best)
            status = "HIT " if ok else "MISS"
            print(f"{status} best={best:.3f}  {q['question']}  -> expected [{q['expected']['filename']}, p. {q['expected']['page']}]")
        else:
            best_out = min(best_out, best)
            status = "OK  " if best > THRESHOLD else "LEAK"
            print(f"{status} best={best:.3f}  {q['question']}  -> expected refusal")
    print(f"\nrecall@{top_k}: {hits_ok}/{in_scope}")
    print(f"worst best-distance of in-scope questions: {worst_in:.3f}")
    print(f"best distance of out-of-scope questions:  {best_out:.3f}")
    print(f"current RAG_MAX_DISTANCE: {THRESHOLD}  (should sit between the two numbers above)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=4)
    main(p.parse_args().top_k)
