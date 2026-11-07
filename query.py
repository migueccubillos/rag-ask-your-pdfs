"""Ask a question over the indexed PDFs and get an answer with [filename, p. N] citations.

Usage:
  python query.py "your question" [--top-k 4] [--debug]
  python query.py            # interactive mode

LLM provider (first key found wins): ANTHROPIC_API_KEY or OPENAI_API_KEY.
Without a key the CLI prints the retrieved evidence with citations instead of an answer.
"""
from __future__ import annotations
import argparse, os, re
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION = "pdf_pages"
MODEL_NAME = "all-MiniLM-L6-v2"
THRESHOLD = float(os.getenv("RAG_MAX_DISTANCE", "0.70"))  # cosine distance; calibrated with eval.py (in-scope <= 0.56, out-of-scope >= 0.85)
REFUSAL = "I don't know based on the provided documents."
CITATION_RE = re.compile(r"\[([^\[\],]+?\.pdf),\s*p\.\s*(\d+)\]", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You answer questions using ONLY the supplied passages. Each passage starts with its source tag "
    "in the form [filename, p. N]. After every factual sentence, cite the passage(s) you used with that "
    "exact tag, e.g. [rag-basics.pdf, p. 2]. Never cite a tag that is not in the passages. "
    f"If the passages do not contain the answer, reply exactly: {REFUSAL}"
)

_model = None


def embed(text: str):
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model.encode(text).tolist()


def search(question: str, top_k: int = 4):
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    if collection.count() == 0:
        raise SystemExit("Corpus is empty. Run: python ingest.py --pdf-dir sample_pdfs")
    res = collection.query(query_embeddings=[embed(question)], n_results=min(top_k, collection.count()),
                           include=["documents", "metadatas", "distances"])
    return list(zip(res["documents"][0], res["metadatas"][0], (float(d) for d in res["distances"][0])))


def call_llm(question: str, context: str) -> str:
    user = f"Question: {question}\n\nPassages:\n{context}"
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"), max_tokens=600,  # SDK 2.x has no temperature param
            system=SYSTEM_PROMPT, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if b.type == "text").strip()
    from openai import OpenAI
    resp = OpenAI().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])
    return resp.choices[0].message.content.strip()


def validate_citations(answer: str, evidence) -> list[str]:
    """Return citations in the answer that do not match any retrieved (filename, page)."""
    allowed = {(m["filename"].lower(), int(m["page"])) for _, m, _ in evidence}
    return [f"[{f}, p. {p}]" for f, p in CITATION_RE.findall(answer) if (f.strip().lower(), int(p)) not in allowed]


def answer(question: str, top_k: int = 4, debug: bool = False) -> str:
    hits = search(question, top_k)
    if debug:
        print(f"[debug] threshold={THRESHOLD}")
        for _, m, d in hits:
            print(f"[debug] {d:.3f}  {m['filename']} p.{m['page']} chunk {m['chunk']}"
                  f"  {'KEEP' if d <= THRESHOLD else 'drop'}")
    evidence = [h for h in hits if h[2] <= THRESHOLD]
    if not evidence:
        return REFUSAL
    context = "\n\n".join(f"[{m['filename']}, p. {m['page']}] {d}" for d, m, _ in evidence)
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")):
        return "No LLM key set (ANTHROPIC_API_KEY or OPENAI_API_KEY). Retrieved evidence:\n\n" + context
    text = call_llm(question, context)
    if REFUSAL.lower().rstrip(".") in text.lower():
        return REFUSAL
    if not CITATION_RE.search(text):
        text += "\n\n(Warning: the model gave no citation; treat this answer as unverified.)"
    bad = validate_citations(text, evidence)
    if bad:
        text += "\n\n(Warning: unverified citations not found in retrieved passages: " + ", ".join(bad) + ")"
    sources = sorted({f"[{m['filename']}, p. {m['page']}]" for _, m, _ in evidence})
    return text + "\n\nRetrieved: " + " ".join(sources)


def main():
    p = argparse.ArgumentParser(description="Ask questions over an indexed PDF corpus")
    p.add_argument("question", nargs="?", help="omit for interactive mode")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--debug", action="store_true", help="print retrieval distances")
    a = p.parse_args()
    if a.question:
        print(answer(a.question, a.top_k, a.debug))
        return
    print("Ask a question (empty line to quit).")
    while (q := input("\n> ").strip()):
        print(answer(q, a.top_k, a.debug))


if __name__ == "__main__":
    main()
