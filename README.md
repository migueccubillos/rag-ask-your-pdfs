# Ask Your PDFs — RAG with page-level citations

A small Python RAG app. It ingests a folder of PDFs, stores page-aware chunks in Chroma, retrieves the top-k passages with Sentence Transformers, and asks an LLM to answer **only** from those passages, citing each claim as `[filename, p. N]`. When the corpus does not contain the answer it replies: **"I don't know based on the provided documents."**

## Architecture

```
PDF → pypdf page text → overlapping chunks (600 chars, 90 overlap, never crossing a page)
    → all-MiniLM-L6-v2 embeddings → Chroma (cosine) with {filename, page, chunk} metadata

question → embedding → top-k search → distance threshold ──(nothing close)──> "I don't know…"
                                          │
                                          └─> LLM (answer only from passages, cite [file, p. N])
                                                └─> citation validation (flags citations not in the retrieved set)
```

| File | Purpose |
|---|---|
| `ingest.py` | PDF → text → chunks → embeddings → Chroma. Re-runnable: drops a file's old chunks before re-indexing. `--reset` rebuilds the index. |
| `query.py` | CLI. One question as an argument, or interactive mode with no argument. `--debug` prints retrieval distances. |
| `eval.py` + `eval_questions.json` | recall@k on 13 in-scope questions + 3 out-of-scope questions; helps calibrate the threshold. |
| `make_sample_pdfs.py` | Regenerates the 4 sample PDFs in `sample_pdfs/`. |

## Setup

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # or: export OPENAI_API_KEY=...
python ingest.py --pdf-dir sample_pdfs
python query.py "How much overlap should consecutive chunks share?"
python query.py                    # interactive mode
```

- The first run downloads `all-MiniLM-L6-v2` (~90 MB) from Hugging Face.
- Provider: `ANTHROPIC_API_KEY` is used if it is set, otherwise `OPENAI_API_KEY`. Change the model with `ANTHROPIC_MODEL` or `OPENAI_MODEL`.
- With no key at all, the CLI prints the retrieved passages with their citations instead of a generated answer, so retrieval can still be tested for free.
- `RAG_MAX_DISTANCE` (default `0.70`, cosine distance) is the refusal threshold, calibrated with `eval.py`. See *Calibration* below.

## Sample corpus

`sample_pdfs/` holds 4 non-sensitive manuals of 3 pages each, written for this project and generated with `make_sample_pdfs.py`. Each fact is on a known page, so citations can be checked by hand: `rag-basics.pdf`, `citations-and-evaluation.pdf`, `troubleshooting.pdf`, `vector-databases.pdf`. Some pages produce two overlapping chunks.

## Sample questions and expected citations

| Question | Expected citation |
|---|---|
| What are the two phases of a RAG system? | `[rag-basics.pdf, p. 1]` |
| How much overlap should consecutive chunks share? | `[rag-basics.pdf, p. 2]` |
| How can the app detect invented citations? | `[citations-and-evaluation.pdf, p. 2]` |
| How is retrieval quality measured with recall at k? | `[citations-and-evaluation.pdf, p. 3]` |
| Why would a scanned PDF produce no indexed text? | `[troubleshooting.pdf, p. 1]` |
| How do I calibrate the distance threshold? | `[troubleshooting.pdf, p. 3]` |
| How many dimensions do all-MiniLM-L6-v2 vectors have? | `[vector-databases.pdf, p. 1]` |
| What is the capital of France? | `I don't know based on the provided documents.` |

Example run (real output, Anthropic `claude-haiku-5-5`):

```
$ python query.py "How many dimensions do all-MiniLM-L6-v2 vectors have?"
All-MiniLM-L6-v2 vectors have 384 dimensions [vector-databases.pdf, p. 1].

Retrieved: [vector-databases.pdf, p. 1]
```

Out-of-corpus question (real output; refused by the distance threshold before the LLM is called):

```
$ python query.py "What is the capital of France?"
I don't know based on the provided documents.
```

All 8 questions in the table above were run against the real model: each answer cited its expected page, the out-of-corpus question returned the exact refusal sentence, and no unverified-citation warnings appeared.

## Calibration

```bash
python eval.py                 # recall@k + distance summary
python query.py "..." --debug  # per-question distances
```

Set `RAG_MAX_DISTANCE` between the worst best-distance of the in-scope questions and the best distance of the out-of-scope questions. Re-check it if the model or chunk size changes.

Measured on the sample corpus (`all-MiniLM-L6-v2`, top-k 4): recall@4 = 13/13, worst best-distance of in-scope questions = 0.559, best distance of out-of-scope questions = 0.855. The default `0.70` sits between them with a margin of about 0.14 on each side.

## Design notes

- **Page-local chunking:** a chunk never crosses a page, so every citation points to exactly one page. Chunks break on word boundaries.
- **Idempotent ingestion:** ids are `sha256(filename:page:chunk)`, and a file's old records are deleted before it is re-indexed, so edited or shortened PDFs leave no stale chunks.
- **Two-layer refusal:** the distance threshold runs before the LLM is called (cheaper, and the model cannot hallucinate from weak evidence), and the system prompt makes the model refuse when the passages are insufficient.
- **Citation validation:** every `[file, p. N]` in the answer is checked against the retrieved set. Unknown citations are flagged as unverified.
- **Limitations:** scanned PDFs need OCR first. Character-based chunking ignores headings and tables. MiniLM is English-centric, so a multilingual corpus needs another model (e.g. `paraphrase-multilingual-MiniLM-L12-v2`).
