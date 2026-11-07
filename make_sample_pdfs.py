"""Generate the sample corpus in sample_pdfs/ (4 multi-page, non-sensitive manuals).

Each fact lives on a specific page so page-level citations can be verified.
Run: python make_sample_pdfs.py
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

OUT = Path(__file__).parent / "sample_pdfs"

DOCS = {
    "rag-basics.pdf": ("RAG Basics: A Short Manual", [
        ("What retrieval augmented generation is",
         "Retrieval augmented generation (RAG) combines a search step over a trusted document collection "
         "with a large language model. Instead of relying on what the model memorised during training, the "
         "application first retrieves the passages most relevant to the user's question and then asks the "
         "model to answer using only those passages. This keeps answers grounded in documents that the "
         "operator controls and can update without retraining the model. "
         "A RAG system has two phases. The offline phase, called ingestion, reads the source files, splits "
         "them into chunks, converts each chunk into an embedding vector and stores the vectors in a vector "
         "database. The online phase, called querying, embeds the question, finds the nearest chunks and "
         "builds a prompt that contains the question together with the retrieved evidence."),
        ("Chunking and overlap",
         "Documents are split into chunks because embedding models have a limited input length and because "
         "smaller passages produce more precise matches. A common starting point is a chunk of roughly 500 "
         "to 1000 characters. Consecutive chunks should share an overlap of about 10 to 15 percent of the "
         "chunk size, for example 90 characters for a 600 character chunk. The overlap prevents a sentence "
         "that falls on a boundary from being cut in half and lost to retrieval. "
         "Chunking should happen inside a single page when exact page citations are required. If a chunk "
         "crossed a page break, the system could not say on which page the supporting sentence appears. "
         "Very short pages produce a single chunk, while long pages produce several overlapping chunks that "
         "all keep the same page number in their metadata."),
        ("Refusal policy",
         "A grounded assistant must be allowed to refuse. When none of the retrieved passages is close "
         "enough to the question, or when the passages do not contain the requested fact, the assistant "
         "should reply with a fixed refusal sentence rather than guessing. The refusal sentence used by "
         "this project is: I don't know based on the provided documents. "
         "Two layers implement the policy. First, a distance threshold discards weak matches before the "
         "language model is called, which also saves cost. Second, the system prompt instructs the model "
         "to use the refusal sentence whenever the remaining evidence is insufficient."),
    ]),
    "citations-and-evaluation.pdf": ("Citations and Evaluation Guide", [
        ("Why page metadata matters",
         "Every chunk stored in the vector database should carry the name of its source file and the page "
         "number it came from. This metadata is what allows the assistant to print a citation next to each "
         "claim and allows a reader to open the original file at the right page to verify the claim. "
         "Without page metadata a citation can only point to a whole document, which makes verification "
         "slow and discourages readers from checking the answer at all."),
        ("Citation format and validation",
         "Citations follow a stable pattern made of the file name, a comma, the abbreviation p. and the "
         "page number, all wrapped in square brackets. A stable pattern can be parsed with a regular "
         "expression after generation. "
         "Language models occasionally invent citations or copy a reference that never appeared in the "
         "evidence. For that reason the application should validate the answer: every citation found in "
         "the generated text must match one of the file and page pairs that were actually retrieved. "
         "Citations that fail this check are reported to the user as unverified."),
        ("Evaluating retrieval and generation separately",
         "Retrieval and generation fail in different ways, so they should be evaluated separately. "
         "Retrieval quality is measured with recall at k: for a set of test questions with known answer "
         "pages, count how often the correct page appears among the top k results. "
         "Generation quality is measured by faithfulness: whether every statement in the answer is "
         "supported by the retrieved passages. A third check covers refusals: questions whose answer is "
         "not in the corpus must produce the refusal sentence. A small set of ten to twenty hand written "
         "questions is enough to compare chunk sizes, thresholds and models."),
    ]),
    "troubleshooting.pdf": ("Troubleshooting a Document Assistant", [
        ("Empty or poor answers",
         "If the assistant returns no useful answer, first confirm that the PDFs were parsed. Scanned "
         "documents contain images instead of text, so the extractor returns empty strings and nothing is "
         "indexed. Such files need optical character recognition before ingestion. "
         "Next confirm that the vector collection actually contains records and that the count grows "
         "after ingestion. A collection that is empty at query time usually means ingestion wrote to a "
         "different database directory than the one the query script reads."),
        ("Unrelated passages are retrieved",
         "When retrieval returns passages that have nothing to do with the question, check that the same "
         "embedding model is used for ingestion and for queries. Mixing two models produces vectors in "
         "different spaces and the nearest neighbours become meaningless. "
         "Also inspect the chunk size. Chunks that are too large mix several topics and match many "
         "questions weakly, while chunks that are too small lose context. Rephrasing a vague question with "
         "the specific terms used in the documents often improves results."),
        ("Calibrating the distance threshold",
         "The distance threshold decides when the assistant refuses. To calibrate it, run the test "
         "questions with a debug flag that prints the distance of every retrieved chunk. Record the "
         "distances of correct matches and the distances returned for questions that are out of scope. "
         "Place the threshold between the two groups. Lower the threshold only after reviewing examples, "
         "because a strict threshold refuses questions the corpus can answer, while a loose threshold lets "
         "unrelated passages reach the model. Re-check the threshold whenever the embedding model or the "
         "chunk size changes."),
    ]),
    "vector-databases.pdf": ("Embeddings and Vector Databases", [
        ("Embeddings",
         "An embedding model maps a piece of text to a fixed length vector of numbers so that texts with "
         "similar meaning end up close to each other. The model all-MiniLM-L6-v2 from the Sentence "
         "Transformers library produces vectors with 384 dimensions and is small enough to run on a laptop "
         "without a graphics card. Its input limit is about 256 word pieces, so longer text is truncated."),
        ("Distance metrics",
         "Vector databases rank results by distance. Cosine distance equals one minus the cosine "
         "similarity of the two vectors. A distance near zero means the texts are almost identical in "
         "meaning, while a distance near one means they are unrelated. Because cosine distance ignores "
         "vector length, it is the usual choice for sentence embeddings."),
        ("Persistence with Chroma",
         "Chroma is an open source vector database that can run embedded inside a Python process. A "
         "persistent client stores the collection on disk in a directory, so the index survives restarts "
         "and ingestion only needs to run again when documents change. "
         "Each record holds an id, the chunk text, its embedding and a metadata dictionary. Using a "
         "deterministic id built from file name, page and chunk index lets re-ingestion update records in "
         "place instead of creating duplicates. Records of a file should be deleted before the file is "
         "re-indexed, otherwise chunks from an older version can remain in the collection."),
    ]),
}


def build():
    OUT.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    for name, (title, pages) in DOCS.items():
        story = []
        for i, (heading, body) in enumerate(pages):
            if i == 0:
                story.append(Paragraph(title, styles["Title"]))
            story += [Paragraph(heading, styles["Heading2"]), Spacer(1, 6), Paragraph(body, styles["BodyText"])]
            if i < len(pages) - 1:
                story.append(PageBreak())
        SimpleDocTemplate(str(OUT / name), pagesize=A4, title=title).build(story)
        print("wrote", OUT / name)


if __name__ == "__main__":
    build()
