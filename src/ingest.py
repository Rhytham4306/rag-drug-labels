"""
Load drug label documents (.txt or .pdf) and split them into retrieval-ready
chunks.

Drug labels (FDA Structured Product Labeling) are naturally organized into
named sections: INDICATIONS AND USAGE, DOSAGE AND ADMINISTRATION, WARNINGS,
CONTRAINDICATIONS, ADVERSE REACTIONS, etc. Splitting on those boundaries
*first*, then recursively by character count, keeps each chunk topically
coherent — a chunk about dosage never bleeds into a chunk about side effects.
This matters a lot for retrieval precision on a document type this
structured.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import settings

# Common SPL / drug-label section headers (uppercase, ALL CAPS in most labels)
SECTION_HEADER_RE = re.compile(
    r"^\s*("
    r"INDICATIONS AND USAGE|INDICATIONS|"
    r"DOSAGE AND ADMINISTRATION|DOSAGE|"
    r"CONTRAINDICATIONS|"
    r"WARNINGS AND PRECAUTIONS|WARNINGS|PRECAUTIONS|"
    r"ADVERSE REACTIONS|SIDE EFFECTS|"
    r"DRUG INTERACTIONS|"
    r"USE IN SPECIFIC POPULATIONS|"
    r"OVERDOSAGE|"
    r"CLINICAL PHARMACOLOGY|"
    r"HOW SUPPLIED|STORAGE AND HANDLING|"
    r"PATIENT COUNSELING INFORMATION|"
    r"DESCRIPTION"
    r")\s*$",
    re.MULTILINE,
)


@dataclass
class Chunk:
    text: str
    source: str          # filename / drug name
    section: str          # e.g. "DOSAGE AND ADMINISTRATION"
    chunk_id: str = field(default="")

    def to_metadata(self) -> dict:
        return {"source": self.source, "section": self.section, "chunk_id": self.chunk_id}


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return read_pdf_file(path)
    return read_text_file(path)


def split_into_sections(raw_text: str) -> list[tuple[str, str]]:
    """Split raw label text into (section_name, section_text) pairs.

    Falls back to a single 'FULL DOCUMENT' section if no headers are found,
    so unstructured documents still work, just without the section-aware
    boost.
    """
    matches = list(SECTION_HEADER_RE.finditer(raw_text))
    if not matches:
        return [("FULL DOCUMENT", raw_text)]

    sections = []
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()
        if body:
            sections.append((name, body))
    return sections


def chunk_document(path: Path) -> list[Chunk]:
    raw_text = load_document(path)
    source_name = path.stem

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for section_name, section_text in split_into_sections(raw_text):
        pieces = splitter.split_text(section_text)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    text=piece,
                    source=source_name,
                    section=section_name,
                    chunk_id=f"{source_name}::{section_name}::{i}",
                )
            )
    return chunks


def load_and_chunk_directory(docs_dir: str | Path) -> list[Chunk]:
    docs_dir = Path(docs_dir)
    all_chunks: list[Chunk] = []
    file_paths = sorted(
        [p for p in docs_dir.glob("**/*") if p.suffix.lower() in {".txt", ".pdf"}]
    )
    if not file_paths:
        raise FileNotFoundError(
            f"No .txt or .pdf files found in {docs_dir}. "
            "Run scripts/fetch_dailymed.py first, or point at data/sample_docs."
        )
    for path in file_paths:
        all_chunks.extend(chunk_document(path))
    return all_chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview chunking of a docs directory")
    parser.add_argument("--docs", default="data/sample_docs")
    args = parser.parse_args()

    chunks = load_and_chunk_directory(args.docs)
    print(f"Loaded {len(chunks)} chunks from {args.docs}\n")
    for c in chunks[:5]:
        print(f"--- {c.chunk_id} ---")
        print(c.text[:200].replace("\n", " "), "...\n")
