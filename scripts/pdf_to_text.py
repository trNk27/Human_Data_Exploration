"""Convert a PDF file to a plain text file using available libraries."""

import sys
import pathlib

PDF_PATH = pathlib.Path("papers/ZETA.pdf")
OUT_PATH = pathlib.Path("papers/ZETA.txt")


def try_pdfplumber(path):
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append(f"=== Page {i+1} ===\n{text}")
    return "\n\n".join(pages)


def try_pymupdf(path):
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append(f"=== Page {i+1} ===\n{text}")
    return "\n\n".join(pages)


def try_pypdf2(path):
    import PyPDF2
    pages = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(f"=== Page {i+1} ===\n{text}")
    return "\n\n".join(pages)


def try_pdfminer(path):
    from pdfminer.high_level import extract_text
    return extract_text(str(path))


extractors = [
    ("pdfplumber", try_pdfplumber),
    ("pymupdf (fitz)", try_pymupdf),
    ("PyPDF2", try_pypdf2),
    ("pdfminer", try_pdfminer),
]

text = None
for name, fn in extractors:
    try:
        print(f"Trying {name}...")
        text = fn(PDF_PATH)
        print(f"Success with {name}. Characters extracted: {len(text)}")
        break
    except ImportError:
        print(f"  {name} not installed, skipping.")
    except Exception as e:
        print(f"  {name} failed: {e}")

if text:
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"Saved to {OUT_PATH}")
else:
    print("All extractors failed. Please install one of: pdfplumber, pymupdf, PyPDF2, pdfminer.six")
    sys.exit(1)
