import fitz  # PyMuPDF
import pandas as pd

def load_pdf_chunks(filepath):
    doc = fitz.open(filepath)
    chunks = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            chunks.append(text)
    return chunks

def load_csv_chunks(filepath):
    df = pd.read_csv(filepath)
    chunks = df.astype(str).agg(" ".join, axis=1).tolist()
    return chunks
