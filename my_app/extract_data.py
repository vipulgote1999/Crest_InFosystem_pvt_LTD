from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain.schema import Document
import os

def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        if not documents:
            raise ValueError("No readable text found in PDF.")
        return documents
    except Exception as e:
        raise ValueError(f"Error loading PDF: {e}")

# 📊 CSV Loader using LangChain
def load_csv(file_path):
    try:
        loader = CSVLoader(file_path=file_path, encoding='utf-8', csv_args={'delimiter': ','})
        documents = loader.load()
        if not documents:
            raise ValueError("CSV is empty or not readable.")
        return documents
    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")

print(load_pdf(r"C:\Users\vg929494.ttl\Downloads\Crest_InFosystem_pvt_LTD-main\my_app\Offer Letter 2025-05-22.pdf"))