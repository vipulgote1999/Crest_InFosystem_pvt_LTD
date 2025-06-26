import argparse
from utils.loaders import load_pdf_chunks, load_csv_chunks
from utils.retriever import retrieve_chunks
from utils.llm_api import query_huggingface_model
from utils.logger import log_interaction
import os
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

def build_prompt(context, question):
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

def summarize_text(text):
    return query_huggingface_model(f"Summarize:\n{text}", model="google/flan-t5-base", token=HUGGINGFACEHUB_API_TOKEN)

def main(filepath, question, filetype):
    chunks = load_pdf_chunks(filepath) if filetype == "pdf" else load_csv_chunks(filepath)
    top_chunks = retrieve_chunks(question, chunks)
    context = "\n\n".join(top_chunks)
    
    prompt = build_prompt(context, question)
    answer = query_huggingface_model(prompt, token="xxxxxxxxxxxxxxxx")
    summary = summarize_text(answer)

    print("\n📝 Detailed Answer:\n", answer)
    print("\n🔹 Summary:\n", summary)
    
    log_interaction(question, top_chunks, answer, summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--type", type=str, choices=["pdf", "csv"], required=True)
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()
    
    main(args.file, args.question, args.type)
