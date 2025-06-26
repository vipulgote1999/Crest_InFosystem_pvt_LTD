import os
import pdfplumber
import mimetypes
import json
import hashlib
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableMap
from langchain_huggingface import ChatHuggingFace
import gradio as gr

HUGGINGFACEHUB_API_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

vector_db = None  # Global state
cache_dir = "vector_cache"
os.makedirs(cache_dir, exist_ok=True)
log_file = "rag_log.jsonl"
blocked_keywords = {"hack", "bomb", "kill", "exploit", "terrorist"}  # Basic keyword filter

def load_pdf(file_path):
    docs = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    docs.append(Document(page_content=text, metadata={"page": i + 1}))
    except Exception as e:
        raise ValueError(f"PDF load error: {e}")
    return docs

def load_csv(file_path):
    try:
        loader = CSVLoader(file_path=file_path)
        return loader.load()
    except Exception as e:
        raise ValueError(f"CSV load error: {e}")

def load_file(file_path):
    mimetype, _ = mimetypes.guess_type(file_path)
    if file_path.lower().endswith('.csv') or (mimetype and 'csv' in mimetype):
        return load_csv(file_path)
    elif file_path.lower().endswith('.pdf'):
        return load_pdf(file_path)
    else:
        raise ValueError("Unsupported file format. Only PDF and CSV are supported.")

def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def cache_path(file):
    hash_val = hashlib.md5(open(file, 'rb').read()).hexdigest()
    return os.path.join(cache_dir, f"{hash_val}.faiss")

def build_vectorstore(docs, file_path):
    cache_fp = cache_path(file_path)
    if os.path.exists(cache_fp):
        return FAISS.load_local(cache_fp, HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
        ))
    embedder = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )
    db = FAISS.from_documents(docs, embedder)
    db.save_local(cache_fp)
    return db

prompt_template = """You are an assistant with access to the following context:

{context}

Answer the following question:
{question}

Return:
- A detailed explanation
- A concise 2-3 bullet point summary"""
prompt = PromptTemplate.from_template(prompt_template)

llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    temperature=0.3,
    max_new_tokens=4096,
    stop_sequences=["<|im_start|>", "<|im_end|>"],
    provider="auto"
)
chat_llm = ChatHuggingFace(llm=llm_endpoint)

rag_chain = RunnableMap({
    "context": lambda inp: inp["context"],
    "question": lambda inp: inp["question"]
}) | prompt | chat_llm

def index_document(file):
    global vector_db
    if file is None:
        return "⚠️ Please upload a file before indexing."
    try:
        docs = load_file(file.name)
        chunks = split_docs(docs)
        vector_db = build_vectorstore(chunks, file.name)
        return "✅ Vector DB created successfully!"
    except Exception as e:
        return f"❌ Indexing failed: {e}"

def filter_inappropriate(output):
    lower_output = output.lower()
    for word in blocked_keywords:
        if word in lower_output:
            return "⚠️ Inappropriate content detected."
    return output

def log_interaction(question, context, response):
    with open(log_file, "a", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "context": context,
            "response": response
        }, f)
        f.write("\n")

def answer_question(question):
    global vector_db
    if vector_db is None:
        return "⚠️ Please index a file first."
    if not question.strip():
        return "⚠️ Enter a valid question."
    try:
        relevant_docs = vector_db.similarity_search(question, k=2)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        response = rag_chain.invoke({"context": context, "question": question})
        answer = response.content if hasattr(response, "content") else str(response)
        answer = filter_inappropriate(answer)

        log_interaction(question, context, answer)
        sources = "\n---\n".join([doc.page_content[:300] for doc in relevant_docs])
        return f"### 📘 Answer\n{answer}\n\n---\n### 📄 Top Matching Contexts:\n{sources}"
    except Exception as e:
        return f"❌ Error during QA: {e}"

with gr.Blocks() as demo:
    gr.Markdown("## 🔍 RAG-powered PDF/CSV Q&A Chatbot")

    file_input = gr.File(label="📄 Upload PDF or CSV", file_types=[".pdf", ".csv"])
    index_btn = gr.Button("📚 Index Document")
    index_output = gr.Textbox(label="🧠 Indexing Status")

    question_input = gr.Textbox(label="❓ Ask a Question", placeholder="e.g., provide me summary of document provided?")
    answer_btn = gr.Button("💬 Get Answer")
    output = gr.Textbox(label="📘 Answer", lines=8)

    index_btn.click(fn=index_document, inputs=[file_input], outputs=index_output)
    answer_btn.click(fn=answer_question, inputs=[question_input], outputs=output)

if __name__ == "__main__":
    demo.launch()
