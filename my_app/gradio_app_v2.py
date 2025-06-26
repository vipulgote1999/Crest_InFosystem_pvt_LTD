import os
import pdfplumber
import mimetypes
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
import os

# 🔑 Set your Hugging Face API Token
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

vector_db = None  # Global state

# Loaders
def load_pdf(file_path):
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                docs.append(Document(page_content=text, metadata={"page": i + 1}))
    return docs

def load_csv(file_path):
    try:
        loader = CSVLoader(file_path=file_path)
        return loader.load()
    except Exception as e:
        print("[❌ ERROR] Failed to load CSV:", e)
        return []

def load_file(file_path):
    mimetype, _ = mimetypes.guess_type(file_path)
    if file_path.lower().endswith('.csv') or (mimetype and 'csv' in mimetype):
        return load_csv(file_path)
    elif file_path.lower().endswith('.pdf'):
        return load_pdf(file_path)
    else:
        raise ValueError("Unsupported file format. Only PDF and CSV are supported.")

# Splitting & Embedding
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def build_vectorstore(docs):
    embedder = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )
    return FAISS.from_documents(docs, embedder)

# Prompt Template
prompt_template = """You are an assistant with access to the following context:

{context}

Answer the following question:
{question}

Return:
- A detailed explanation
- A concise 2-3 line summary"""
prompt = PromptTemplate.from_template(prompt_template)

# LLM Setup
llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    temperature=0.3,
    max_new_tokens=4096,
    stop_sequences=["<|im_start|>", "<|im_end|>"],
    provider="auto"
)
chat_llm = ChatHuggingFace(llm=llm_endpoint)

# RAG chain
rag_chain = RunnableMap({
    "context": lambda inp: inp["context"],
    "question": lambda inp: inp["question"]
}) | prompt | chat_llm

# Indexing step
def index_document(file):
    global vector_db
    if file is None:
        return "⚠️ Please upload a file before indexing."
    try:
        docs = load_file(file.name)
        chunks = split_docs(docs)
        vector_db = build_vectorstore(chunks)
        return "✅ Vector DB created successfully!"
    except Exception as e:
        return f"❌ Indexing failed: {e}"

def answer_question(question):
    global vector_db
    if vector_db is None:
        return "⚠️ Please index a file first."
    if not question.strip():
        return "⚠️ Enter a valid question."
    try:
        relevant_docs = vector_db.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        response = rag_chain.invoke({"context": context, "question": question})
        answer = response.content if hasattr(response, "content") else str(response)

        sources = "\n---\n".join([doc.page_content[:500] for doc in relevant_docs])
        return f"### 📘 Answer\n{answer}\n\n---\n### 📄 Top Matching Contexts:\n{sources}"
    except Exception as e:
        return f"❌ Error during QA: {e}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 RAG-powered PDF/CSV Q&A Chatbot")

    file_input = gr.File(label="📄 Upload PDF or CSV", file_types=[".pdf", ".csv"])
    index_btn = gr.Button("📚 Index Document")
    index_output = gr.Textbox(label="🧠 Indexing Status")

    question_input = gr.Textbox(label="❓ Ask a Question", placeholder="e.g., What is the offer CTC?")
    answer_btn = gr.Button("💬 Get Answer")
    output = gr.Textbox(label="📘 Answer", lines=8)

    index_btn.click(fn=index_document, inputs=[file_input], outputs=index_output)
    answer_btn.click(fn=answer_question, inputs=[question_input], outputs=output)

if __name__ == "__main__":
    demo.launch()
