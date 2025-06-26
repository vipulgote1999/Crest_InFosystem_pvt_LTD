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
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os


# 🔑 Set your Hugging Face API Token
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# 1. Load PDF using pdfplumber
def load_pdf(file_path):
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                docs.append(Document(page_content=text, metadata={"page": i + 1}))
    return docs

# 1b. Load CSV using LangChain
def load_csv(file_path):
    try:
        loader = CSVLoader(file_path=file_path)
        return loader.load()
    except Exception as e:
        print("[❌ ERROR] Failed to load CSV:", e)
        return []

# 1c. Unified loader
def load_file(file_path):
    mimetype, _ = mimetypes.guess_type(file_path)
    if file_path.lower().endswith('.csv') or (mimetype and 'csv' in mimetype):
        return load_csv(file_path)
    elif file_path.lower().endswith('.pdf'):
        return load_pdf(file_path)
    else:
        raise ValueError("Unsupported file format. Only PDF and CSV are supported.")

# 2. Split documents
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# 3. Build FAISS vector store
def build_vectorstore(docs):
    try:
        embedder = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        )
        db = FAISS.from_documents(docs, embedder)
        return db
    except Exception as e:
        print("[❌ ERROR] Failed to build vectorstore:", e)
        raise

# 4. Define prompt
prompt_template = """You are an assistant with access to the following context:

{context}

Answer the following question:
{question}

Return:
- A detailed explanation
- A concise 2-3 line summary"""
prompt = PromptTemplate.from_template(prompt_template)

# 5. Define the LLM endpoint
llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    temperature=0.3,
    max_new_tokens=4096,
    stop_sequences=["<|im_start|>", "<|im_end|>"],
    provider="auto"
)

chat_llm = ChatHuggingFace(llm=llm_endpoint)

# 6. RAG chain
rag_chain = RunnableMap({
    "context": lambda inp: inp["context"],
    "question": lambda inp: inp["question"]
}) | prompt | chat_llm

# 7. Full RAG QA Function
def rag_qa(file_path, question, k=3):
    docs = load_file(file_path)
    chunks = split_docs(docs)
    db = build_vectorstore(chunks)
    relevant_docs = db.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    try:
        response = rag_chain.invoke({"context": context, "question": question})
        return response
    except Exception as e:
        print("[❌ ERROR] LLM invocation failed:", e)
        return "Error in LLM generation."

# 8. Run the pipeline
if __name__ == "__main__":
    # pdf_path = r"C:\path\to\document.pdf"
    file_path = r"C:\Users\vg929494.ttl\Downloads\Crest_InFosystem_pvt_LTD-main\my_app\sample_data.csv"  # PDF or CSV
    question = "What is the total offer ctc amount offer to vipul gote?"
    answer = rag_qa(file_path, question)
    print("\n📘 Answer:\n", answer)
