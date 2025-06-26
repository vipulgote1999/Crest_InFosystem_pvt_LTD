import os
import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


# 🔑 Set your Hugging Face API Token
HUGGINGFACEHUB_API_TOKEN = "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"
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


# 1. Define the LLM endpoint for chat models
llm_endpoint = HuggingFaceEndpoint(
    # repo_id="Qwen/Qwen3-8B",
    # repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    repo_id="Qwen/Qwen3-4B",
    # repo_id="google/flan-t5-base",
    # task="conversational",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    temperature=0.3,
    max_new_tokens= 4096,
    stop_sequences=[ "<|im_start|>", "<|im_end|>" ],
    provider="auto"
)

# 2. Wrap it in chat-aware class
chat_llm = ChatHuggingFace(llm=llm_endpoint)

# 3. Use ChatHuggingFace in the chain
rag_chain = RunnableMap({
    "context": lambda inp: inp["context"],
    "question": lambda inp: inp["question"]
}) | prompt | chat_llm

# 7. Full RAG QA Function
def rag_qa(pdf_path, question, k=3):
    docs = load_pdf(pdf_path)
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
    pdf_path = r"C:\Users\vg929494.ttl\Downloads\Crest_InFosystem_pvt_LTD-main\my_app\Offer Letter 2025-05-22.pdf"
    # question = "What is the data described in the document?"
    question = "What is the total offer ctc amount?"
    answer = rag_qa(pdf_path, question)
    print("\n📘 Answer:\n", answer)
