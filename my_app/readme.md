
# 📘 RAG-powered PDF/CSV Q&A Chatbot

This repository contains a Gradio-based application that allows users to upload a PDF or CSV document and ask questions about its content using Retrieval-Augmented Generation (RAG). The app retrieves the most relevant chunks from the document and passes them to a language model hosted on Hugging Face to generate a detailed answer and summary.

## 🚀 Features

- Supports PDF and CSV file uploads.
- Indexes documents using FAISS and HuggingFace sentence embeddings.
- Uses `Qwen/Qwen3-4B` for response generation via Hugging Face Inference Endpoint.
- Filters inappropriate output based on keyword blacklist.
- Logs interactions with context and answers.
- Simple Gradio UI for easy interaction.

## 🧰 Tech Stack

- Python
- LangChain
- FAISS
- HuggingFace Transformers & Inference API
- Gradio
- pdfplumber (for PDF parsing)

## 📂 Project Structure

- `app.py`: Main Gradio application with all logic.
- `vector_cache/`: Stores FAISS vector indexes per document.
- `rag_log.jsonl`: Logs of question-context-response interactions.
- `README.md`: This file.

## ⚙️ Installation & Setup

1. **Clone this repository**

```bash
git clone https://github.com/yourusername/rag-pdf-csv-chatbot.git
cd rag-pdf-csv-chatbot
```

2. **Install dependencies**

Make sure you are using Python 3.8+ and have pip installed.

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not available, install manually:
```bash
pip install langchain faiss-cpu gradio pdfplumber huggingface_hub
```

3. **Set your Hugging Face token**

You can add this to your environment or hardcode into the script.

```bash
export HUGGINGFACEHUB_API_TOKEN=your_token_here
```

## 🏃 Running the App

Simply run the `app.py` script:

```bash
python app.py
```

Gradio will start the web UI on a local URL (typically http://127.0.0.1:7860). Open it in your browser.

## 📝 How to Use

1. Upload a `.pdf` or `.csv` file using the UI.
2. Click “📚 Index Document” to create the vector index.
3. Enter your question and click “💬 Get Answer”.
4. View the detailed answer and context segments below.

## 🔒 Safety Filter

This app includes a basic filter that blocks responses containing keywords such as:
- `hack`, `bomb`, `kill`, `exploit`, `terrorist`

If any of these appear in the model output, a warning will be shown.

## 📜 Logging

Every Q&A interaction is logged to `rag_log.jsonl` with timestamp, question, context, and model response.

## 🧠 Caching

Indexed vector databases are cached in the `vector_cache/` folder using a hash of the file contents to speed up repeated use of the same document.

## 📬 Contact

For issues, please open a GitHub issue or contact the project maintainer.

---

MIT License © 2025
