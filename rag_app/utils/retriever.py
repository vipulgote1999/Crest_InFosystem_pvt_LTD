# from sentence_transformers import SentenceTransformer, util

# model = SentenceTransformer("all-MiniLM-L6-v2")

# def retrieve_chunks(question, chunks, top_k=2):
#     chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
#     question_embedding = model.encode(question, convert_to_tensor=True)
#     scores = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
#     top_results = scores.topk(k=top_k)
#     return [chunks[i] for i in top_results.indices]

import os
from huggingface_hub import InferenceClient


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
client = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",provider="hf-inference",
    api_key=HUGGINGFACEHUB_API_TOKEN,
)

def retrieve_chunks_remote(question, chunks, top_k=2):
    # Get similarity scores
    result = client.sentence_similarity({
        "source_sentence": question,
        "sentences": chunks
    })

    # Get top-k indices
    top_k_indices = sorted(range(len(result)), key=lambda i: result[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_k_indices]