import json

def log_interaction(question, retrieved, answer, summary, filepath="logs/rag_log.jsonl"):
    record = {
        "question": question,
        "retrieved": retrieved,
        "answer": answer,
        "summary": summary
    }
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")
