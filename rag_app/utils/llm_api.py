import requests,os
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

def query_huggingface_model(prompt, model="meta-llama/Llama-3.1-8B-Instruct", token=HUGGINGFACEHUB_API_TOKEN):
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]["generated_text"]
