import os
import pdfplumber
import chromadb
import requests  # New import for DeepSeek API calls
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# (Remove OpenAI API key and openai import since we're not using it now)

# Initialize ChromaDB (Persistent Storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("bayesian_docs")

# Initialize Sentence Transformer Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

### Step 1: Extract Text from PDF ###
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {e}")
    return text

### Step 2: Chunk Text for Retrieval ###
def chunk_text(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

### Step 3: Generate Text Embeddings ###
def get_embedding(text):
    return embedding_model.encode(text).tolist()

### Step 4: Process & Store PDF Data ###
def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    if not text:
        logging.warning(f"No text extracted from {pdf_path}")
        return
    chunks = chunk_text(text)
    
    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk)
            collection.add(
                ids=[f"{pdf_path}_{i}"], embeddings=[embedding], documents=[chunk],
                metadatas=[{"source": pdf_path}]
            )
        except Exception as e:
            logging.error(f"Error storing chunk {i} of {pdf_path}: {e}")
    logging.info(f"✅ Processed and stored: {pdf_path}")

### Step 5: Retrieve Relevant Text from ChromaDB ###
def retrieve_documents(query, top_k=5):
    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    # Assumes a single query embedding was passed; returns the list of documents.
    return results["documents"][0] if results["documents"] else []

### Step 6: Use DeepSeek to Generate Responses ###
def generate_response(query):
    relevant_docs = retrieve_documents(query)
    context = "\n\n".join(relevant_docs)
    
    prompt = f"""
    You are an expert in Bayesian statistics.
    Use the following retrieved knowledge to answer the user query:
    
    {context}

    User Query: {query}
    """
    
    # Construct the DeepSeek API request
    url = "https://api.deepseek.ai/generate"  # Replace with the actual DeepSeek endpoint
    payload = {
        "prompt": prompt,
        "max_tokens": 150,  # Adjust according to DeepSeek's API parameters
        # You might add more parameters like temperature, top_p, etc.
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_DEEPSEEK_API_KEY"  # Replace with your DeepSeek API key
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raises an exception for HTTP error codes
        result = response.json()
        # Adjust the parsing based on DeepSeek's API response structure.
        return result.get("response", "No response from DeepSeek.")
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "An error occurred while generating the response."

### Step 7: Expose API via FastAPI ###
app = FastAPI()

@app.post("/ask/")
def ask(query: str):
    response = generate_response(query)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn

    # Process all PDFs in the "pdfs" folder
    pdf_folder = "./pdfs"
    os.makedirs(pdf_folder, exist_ok=True)

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            process_pdf(os.path.join(pdf_folder, file))

    # Start the API
    logging.info("🚀 RAG API running at: http://127.0.0.1:8000/ask/")
    uvicorn.run(app, host="0.0.0.0", port=8000)
