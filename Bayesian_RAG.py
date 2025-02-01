# Bayesian_RAG.py
import bootstrap  # Ensure the SQLite monkey-patch is applied first

import os
import pdfplumber
import chromadb
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import streamlit as st  # For accessing st.secrets

# Retrieve your DeepSeek API key from Streamlit secrets.
DEEPSEEK_API_KEY = st.secrets["deepseek_api_key"]
# Use the correct DeepSeek endpoint.
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Initialize ChromaDB (Persistent Storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("bayesian_docs")

# Initialize Sentence Transformer Model.
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
        print(f"Error processing {pdf_path}: {e}")
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
        print(f"No text extracted from {pdf_path}")
        return
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk)
            collection.add(
                ids=[f"{pdf_path}_{i}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": pdf_path}]
            )
        except Exception as e:
            print(f"Error storing chunk {i} of {pdf_path}: {e}")
    print(f"✅ Processed and stored: {pdf_path}")

### Step 5: Retrieve Relevant Text from ChromaDB ###
def retrieve_documents(query, top_k=5):
    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0] if results["documents"] else []

### Step 6: Use DeepSeek to Generate Responses (with debugging)
def generate_response(query):
    relevant_docs = retrieve_documents(query)
    context = "\n\n".join(relevant_docs)
    
    if not context:
        print("No relevant documents found for the query.")
    
    prompt = f"""
    You are an expert in Bayesian statistics.
    Use the following retrieved knowledge to answer the user query:
    
    {context}

    User Query: {query}
    """
    
    # Construct the payload for DeepSeek.
    payload = {
        "messages": [
            {"role": "system", "content": "You are a Bayesian statistics assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        if "response" in result:
            return result["response"]
        else:
            print("DeepSeek response JSON did not contain 'response':", result)
            return "DeepSeek API did not return a proper response."
    except Exception as e:
        print("Error generating response:", e)
        try:
            print("HTTP Response content:", response.text)
        except Exception as inner:
            print("Could not retrieve response content:", inner)
        return "An error occurred while generating the response."
