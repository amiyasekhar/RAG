import faiss
from transformers import AutoModel, AutoTokenizer
import torch
import streamlit as st
import numpy as np
import os
import psutil

device = torch.device("cpu") # By forcing the computations to run on the CPU, we avoid potential crashes that could arise from GPU memory being insufficient to handle large batches of document embeddings.
torch.set_num_threads(1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
faiss.omp_set_num_threads(1)

# Load LegalBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Initialize FAISS index
dimension = 768  # The embedding dimension for LegalBERT is 768
index = faiss.IndexFlatL2(dimension)
document_texts = []

# Memory monitoring function
def log_memory_usage(stage):
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"{stage} - Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB")
    return mem_info.rss / 1024 ** 2  # Return memory in MB

# Calculate maximum batch size based on memory
def calculate_max_batch_size():
    total_memory = psutil.virtual_memory().total / 1024 ** 2  # Total system memory in MB
    available_memory = psutil.virtual_memory().available / 1024 ** 2  # Available memory in MB
    print(f"Available memory: {available_memory:.2f} MB")

    # Each embedding is `768 dimensions * 4 bytes = 3072 bytes (or 3 KB) per embedding`
    # We estimate a buffer (e.g., 30%) of available memory to avoid segmentation fault.
    memory_buffer = 0.8 * available_memory
    usable_memory = available_memory - memory_buffer

    # Estimate number of embeddings that can fit into usable memory (3 KB per embedding)
    max_embeddings = usable_memory * 1024 / 3  # In KB
    print(f"Estimated max number of embeddings before crash: {int(max_embeddings)}")

    # Return batch size based on estimated embeddings we can safely process
    # Divide the estimated number of embeddings by the number of dimensions (768) to get the batch size
    return max(int(max_embeddings // dimension), 1)  # Make sure the batch size is at least 1

# Encoding documents into vectors for use in FAISS
def encode_documents(texts):
    log_memory_usage("Before encoding documents")
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    log_memory_usage("After encoding documents")
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()  # Force to use CPU
    
    return embeddings

def add_documents_to_index(documents):
    global document_texts
    document_texts = documents
    
    # Calculate maximum batch size based on available memory
    max_batch_size = calculate_max_batch_size()
    # max_batch_size = max(calculate_max_batch_size() // 9000, 1)
    print(f"Max batch size: {max_batch_size}")
    # Process and add documents in batches
    for i in range(0, len(documents), max_batch_size):
        batch = documents[i:i + max_batch_size]
        print(f"Processing batch {i // max_batch_size + 1} with size {len(batch)}")
        embeddings = encode_documents(batch)
        log_memory_usage("Before adding to FAISS index")
        index.add(embeddings)
        log_memory_usage("After adding to FAISS index")

def vector_retrieve_documents(query):
    # Retrieve documents based on query embeddings
    query_embedding = encode_documents([query])
    D, I = index.search(query_embedding, k=3)
    return [document_texts[i] for i in I[0]]


# Example usage: You can integrate this with your HybridRAG or any other logic
# to add documents and retrieve results using FAISS efficiently with batching.