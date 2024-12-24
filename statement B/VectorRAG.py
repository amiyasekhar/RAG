# import faiss
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

device = torch.device("cpu")

# Load LegalBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Initialize FAISS index
# dimension = 768  # The embedding dimension for LegalBERT is 768
# index = faiss.IndexFlatL2(dimension)
document_texts = []


# When not using faiss, uncomment this function
def encode_documents(texts):
    # Process in smaller batches if your system memory is limited
    batch_size = 1  # Lower the batch size
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.concatenate(embeddings, axis=0)

'''
def encode_documents(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()  # Force to use CPU
    return embeddings
'''


def add_documents_to_index(documents):
    global document_texts
    document_texts = documents
    embeddings = encode_documents(documents)
    # index.add(embeddings)

def vector_retrieve_documents(query):
    # Retrieve documents based on query embeddings
    query_embedding = encode_documents([query])
    # D, I = index.search(query_embedding, k=3)
    # return [document_texts[i] for i in I[0]]
    
    return document_texts


# https://streamlit.io/gallery
# anmol@staram.in

'''
https://indiankanoon.org/doc/48103131/
https://indiankanoon.org/browselaws/ - legislation
https://indiankanoon.org/browse - judgements

+91-9113356699

Anmol Deep

'''