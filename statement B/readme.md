# Hybrid RAG with FAISS and Streamlit

This repository contains a Hybrid Retrieval-Augmented Generation (RAG) system that combines vector-based retrieval using FAISS and graph-based retrieval using a knowledge graph. The system uses OpenAI GPT models to generate responses based on user queries and uploaded documents. The frontend is built using Streamlit to allow users to upload documents, submit queries, and receive detailed responses.

## Files

### 1. `HybridRAG.py`
This is the main driver code for the application. It integrates all components, including document upload, retrieval using FAISS, and response generation using OpenAI's GPT. It handles:
- File uploads (PDF or DOCX).
- User queries through the Streamlit interface.
- Document processing and retrieval using both vector and graph-based approaches.
- Displaying the generated response to the user.

### 2. `VectorRAG.py`
This file contains the logic for vector-based document retrieval without using FAISS. Key functions include:
- **Document Encoding**: Converts documents into embeddings using transformer models (e.g., LegalBERT).
- **Retrieval**: Uses cosine similarity to find the most relevant documents based on the user's query.

### 3. `VectorRAG_with_faiss.py`
This file extends `VectorRAG.py` by incorporating FAISS for efficient vector-based retrieval. Key features include:
- **FAISS Indexing**: Adds document embeddings to a FAISS index for fast nearest-neighbor search.
- **Batch Processing**: Supports batching for processing large documents to avoid memory issues.
- **FAISS-based Retrieval**: Retrieves relevant documents by searching the FAISS index based on the query embeddings.

### 4. `GraphRAG.py`
This file implements the graph-based retrieval logic. It allows:
- **Knowledge Graph Creation**: Builds a graph from document content, creating relationships between entities (e.g., people, organizations, court orders).
- **Entity and Relationship Retrieval**: Complements vector-based retrieval with structured information from the knowledge graph.

### 5. `requirements.txt`
This file lists all the dependencies required to run the project, including:
- **faiss-cpu**: For fast similarity search.
- **transformers** and **torch**: For document embedding using transformer models.
- **pymupdf**: For extracting text from PDF files.
- **Streamlit**: For building the frontend interface.
- **OpenAI**: For integrating GPT-based response generation.

## Installation

1. Clone the repository:

   ```bash
   git clone *TBD*
   cd hybrid-rag-faiss

2. Install the dependencies:
    pip3 install -r requirements.txt

To run the Streamlit application, use the following command:
    streamlit run HybridRAG.py

The app will launch in your browser. You can:
    Upload documents (PDF or DOCX).
    Enter a query to retrieve relevant information from the uploaded documents.
    View the generated response in the browser.

___________________________________________________________________________________________________________

## Explanation of Key Components

### Hybrid Retrieval
The system uses a combination of:

- **Vector-based retrieval (RAG)**: Documents are encoded into embeddings, which are compared to the query's embedding to find the closest matches. FAISS is used to perform fast, approximate nearest neighbor search.
- **Graph-based retrieval (GraphRAG)**: A knowledge graph is constructed from entities and relationships found in the documents. This structured data enhances the unstructured text retrieval from the vector-based system.

### FAISS Integration
FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. In this project, FAISS helps by:

- **Indexing embeddings** of the documents for fast retrieval.
- **Querying the index** to find the top documents that are most relevant to the user's query.

### OpenAI GPT Integration
The system uses OpenAI's GPT models to generate a natural language response based on the retrieved documents and the query. The response is structured and informative, providing the user with detailed insights based on the retrieved text and structured knowledge.

### Streamlit Frontend
Streamlit provides a simple interface where users can:

- **Upload files** (PDF or DOCX).
- **Input their queries**.
- **Receive responses** generated by OpenAI GPT, based on the combined vector and graph retrieval methods.

## Example

1. Upload files such as **Special Leave Petition Civil**, **Form 28.pdf**, and more.
2. Enter a query like:

    ```text
    Does the Special Leave Petition Civil text adhere to the structure provided in Form 28?
    ```

3. The system will process the documents, retrieve relevant information, and return an answer based on the documents' content and their compliance with the Form 28 structure.

## Troubleshooting

- **Memory Issues**: If you encounter memory issues, consider adjusting the batch size in `VectorRAG_with_faiss.py` to process fewer documents at a time.
- **Document Processing Errors**: Ensure that the uploaded documents are either PDF or DOCX formats. Double-check that the documents are properly formatted so that the system can extract text correctly.

Receive responses generated by OpenAI GPT, based on the combined vector and graph retrieval methods.