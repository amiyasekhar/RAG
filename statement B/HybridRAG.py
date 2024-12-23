import openai
import streamlit as st
import os
import faiss
from dotenv import load_dotenv
from GraphRAG import build_knowledge_graph, graph_retrieve_entities
from VectorRAG_with_faiss import encode_documents, log_memory_usage
import docx
import fitz  # PyMuPDF

# Load environment variables from the .env file
load_dotenv()

# Fetch the API key from the environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

# Streamlit interface setup
st.set_page_config(layout="wide")
st.title("SC: SLP Chat Bot")
st.subheader("(Demo for the SC Hackathon 2024 by LexAI)")

# Initialize FAISS index and document texts in session state
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(768)  # Initialize FAISS index for 768-dimensional embeddings
    st.session_state.document_texts = []

# Debugging session state
print("Current FAISS index:", st.session_state.faiss_index)
print("Current document_texts length:", len(st.session_state.document_texts))

# Query input using Streamlit's text input
query = st.text_input("Enter your query:")

# File uploader for accepting multiple files
uploaded_files = st.file_uploader("Upload your documents (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

# Function to combine results from both RAG approaches (VectorRAG and GraphRAG)
def hybrid_retrieve(query):
    # Retrieve unstructured data using VectorRAG
    vector_results = vector_retrieve_documents(query)
    
    # Retrieve structured data using GraphRAG
    graph_results = graph_retrieve_entities(query)
    
    return vector_results, graph_results

# OpenAI GPT generation function
def generate_response_from_hybrid_rag(query, retrieved_texts, retrieved_entities, document_names):
    custom_prompt = f"""
    You are a helpful assistant who queries the pleadings in a case. The pleadings are uploaded to your database. Only based on relevant information in a database are you to provide an answer. 
    
    Identify the Impugned Order:
    Scan the Index section for terms like "Impugned Order," "High Court Order," or equivalent keywords.
    Extract the annexure label (e.g., P-1, P-30) and the corresponding page numbers.
    Verify if the annexures contain a certified copy by checking for any certification stamps or noted mentions of certification.

    Summarize the Grounds:
    Look for a section explicitly titled "Grounds for Appeal," "Grounds,” or similar.
    Extract key arguments, legal principles, or points raised by the petitioner challenging the lower court's judgment.
    Summarize these grounds concisely while preserving key legal points and rationales.

    Summarize the Facts:
    Locate the "Statement of Facts," "Background Information," or similar section.
    Extract relevant chronological events, key figures involved, and context of the case.
    Provide a coherent, concise summary covering the essence of the case's background.

    Summarize the Prayers:
    Identify the section with "Prayers," "Relief Sought," or similar terminology.
    Extract specific reliefs or orders sought by the petitioner from the court.
    Summarize all reliefs concisely and clearly.

    Check for Office Objections:
    Match the petition content against a standardized list of common objections which are available in your database, such as:
    - Missing signatures
    - Non-filing of affidavits
    - Issues with Vakalatnama
    - Incorrect or missing translations
    - Incomplete or incorrect annexure labeling
    - Proof of proper certification for documents

    Highlight any discrepancies or missing elements based on this checklist.
    
    Example Step-by-Step Instructions:
    - Locate the Impugned Order: Search Index using keywords like "Impugned Order" or "High Court Order."
    - Extract and verify annexure details: Page numbers, certification status.
    - Summarize Grounds: Search Document for Keywords - "Grounds for Appeal," "Grounds."
    - Extract and Summarize: Focus on legal arguments and challenges.
    - Summarize Facts: Search Document for Keywords - "Statement of Facts," "Background."
    - Extract and Summarize: Chronology, key events, and context.
    - Summarize Prayers: Search Document for Keywords - "Prayers," "Relief Sought."
    - Extract and Summarize: Specific reliefs requested.
    - Check for Office Objections: use Standardized Checklist and Verify against common procedural objections.
    - Highlight Findings: Indicate any missing or incorrect procedural components.

    Summary Example for Bot Instructions:
    
    - Index (High Court Order): Locate "Final Order of the High Court of” and reference page number from the Index
    - Index (Impugned Order): Locate "Impugned Order." E.g., Annexure P-1, pages 40-52.
    - Annexures (Grounds): Section “Grounds for Appeal.” Summarize legal points.
    - Annexures (Facts): Section “Statement of Facts.” Provide key events summary.
    - Annexures (Prayers): Section “Prayers.” Summarize reliefs sought.
    - Office Objections: List all checked items, e.g., missing signatures, incorrect translations.
    
    The following unstructured texts have been retrieved:
    {retrieved_texts}
    
    The following structured knowledge has been retrieved:
    {retrieved_entities}

    Based on this, provide a detailed answer to the query: {query}
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a legal assistant."},
            {"role": "user", "content": custom_prompt}
        ]
    )
    # Get the generated response content
    generated_answer = response.choices[0].message.content

    # Prepend the document list and query to the generated answer
    documents_info = f"Documents fed:\n" + "\n".join(document_names) + f"\n\nUser query: {query}\n\n"

    # Combine the documents info and the actual generated response
    final_answer = documents_info + generated_answer

    return final_answer

# Function to extract text from a .docx file
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to extract text from a PDF file using in-memory buffer
def extract_text_from_pdf(pdf_file):
    # Read PDF using in-memory file object
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")  # Use stream instead of file path
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Adding documents to FAISS index with session persistence
def add_documents_to_index(documents):
    for doc in documents:
        # Encode the document
        embeddings = encode_documents([doc])
        
        # Add to the FAISS index stored in session state
        st.session_state.faiss_index.add(embeddings)
        
        # Add document to session_state document_texts for retrieval later
        st.session_state.document_texts.append(doc)
    
    # Debugging session state after adding documents
    print("FAISS index updated:", st.session_state.faiss_index)
    print("Updated document_texts length:", len(st.session_state.document_texts))

# Retrieving documents using FAISS
def vector_retrieve_documents(query):
    query_embedding = encode_documents([query])
    D, I = st.session_state.faiss_index.search(query_embedding, k=3)

    # Debugging the retrieval process
    print("Indexes returned by FAISS:", I)
    
    return [st.session_state.document_texts[i] for i in I[0] if i < len(st.session_state.document_texts)]

# Main function to control the workflow
def main():
    if st.button("Submit Query"):
        if uploaded_files and query:
            documents = []
            document_names = []

            # Process each uploaded file
            for uploaded_file in uploaded_files:
                file_extension = uploaded_file.name.split('.')[-1]
                document_names.append(uploaded_file.name)

                if file_extension == "docx":
                    documents.append(extract_text_from_docx(uploaded_file))
                elif file_extension == "pdf":
                    documents.append(extract_text_from_pdf(uploaded_file))

            st.write("Processing documents and indexing...")

            # Add documents to FAISS index
            add_documents_to_index(documents)

            # Hybrid retrieval of data
            st.write("Retrieving relevant information...")
            retrieved_texts, retrieved_entities = hybrid_retrieve(query)

            # Generate response from hybrid RAG
            st.write("Generating response based on retrieved documents...")
            answer = generate_response_from_hybrid_rag(query, retrieved_texts, retrieved_entities, document_names)

            # Display the answer
            st.text_area("Answer:", value=answer, height=500)
        else:
            st.write("Please upload documents and enter a query.")

if __name__ == '__main__':
    main()