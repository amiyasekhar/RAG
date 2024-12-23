import openai
import docx
import fitz  # PyMuPDF for handling PDF files
from io import BytesIO
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import multiprocessing
import json
import requests
import re
import os
import easyocr

# For Streamlit interface
import streamlit as st
from dotenv import load_dotenv

# For using Huggingface Transformers model for Named Entity Recognition (NER)
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load environment variables (like OpenAI API key)
load_dotenv()

poppler_path = "./poppler/bin"

# OpenAI API setup
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load LegalBERT for named entity extraction
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("nlpaueb/legal-bert-base-uncased")
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Uploading files for the High Court Order, SC Petition, and List of Defects
uploaded_files = st.file_uploader("Upload your PDFs of High Court Order, SC Petition, and List of Defects", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Defect list (same as in your original code)
defect_list = {  # Truncated for brevity, include the full list here
    1.1: "Non-filing of SLP(Civil) in Form No.28 with certificate...",
    # Add the rest here...
}

# Function to extract text from PDF with multiprocessing
def extract_text_from_pdf_pages(page_num, pdf_bytes):
    """
    Extracts text from a single page of a PDF using OCR.
    Used for parallel processing with multiprocessing.
    """
    # Convert page to image
    pages = convert_from_bytes(pdf_bytes, dpi=300, first_page=page_num + 1, last_page=page_num + 1)
    text = pytesseract.image_to_string(pages[0])
    return text

def process_file(file, slp, file_type, sc_or_hc):
    text = extract_text(file)
    file_name = f"SLP_{slp}_{file_type}_Text.txt"
    result_file = f"statementA_results_slp_{slp}.txt"
    write_mode = get_file_mode(result_file)
    
    with open(result_file, write_mode) as results, open(file_name, 'w') as output:
        results.write(f"{sc_or_hc} {file_type} {slp}: {text}\n{'-'*15}\n")
        output.write(text)
    return text

# Multiprocessing-enabled function to process PDF files
def extract_text_from_pdf_multiprocessing(pdf_file):
    """
    Uses multiprocessing to extract text from all pages of a PDF file.
    """
    pdf_bytes = pdf_file.read()
    pdf_file.seek(0)  # Reset file pointer

    # Get the number of pages in the PDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    num_pages = doc.page_count

    # Create a pool of workers to process each page in parallel
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.starmap(extract_text_from_pdf_pages, [(i, pdf_bytes) for i in range(num_pages)])

    # Combine results into one string
    return "\n\n".join(results)

# Modify the process_slp_files and process_io_files to use multiprocessing for PDF text extraction
def process_slp_files(file, slp):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf_multiprocessing(file)
    return process_file(file, slp, "SC Petition", "SC")

def process_io_files(file, slp):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf_multiprocessing(file)
    return process_file(file, slp, "HC Order", "HC")

# Main function
def main():
    if st.button("Submit"):
        defect_list_text, hc_order_text, sc_petition_text = "", "", ""
        slp_numbers = ["3", "4", "5"]
        results_file = "results.txt"

        for uploaded_file in uploaded_files:
            for slp in slp_numbers:
                if uploaded_file.name.startswith(f"SLP {slp}"):
                    sc_petition_text = process_slp_files(uploaded_file, slp)
                elif uploaded_file.name.startswith(f"Impugned Order {slp}"):
                    hc_order_text = process_io_files(uploaded_file, slp)
                elif uploaded_file.name.startswith("Defects list"):
                    defect_list_text = extract_text(uploaded_file)
        
        hc_metatags = extract_metatags(hc_order_text, "hc")
        sc_metatags = extract_metatags(sc_petition_text, "sc")
        
        comparison_result = compare_metatags(hc_metatags, sc_metatags)
        defect_check_result = check_defects_using_list(sc_petition_text)

        final_output = f"Defects: {defect_check_result}\nComparison: {comparison_result}\n"

        with open(results_file, 'w') as results:
            results.write("********** DEFECT LIST **********\n")
            results.write(defect_list_text + "\n")
            results.write("********** HC ORDER **********\n")
            results.write(hc_order_text + "\n")
            results.write("********** SC PETITION **********\n")
            results.write(sc_petition_text + "\n")
            results.write("********** HC METATAG **********\n")
            results.write(json.dumps(hc_metatags, indent=2) + "\n")
            results.write("********** SC METATAG **********\n")
            results.write(json.dumps(sc_metatags, indent=2) + "\n")
            results.write("********** DEFECT CHECK RESULT **********\n")
            results.write(defect_check_result + "\n")
            results.write("********** COMPARISON RESULT **********\n")
            results.write(comparison_result + "\n")
        st.write(final_output)
        print("We're done")

if __name__ == '__main__':
    main()