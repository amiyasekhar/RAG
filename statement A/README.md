# Legal Document Metatag Extraction and Comparison Tool

This is a Python-based tool that allows users to upload legal documents (High Court Order, SC Petition, and Defect List), extract specific named entities using LegalBERT, and compare the extracted metadata. Additionally, it checks the SC Petition against a provided Defect List to find any discrepancies.

## Features
- **File Upload:** Users can upload multiple files in `PDF`, `DOCX`, and `TXT` formats.
- **OCR for Scanned PDFs:** The tool can extract text from scanned PDFs using Tesseract OCR.
- **Legal Named Entity Extraction:** Uses `LegalBERT` to extract specific legal named entities from High Court Orders and SC Petitions.
- **Document Comparison:** Compares extracted metatags from High Court Orders and SC Petitions to identify inconsistencies.
- **Defect Check:** Checks the SC Petition for defects based on the provided defect list.
- **Final Report Generation:** Combines results from metatag comparison and defect checks into a final output report.

## Installation

### Prerequisites

Ensure the following software is installed:
- Python 3.8+
- `pip` for package management
- `poppler` (for PDF to image conversion)
- `Tesseract OCR` (for OCR on scanned PDFs)

### Install dependencies
pip install -r requirements.txt
brew install tesseract
sudo apt-get install tesseract-ocr
brew install poppler
sudo apt-get install poppler-utils

### Use OpenAI Key
OPENAI_API_KEY=your_openai_api_key

### Run the program
streamlit run test.py

