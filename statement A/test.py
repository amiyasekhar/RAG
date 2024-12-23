import openai
import docx
import fitz  # PyMuPDF for handling PDF files
from io import BytesIO
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from threading import Thread
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

defect_list = {
    1.1: "Non-filing of SLP(Civil) in Form No.28 with certificate as per Notification dated 17.6.1997",
    1.2: "Deficit Court Fees – More court fee required",
    1.3: "Non-filing of certificate by AoR in support of SLP",
    1.4: "No clarification regarding filing of SLP against Final or Interim Order",
    1.5: "Non-mentioning of the findings of the High Court/Courts below in the list of dates and para 1 of SLP",
    1.6: "Incomplete prayer/details of property, land etc. not mentioned/correctly mentioned in prayer for interim relief",
    1.7: "Non-mentioning of date of filing/drawn by",
    1.8: "Why the main matter has not been challenged",
    2.1: "Non-filing of brief list of dates/events",
    2.2: "Non-filing of application for permission to file lengthy list",
    2.3: "Incorrect numbering of paragraphs and pagination of paper books and description in Index",
    2.4: "Non-mentioning of volume number with page numbers in Index of Paper-books",
    2.5: "Non-filing of office report on limitation with cause title on Green/White sheet",
    2.6: "Non-mentioning of page numbers of annexures in the list of dates/petition",
    2.7: "Incorrect mentioning of description of annexures in list of dates/petition",
    2.8: "Incorrect mentioning of description of annexures in Index",
    2.9: "Wrong mentioning of Court name at page No.",
    2.10: "Non-mentioning of Assessment Year on cover page of paper books and page B",
    3.1: "Not in double space on one side of the paper",
    3.2: "Page not clear/legible/small font/dim/missing",
    3.3: "Pages contain underlines/highlights/blanks/torn condition",
    3.4: "Pages not included horizontally",
    3.5: "Page numbers not indicated on the right side of top of pages",
    4.1: "Non-bearing of signature of the counsel/party-in-person in petition/application",
    5.1: "Non-filing of affidavit containing statements based on information, whether the deponent has disclosed the source",
    5.2: "Non-filing of affidavit containing the statement that the facts stated in the petition are true to the knowledge",
    5.3: "Non-filing of affidavit properly executed",
    5.4: "Non-disclosure of deponent’s capacity (in case matter is filed on behalf of or by organisation/company/pairokar)",
    5.5: "Blanks in the affidavit",
    6.1: "Non-filing of translation of vernacular document(s)",
    6.2: "Non-mentioning of annexure numbers of the translated document(s)",
    6.3: "Non-filing of application for exemption from filing official translation, with affidavit and court fee",
    7.1: "Non-filing of application for setting aside abatement",
    7.2: "Non-filing of application for substitution containing details regarding date of death/age/relationship and address of LRs",
    7.3: "Non-filing of application for substitution containing details regarding date of death/age/relationship and address of LRs",
    7.4: "Non-filing of photocopy of Death Certificate in substitution application",
    7.5: "Non-filing of application for condonation of delay in filing substitution",
    7.6: "Non-filing of application for permission to file SLP along with application for substitution",
    7.7: "Non-filing of translation of Vernacular Death Certificate",
    8.1: "Improper execution of Vakalatnama/Memo of Appearance",
    8.2: "Non-affixation of Welfare Stamp",
    8.3: "Non-mentioning of capacity of executant for signing of vakalatnama",
    8.4: "Non-filing of Power of Attorney in English/translated copy",
    8.5: "Non-inclusion of Vakalatnama attested from jail",
    9.1: "Application seeking permission to appear and argue in person not filed by petitioner-in-person",
    10.1: "The petitioner has not filed an affidavit stating that there is no personal gain or oblique reason in filing the Public Interest Litigation",
    10.2: "Non-furnishing of details regarding registration and authorization in para 1A of the writ petition",
    10.3: "Writ not filed in terms of ORDER XXXVIII SCR, 2013 if it is filed as PIL",
    10.4: "Non-Mentioning in para 1A of writ petition regarding petitioner having approached the concerned authority/respondents",
    10.5: "Why private parties arrayed as respondents in Writ Petition",
    11.1: "Non-filing of authorization letter issued by body incorporate to file petition with proof",
    11.2: "Non-filing of copy of registration certificate in case petition is filed by a body registered under any Act or Rules",
    12.1: "Non-mentioning in petitions/appeal of statement in terms of Order XXI Rule 3/Order XXII Rule 2 of Supreme Court Rules",
    12.2: "Whether the petitioner has moved any petition for the same relief",
    12.3: "Non-furnishing of statement as to whether LPA or Writ Appeal lies against the impugned judgment",
    13.1: "Non-filing of certified copy of the impugned judgment",
    13.2: "Name of High Court, Cause Title etc. not shown in plain copy of impugned order",
    13.3: "As certified copy is not available, application for exemption from filing certified copy has not been filed",
    13.4: "Application seeking permission to file SLP without certified copy as well as plain copy of impugned order not filed",
    13.5: "Contents of certified copy and typed copy don’t tally",
    14.1: "Particulars of the impugned judgments not uniformly written in all the documents",
    15.1: "Non-filing of Memo of Parties, as detailed cause title not given in the impugned judgment",
    15.2: "Non-furnishing of name and addresses of the counsel, who appeared before the court below",
    15.3: "Incomplete/Incorrect addresses/status of the parties and their representation",
    15.4: "Separate cause title not shown (If more than one matter)",
    15.5: "Cause title of the petition/appeal not corresponding to that of the impugned judgment and names of parties therein",
    15.6: "Contesting/proforma respondents not mentioned separately",
    15.7: "Details regarding name and address of each petitioner along with status before High Court and Lower Court not given",
    15.8: "Why President of India/Governor/Tribunal/Judicial Officer has been made party",
    16.1: "Appeal not accompanied by judgment and decree appealed from with order granting certificate",
    17.1: "Non-mentioning of the number of days of delay",
    17.2: "Non-filing of application for condonation of delay in time barred petition/appeal",
    18.1: "Separate annexures to be filed and not collectively",
    18.2: "Annexures referred to in the petition not true copies of the documents before the court below",
    18.3: "Date(s) of annexures do not tally",
    18.4: "Date(s) of annexures not given",
    18.5: "Case No.(s) of annexures not given",
    18.6: "Non-filing of copy of judgment/order/notification/award/annexure",
    18.7: "Non-filing of copy of the petition in transfer petition",
    19.1: "Application for taking additional grounds/filing of documents with affidavit and court fee not filed",
    19.2: "Non-mentioning/improper mentioning of annexures with page numbers in application for taking additional grounds/documents",
    19.3: "Non-filing of annexures along with the application for taking additional grounds/documents on record",
    20.1: "Non-filing of copies of the orders of the Trial Court/First Appellate Court in case of SLP filed against RSA",
    20.2: "Non-filing of copies of the orders of courts below",
    21.1: "Non-filing of application for exemption from surrendering",
    21.2: "Non-furnishing of the statement in the petition whether the petitioner has surrendered",
    21.3: "Non-inclusion of the copy of surrender/custody certificate in the paper book",
    21.4: "Non-inclusion of Custody Certificate attested from jail",
    21.5: "Non-filing of copy of surrender proof/custody certificate in respect of all convicts",
    21.6: "Non-filing of application for exemption from filing separate proof of surrender",
    21.7: "Non-mentioning of period of sentence already undergone in application for bail",
    22.1: "Non-filing of copy of petition filed before High Court under Section 482 of Cr.P.C.",
    23.1: "Copy of FIR/translated copy of FIR not filed",
    24.1: "Non-mentioning of the period of custody undergone by the accused",
    24.2: "Non-inclusion of complete listing proforma filled in, signed in the paper-books",
    25.1: "Non-furnishing of complete particulars of any identical matter pending/disposed of by Supreme Court",
    26.1: "Statement in terms of Order XIX Clause 3(1) of Rules, not given in the Petition of Appeal",
    27.1: "Non-filing of proof of depositing Rs.50,000/-",
    28.1: "Non-filing of order refusing/granting leave to file Appeal",
    29.1: "Receipt of Rs.15,000/- received from the Cash Branch not filed",
    29.2: "If commercial litigation matter, CD format/e-mail addresses of all the parties not filed",
    29.3: "Sub-section of Section 11 under which Arbitration Petition filed, not mentioned",
    29.4: "Clause of the agreement challenged in the petition, not mentioned",
    29.5: "Pagination with description of annexures not mentioned in the petition",
    29.6: "Para i.e. all conditions set out in Section 11 of Arbitration Act have been satisfied, not given",
    29.7: "Original Arbitration Agreement or a duly certified copy or application for exemption from filing certified copy of agreement, not filed",
    29.8: "Names and addresses of the parties to the Arbitration Agreement, not given",
    29.9: "Names and addresses of the Arbitrators, if any, already appointed, not given",
    29.10: "Name and address of the person or institution, if any, to whom or to which any function has been entrusted by the parties",
    29.11: "The qualifications required, if any, of the arbitrators by the agreement of the parties, not given",
    29.12: "Brief written statement describing the general nature of dispute and the points at issue, not given",
    29.13: "Relief or remedy sought, not given",
    29.14: "Affidavit, supported by relevant documents, to the effect that the condition to be satisfied under sub-section (4), (5), or (6) of Section 11 before making the request to Hon. the Chief Justice, has been satisfied",
    30.1: "Non-clarification as to the nature of matter whether Civil or Criminal",
    30.2: "Non-clarification as to why SLP/Appeal filed without approaching High Court/Tribunal",
    30.3: "No clarification of Advocate if FIR sought to be transferred in a Transfer Petition",
    30.4: "No clarification as to maintainability of writ petition (where SLP/CA already filed/disposed of)"
}

# Get write mode
def get_file_mode(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist. Using 'w' mode.")
        return 'w'
    
    # If the file exists, check if it is blank (including only new line characters)
    with open(file_path, 'r') as f:
        content = f.read()
        if not content.strip():  # Check if content is empty or only contains whitespace (like newlines)
            print(f"File '{file_path}' is empty or contains only new lines. Using 'w' mode.")
            return 'w'
    
    print(f"File '{file_path}' contains valid content. Using 'a' mode.")
    return 'a'

# Checking if file has loaded properly
def check_file(file):
    if file is None or file.read() == b"":
        print(f"Uploaded file named '{file.name}' is empty or not properly uploaded.")
        return False

    file.seek(0)  # Reset the file pointer to the beginning
    return True

# Extract text from pdf
def extract_text_from_pdf(pdf_file):
    # Initialize EasyOCR reader with GPU support
    reader = easyocr.Reader(['en'], gpu=False)

    # Open the PDF file
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")

    extracted_text = ""

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()

        # Convert the page to an image
        img = Image.open(io.BytesIO(pix.tobytes()))

        # Convert the image to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Perform OCR on the image
        result = reader.readtext(img_byte_arr)

        # Extract and concatenate the text
        page_text = ""
        for detection in result:
            page_text += detection[1] + " "

        # Add the page text to the overall extracted text with a newline
        extracted_text += page_text.strip() + "\n\n"

    return extracted_text.strip()

# Function to extract text from images in scanned PDFs
def extract_images(pdf_file: BytesIO) -> str:
    """
    Given a PDF file as a BytesIO object, extract and perform OCR on each page image and return the extracted text.
    """
    try:
        # Reset the BytesIO file pointer to the beginning
        pdf_file.seek(0)
        
        # Convert BytesIO object to bytes and pass to convert_from_bytes
        pdf_bytes = pdf_file.read()
        pages = convert_from_bytes(pdf_bytes, dpi=300)
    except Exception as e:
        raise ValueError("Error extracting images from pdf") from e
    
    list_text = []
    
    # Iterate through pages and perform OCR
    for i, page in enumerate(pages):
        print(f"Starting to convert Page {i+1} to text")
        img_byte_arr = BytesIO()
        page.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        
        # Perform OCR directly on the image
        text = pytesseract.image_to_string(Image.open(BytesIO(img_byte_arr)))
        list_text.append(text)
    
    return "\n\n".join(list_text)

# Extracting text from DOCX, PDF, and TXT files
def extract_text(file):
    """
    Extracts text from DOCX, PDF, and TXT files.
    If the PDF is a scanned document, it uses OCR to extract the text.
    """
    file_extension = file.name.split('.')[-1].lower()
    text = ""

    # If it's a DOCX file
    if file_extension == "docx":
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])

    # If it's a PDF file
    elif file_extension == "pdf":
        try:
            file.seek(0)  # Reset file pointer
            text = extract_text_from_pdf(file)

            if not text.strip():  # If text is empty or whitespace, fallback to OCR
                print("PDF text extraction returned empty, trying OCR...")
                file.seek(0)  # Reset the file pointer
                text = extract_images(BytesIO(file.read()))  # Use BytesIO for OCR
        except Exception as e:
            print("PDF text extraction failed, trying OCR...")
            file.seek(0)  # Reset the file pointer
            text = extract_images(BytesIO(file.read()))  # Use BytesIO for OCR

    # If it's a TXT file
    elif file_extension == "txt":
        text = file.read().decode('utf-8')

    else:
        raise ValueError("Unsupported file format. Only DOCX, PDF, and TXT are supported.")

    return text

# Extract metatag from SLP / IO
def extract_metatags(document_text, prompt_type="sc"):
    prompts = {
        "sc": f"""
Following is a Special Leave Petition to be filed to the Supreme Court of India: {document_text}
        From the above text document, extract values for the following Legal Named Entities : ["BenchCoram", "CaseNo", "CasesReferredTo", "Court", "JudgmentDate", "LawyerForPetitioner", "LawyerForRespondent", "Petitioner", "Respondent"]
        where the explaination or interpretation of the named entities are given as follows:
                "BenchCoram": Name(s) of the judges present in the previous High Court Order or Judgment,
                "CaseNo": Case number(s) of the previous High Court Order or Judgment; This should not include the SLP (Special Leave Petition) No., rather only the Case Number corresponding to the High Court Judgment or Order,
                "CasesReferredTo": Names(s) and citation of precedent cases reffered in the previous High Court Order or Judgment,
                "Court": Name of the High Court which has handled delivered judgment before filing this petition to the Supreme Court of India,
                "JudgmentDate": The date on which the judgment of the High Court was announced,
                "LawyerForPetitioner": Name(s) of the lawyer representing the petitioners in the High Court,
                "LawyerForRespondent": Name(s) of the lawyer representing the respondents in the High Court,
                "LegislationsReferred": Name(s) of the acts/ sections/ legislations/ Indian Penal Codes reffered in the High Court Order or Judgment,
                "Petitioner": Name(s) of the petitioners or appellants of the current Special Leave Petition,
                "Respondent": Name(s) of the respondents or oppositions of the current Special Leave Petition,
                
        The pairs of named entities and their corresponding values should be given in python dictionary format (where both keys and values are string only). These named entities are multivalued i.e., more than one values can be present for them: ["BenchCoram", "CaseNo", "CasesReferredTo", "Court", "LawyerForPetitioner", "LawyerForRespondent", "Petitioner", "Respondent"]. For multivalued entities the values should be seperated by a semicolon delimeter (';'). For entities having no values write 'NA' as their value.
        An example of desired output looks like following:

            {{
                "BenchCoram": "Bhaskar Raj Pradhan",
                "CaseNo": "W.P. (C) No.07 of 2023",
                "CasesReferredTo": "Hari Prasad Sharma vs. Union of India & Ors; Hari Prasad Sharma vs. Union of India & Ors. RCC",
                "Court": "KARNATAKA HIGH COURT",
                "JudgmentDate": "08/09/2024",
                "LawyerForPetitioner": "Babita Kumari; Thupden G. Bhutia",
                "LawyerForRespondent": "Mohan Sharma; Natasha Pradhan; Purnima Subba; Sangita Pradhan; Shakil Raj Karki; Zangpo Sherpa",
                "LegislationsReferred": "N.H. Act, 1956",
                "Petitioner": "Hari Prasad Sharma",
                "Respondent": "Union of India; Union of India & Ors.",
            }}        
        """,
        "hc": f"""
Following is a High court Judgment order: {document_text}
        From the above text document extract values for the following Legal Named Entities : ["BenchCoram", "CaseNo", "CasesReferredTo", "Court", "JudgmentDate", "LawyerForPetitioner", "LawyerForRespondent", "Petitioner", "Respondent"]
        where the explaination or interpretation of the named entities are given as follows:
                "BenchCoram": Name(s) of the judges present in the judgment bench of the current case,
                "CaseNo": Case number(s) of the current case for which judgment is performed,
                "CasesReferredTo": Names(s) and citation of precedent cases reffered in current case,
                "Court": Name of the court delivering the current judgment,
                "JudgmentDate": The date on which the judgment of the current case is announced,
                "LawyerForPetitioner": Name(s) of the lawyer representing the petitioners,
                "LawyerForRespondent": Name(s) of the lawyer representing the respondents,
                "LegislationsReferred": Name(s) of the acts/ sections/ legislations/ Indian Penal Codes reffered in the current case,
                "Petitioner": Name(s) of the petitioners or appellants of the current case,
                "Respondent": Name(s) of the respondents or oppositions of the current case,
                
        The pairs of named entities and therir corresponding values should be given in python dictionary format (where both keys and values are string only). These named entities are multivalued i.e., more than one values can be present for them: ["BenchCoram", "CaseNo", "CasesReferredTo", "LawyerForPetitioner", "LawyerForRespondent", "Petitioner", "Respondent"]. For multivalued entities the values should be seperated by a semicolon delimeter (';'). For entities having no values write 'NA' as their value.
        An example of desired output looks like following:

            {{
                "BenchCoram": "Bhaskar Raj Pradhan",
                "CaseNo": "W.P. (C) No.07 of 2023",
                "CasesReferredTo": "Hari Prasad Sharma vs. Union of India & Ors; Hari Prasad Sharma vs. Union of India & Ors. RCC",
                "Court": "KARNATAKA HIGH COURT",
                "JudgmentDate": "08/09/2024",
                "LawyerForPetitioner": "Babita Kumari; Thupden G. Bhutia",
                "LawyerForRespondent": "Mohan Sharma; Natasha Pradhan; Purnima Subba; Sangita Pradhan; Shakil Raj Karki; Zangpo Sherpa",
                "LegislationsReferred": "N.H. Act, 1956",
                "Petitioner": "Hari Prasad Sharma",
                "Respondent": "Union of India; Union of India & Ors.",
            }}
        """
    }
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a legal assistant."},
            {"role": "user", "content": prompts[prompt_type]}
        ]
    )
    generated_answer = response.choices[0].message.content
    generated_answer = generated_answer.replace('python','').replace("```", "").strip()
    try:
        json_data = re.search(r'\{.*\}', generated_answer, flags=re.DOTALL)
        if json_data:
            json_string = json_data.group(0)
            res_dict = json.loads(json_string)
            return res_dict
        else:
           return generated_answer 
    except:
        return generated_answer

# Use OpenAI GPT for comparing extracted metatags
def compare_metatags(hc_order_metatags, sc_petition_metatags):
    prompt = f"""
    Compare the following metatag values obtained from the High Court Order and the Special Leave Petition Document to find inconsistencies:
    
    Metadata from High Court Order:
    {hc_order_metatags}
    
    Metadata from Special Leave Petition Document:
    {sc_petition_metatags}

    List any discrepancies and inconsistencies in the above data with respect to the corresponding fields.
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a legal assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    generated_answer = response.choices[0].message.content
    return generated_answer

# Use OpenAI GPT to check defects using the Defect List
def check_defects_using_list(sc_petition_text):
    defects = []
    response_string = ""

    for defect_code, description in defect_list.items():

        prompt = f"""
        Following is a "Possible Defect" that can occur while filing a Special Leave Petition:
        
        {description}
        ------------------------------------------- END of LIST --------------------------------------------

        Thoroughly scrutinize the following Special Leave Petition text, based on the aspects described in the above "List of Possible Defects" and precisely list down the defects present matching from the given criteria. 
        To do this scrutiny, focus mainly on the pages containing lots of metadata, the "SYNOPSIS" section, and the "MOST RESPECTFULLY SHOWETH" section of the Special Leave Petition:
        
        {sc_petition_text}
        ------------------------------------------- END of DOCUMENT --------------------------------------------

        Strictly categorize the defects found using points provided in the above list of defects.
        If you find that this "Possible Defect" is not found in the Special Leave Petition text, your response should be "{defect_code}. {description}: Defect Not Found"
        If you find that this "Possible Defect" is found in the Special Leave Petition text, your response should be "{defect_code}. {description}:" immediately followed by your findings.
        """

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a legal assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        generated_answer = response.choices[0].message.content
        defects.append(generated_answer)

    return defects

def process_file(file, slp, file_type, sc_or_hc):
    text = extract_text(file)
    file_name = f"SLP_{slp}_{file_type}_Text.txt"
    result_file = f"statementA_results_slp_{slp}.txt"
    write_mode = get_file_mode(result_file)
    
    with open(result_file, write_mode) as results, open(file_name, 'w') as output:
        results.write(f"{sc_or_hc} {file_type} {slp}: {text}\n{'-'*15}\n")
        output.write(text)
    return text

def process_slp_files(file, slp):
    return process_file(file, slp, "SC Petition", "SC")

def process_io_files(file, slp):
    return process_file(file, slp, "HC Order", "HC")

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
        defect_string = ""
        for defect in defect_check_result:
            defect_string += defect + "\n"
        final_output = f"Defects:\n {defect_string}\nComparison: {comparison_result}\n"

        with open(results_file, 'w') as results:
            results.write("********** DEFECT LIST **********\n")
            results.write(defect_list_text + "\n")
            results.write("********** HC ORDER **********\n")
            results.write(hc_order_text + "\n")
            results.write("********** SC PETITION **********\n")
            results.write(sc_petition_text + "\n")
            # Convert dictionaries to JSON-formatted strings
            results.write("********** HC METATAG **********\n")
            results.write(json.dumps(hc_metatags, indent=2) + "\n")  # Convert to string
            results.write("********** SC METATAG **********\n")
            results.write(json.dumps(sc_metatags, indent=2) + "\n")  # Convert to string
            results.write("********** DEFECT CHECK RESULT **********\n")
            results.write(defect_string)
            results.write("********** COMPARISON RESULT **********\n")
            results.write(comparison_result + "\n")
        st.write(final_output)
        print("We're done")

if __name__ == '__main__':
    main()

