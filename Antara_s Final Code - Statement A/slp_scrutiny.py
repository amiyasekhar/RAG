import openai
# import pytesseract
# from pdf2image import convert_from_bytes
# from PIL import Image
# from pdf2image import convert_from_bytes
# import os
import docx
import fitz  # PyMuPDF for handling PDF files
from rr_pipeline import * 
from threading import Thread
import json
import requests
import re

import pytesseract
from io import BytesIO
from pdf2image import convert_from_bytes
from PIL import Image



openai.api_key = '' ## add your API key gere



# Convert PDF to images
class simple_image_extract(object):
    """
    Extracts images using pdf2image library
    """

    def __init__(self, dpi: int = 300) -> None:
        self.dpi = dpi

    def extract_images(self, pdf_file: BytesIO,) -> str:
        """
        Given a path to a pdf file, extract and store images in a folder
        """
        try:
            pages = convert_from_bytes(pdf_file=pdf_file, dpi=self.dpi)
        except Exception as e:
            raise ValueError("Error extracting images from pdf") from e
        list_text = []

        # Iterate through pages
        for i, page in enumerate(pages):
            # Save each page as an image
            print(f"Starting to convert Page{i+1} to text")
            img_byte_arr = BytesIO()
            page.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()

            # Perform OCR directly on the bytes
            text = pytesseract.image_to_string(Image.open(BytesIO(img_byte_arr)))

            list_text.append(text)

        return "\n\n".join(list_text)





# Extracting text from DOCX, PDF, and TXT files
def extract_text(file):
    # Get the file extension to determine the file type
    file_extension = file.name.split('.')[-1].lower()
    text = ""
    # If it's a DOCX file, use python-docx to extract text
    if file_extension == "docx":
        doc = docx.Document(file)
        # with open(f"{file.name}.txt", 'w') as copy:
        #     copy.write(f"{file.name}\n")

        for para in doc.paragraphs:
            paragraph_text = para.text
            # with open(f"{file.name}.txt", 'a') as copy:
            #     copy.write(f"{paragraph_text}\n")
            text += paragraph_text + "\n"
    # If it's a PDF file, use PyMuPDF to extract text
    elif file_extension == "pdf":
        ### amiya's code
        # doc = fitz.open(stream=file.read(), filetype="pdf")  # Read PDF file
        # # with open(f"{file.name}.txt", 'w') as copy:
        # #     copy.write(f"{file.name}\n")
        # for page in doc:
        #     page_text = page.get_text()
        #     # with open(f"{file.name}.txt", 'a') as copy:
        #     #     copy.write(f"{page_text}\n")
        #     text += page_text + "\n"

        ### prateek's code for OCR
        text_extract = simple_image_extract()
        # with open(file.name, "rb") as f:
        #     content = f.read()
        content = file.read()
        text = text_extract.extract_images(content)

    # If it's a TXT file, handle the uploaded file directly
    elif file_extension == "txt":
        text = file.read().decode('utf-8')  # Reading content as a string
        # with open(f"{file.name}.txt", 'w') as copy:
        #     copy.write(f"{file.name}\n{text}\n")
    else:
        raise ValueError("Unsupported file format. Only DOCX, PDF, and TXT are supported.")
    return text


# Extract metatags from SC Petition
def extract_metatags_from_sc_petition(document_text):
    prompt = f""" Following is a Special Leave Petition to be filed to the Supreme Court of India: {document_text}
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
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a legal assistant."},
            {"role": "user", "content": prompt}
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






# Extract metatags from HC Petition
def extract_metatags_from_hc_order(document_text):
    prompt = f""" Following is a High court Judgment order: {document_text}
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
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a legal assistant."},
            {"role": "user", "content": prompt}
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
    # url = 'https://api.openai.com/v1/chat/completions'
    # headers = {
    #     'Authorization': f'Bearer {OPENAI_API_KEY}',
    #     'Content-Type': 'application/json'
    # }

    # data = {
    #     'model': 'gpt-4o',  # Specify the model you are using
    #     'messages': [
    #         {'role': 'system', 'content': 'You are a legal assistant.'},
    #         {"role": "user", "content": prompt}]
    # }
    # result_dict = dict()
    # response = requests.post(url, headers=headers, json=data)
    # if response.status_code == 200:
    #     response_dict = response.json()
    #     if 'choices' in response_dict and len(response_dict['choices']) > 0:
    #         content = response_dict['choices'][0]['message']['content']
    #         try:
    #             result_dict = json.loads(content)
    #             print(result_dict)
    #         except json.JSONDecodeError:
    #             print("Failed to parse response as JSON.")
    #     else:
    #         print("No choices found in the response.")
    # else:
    #     print(f"Request failed with status code {response.status_code}")
    #     print(response.text)
    # return result_dict



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
def check_defects_using_list(defect_list_text, sc_petition_text):
    prompt = f"""
    Following is a "List of Possible Defects" that can occur while filing a Special Leave Pettition:
    
    {defect_list_text}
    ------------------------------------------- END of LIST --------------------------------------------

    Thoroughly scrutinize the following Special Leave Petition text, based on the aspects described in the above "List of Possible Defects" and precisely list down the defects present matching from the given criterias. 
    To do this scrutiny, focus mainly in the pages containing lots of metadata, the "SYNOPSIS" section, and the "MOST RESPECTFULLY SHOWETH" section of the Special Leave Petition:
    
    {sc_petition_text}
    ------------------------------------------- END of DOCUMENT --------------------------------------------

    Strictly categorize the defects found using points provided in the above list of defects.
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



# Wrapper function to store the result in a shared list
def thread_function(func, args, result_list, index):
    result_list[index] = func(*args)  # Use *args to pass multiple arguments



def slp_analysis(uploaded_files):
    extract_results = [None] * 3
    t01 = Thread(target=thread_function, args=(extract_text, (uploaded_files[1],), extract_results, 0))
    t02 = Thread(target=thread_function, args=(extract_text, (uploaded_files[0],), extract_results, 1))
    t03 = Thread(target=thread_function, args=(extract_text, (uploaded_files[2],), extract_results, 2))
    t01.start()
    t02.start()
    t03.start()

    results = [None] * 5
    t01.join()
    hc_order_text = extract_results[0]
    t1 = Thread(target=thread_function, args=(extract_metatags_from_hc_order, (hc_order_text,), results, 0))
    t1.start()
    t3 = Thread(target=thread_function, args=(rr_call, (hc_order_text,), results, 2)) 
    t3.start()

    t02.join()
    sc_petition_text = extract_results[1]
    t2 = Thread(target=thread_function, args=(extract_metatags_from_sc_petition, (sc_petition_text,), results, 1))
    t2.start()
    t1.join()
    t2.join()

    metatags_hc_order, metatags_sc_petition = results[0], results[1]

    t4 = Thread(target=thread_function, args=(compare_metatags, (metatags_hc_order, metatags_sc_petition), results, 3))
    t4.start()

    t03.join()
    defect_list_text = extract_results[2]
    t5 = Thread(target=thread_function, args=(check_defects_using_list, (defect_list_text, sc_petition_text), results, 4))
    t5.start()

    t3.join()
    rr_hc_order = results[2][0]
    if isinstance(metatags_hc_order, dict):
        for k, v in rr_hc_order.items():
            if k in ['Facts', 'RLC', 'RPC']:
                metatags_hc_order[k] = v        
    elif isinstance(metatags_hc_order, str):
        rr_dict = {k:v for k,v in rr_hc_order.items() if k in ['Facts', 'RLC', 'RPC']}
        metatags_hc_order = metatags_hc_order + '\n' + str(rr_dict)

    t4.join()
    t5.join()
    comparison_result, defect_check_result = results[3], results[4]

    text_response = f"\n ------------------------- Metatag Analysis -------------------------  \n\n{comparison_result}\n\n ------------------------- Suggested defects to be corrected ------------------------- \n\n{defect_check_result}"
    return [metatags_sc_petition, metatags_hc_order, text_response]