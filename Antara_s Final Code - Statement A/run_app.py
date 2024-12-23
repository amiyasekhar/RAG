import streamlit as st
from slp_scrutiny import *

st.set_page_config(layout="wide")


def navigate(page):
    st.session_state.page = page
    st.rerun()


def navigate_v2(page, metatags_sc_petition, metatags_hc_order, text_response):
    st.session_state.slp_tag_values = metatags_sc_petition
    st.session_state.hc_tag_values = metatags_hc_order
    st.session_state.text_response = text_response
    st.session_state.page = page
    st.rerun()




# Initializing session state variables
if 'slp_tag_values' not in st.session_state:
    st.session_state.slp_tag_values = None

if 'hc_tag_values' not in st.session_state:
    st.session_state.hc_tag_values = None

if 'text_response' not in st.session_state:
    st.session_state.text_response = None

if 'page' not in st.session_state:
    st.session_state.page = 1


# Sidebar navigation
# st.sidebar.title("Navigation")
# st.sidebar.button("SLP Analysis", on_click=lambda: navigate(1))


if st.session_state.page == 1:
    st.title("Upload Documents for SLP Scrutiny and Analysis")
    slp_doc = st.file_uploader("Upload SLP document", type=["pdf", "docx", "txt"])
    hc_order = st.file_uploader("Upload High Court Order", type=["pdf", "docx", "txt"])
    defect_list = st.file_uploader("Upload Defects List", type=["pdf", "docx", "txt"])
    submit_button = st.button(label='Generate Response', type="primary")
    if submit_button:
        uploaded_files = None
        if slp_doc != None and hc_order != None and defect_list != None:
            uploaded_files = [slp_doc, hc_order, defect_list]
        if uploaded_files is not None:
            metatags_sc_petition, metatags_hc_order, text_response = slp_analysis(uploaded_files)
            navigate_v2(2, metatags_sc_petition, metatags_hc_order, text_response)
        else:
            msg = "Please upload all of the three required documents in TEXT/ DOCX/ PDF format"
            st.warning(msg)



elif st.session_state.page == 2:
    st.title("SLP Analysis")
    if st.session_state.slp_tag_values != None and st.session_state.hc_tag_values != None:
        col1, col2 = st.columns([0.5,0.5])
        with col1:
            cname = "Metadata Extracted from SLP Document"
            st.markdown(f"<div style='font-size:20px; font-weight:bold; color:#ff6666'>{cname}</div>", unsafe_allow_html=True)
            # labels = sorted(["Court", "CaseNo", "Petitioner", "Respondent", "JudgmentDate", "JudgmentBy", "BenchCoram", "LawyerForPetitioner", "LawyerForRespondent"])
            data_dict1 = st.session_state.slp_tag_values
            if isinstance(data_dict1, dict):
                st.json(data_dict1, expanded=2)
            else:
                st.text_area("",str(data_dict1), height=400, disabled=False)
        with col2:
            cname = "Metadata Extracted from High Court Order"
            st.markdown(f"<div style='font-size:20px; font-weight:bold; color:#ff6666'>{cname}</div>", unsafe_allow_html=True)
            data_dict2 = st.session_state.hc_tag_values
            if isinstance(data_dict2, dict):
                st.json(data_dict2, expanded=2)
            else:
                st.text_area("",str(data_dict2), height=400, disabled=False)
    if st.session_state.text_response != None:
        cname = "Detected defects and Suggestions"
        st.markdown(f"<div style='font-size:22px; font-weight:bold; color:#ff3358'>{cname}</div>", unsafe_allow_html=True)
        llm_response = str(st.session_state.text_response)
    st.text_area("",llm_response, height=500, disabled=False)
    st.write('\n\n')
    submit_button = st.button(label='New Case Upload', type="primary")
    if submit_button:
        navigate(1)
