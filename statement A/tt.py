import streamlit as st

# Function to parse the text file and extract defects dynamically
def extract_defects(text):
    defects_section = []
    in_defects = False
    for line in text.splitlines():
        if "Defect Check Result:" in line:
            in_defects = True
        elif "Final Output:" in line:
            break
        elif in_defects:
            defects_section.append(line)
    
    return "\n".join(defects_section)

# Streamlit function to display defects in an organized and customized way
def display_defect_check_result(file_path):
    # Read the contents of the file
    with open(file_path, "r") as file:
        text = file.read()
    
    # Extract defects
    defects = extract_defects(text)

    # Define custom HTML with CSS for the box
    box_html = f"""
    <div style="
        width: 100%; 
        height: 400px; 
        border: 2px solid #4CAF50; 
        padding: 10px; 
        overflow: scroll; 
        white-space: pre-wrap;
        overflow-x: auto; 
        overflow-y: auto;
        ">
        {defects}
    </div>
    """

    # Display the box with defects using Streamlit's markdown function
    st.markdown(box_html, unsafe_allow_html=True)

# Streamlit app
def main():
    st.title("SLP Defect Check")
    
    uploaded_file = st.file_uploader("Choose a text file", type="txt")
    if uploaded_file is not None:
        file_path = f"./{uploaded_file.name}"
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        display_defect_check_result(file_path)

if __name__ == "__main__":
    main()
