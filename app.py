import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from utils import extract_text_from_file
import tempfile
import pandas as pd
from datetime import datetime
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import numpy as np
import re
import base64
import time
import random


from components.generate_quotation import generate_quotation_prompt
from components.extract_json import extract_json_from_response
from components.create_excel import create_excel_from_quotation



# Load environment variables
load_dotenv()

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Set page config
st.set_page_config(
    page_title="SWAI - Smart Quotation Generator",
    page_icon="Asserts/logo.webp",
    layout="wide"
)

# Add logo and title with improved styling
st.markdown("""
    <style>
    .logo-container {
        display: flex;
        align-items: center;
        padding: 1rem 0;
    }
    .logo-img {
        max-height: 60px;
        width: auto;
        margin-right: 1rem;
    }
    .title-text {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

# Create a container for logo and title
st.markdown("""
    <div class="logo-container">
        <img src="data:image/webp;base64,{}" class="logo-img">
        <h1 class="title-text">Smart Quotation Generator</h1>
    </div>
    """.format(base64.b64encode(open("Asserts/logo.webp", "rb").read()).decode()), unsafe_allow_html=True)

# Initialize session state
if 'examples' not in st.session_state:
    st.session_state.examples = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_example_overlay' not in st.session_state:
    st.session_state.show_example_overlay = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'template' not in st.session_state:
    st.session_state.template = None
if 'current_quotation_data' not in st.session_state:
    st.session_state.current_quotation_data = None
if 'current_excel_file' not in st.session_state:
    st.session_state.current_excel_file = None
if 'current_uploaded_files' not in st.session_state:
    st.session_state.current_uploaded_files = None

def initialize_model():
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model

def create_vector_store(documents):
    """Create a vector store from documents for RAG"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    texts = []
    for doc in documents:
        texts.extend(text_splitter.split_text(doc['input_text']))
        texts.extend(text_splitter.split_text(doc['output_text']))
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_texts(texts, embeddings)
    return vector_store

def find_similar_examples(query, k=3):
    """Find similar examples using vector similarity"""
    if st.session_state.vector_store is None:
        return []
    
    docs = st.session_state.vector_store.similarity_search(query, k=k)
    return docs

def process_documents_with_gemini(documents):
    """Process documents using Gemini's file handling capabilities"""
    client = genai.Client()
    processed_files = []
    
    for doc in documents:
        file_data = io.BytesIO(doc['file'].getvalue())
        processed_file = client.files.upload(
            file=file_data,
            config=dict(mime_type=f'application/{doc["file"].name.split(".")[-1]}')
        )
        processed_files.append(processed_file)
    
    return processed_files

def extract_template_placeholders(template_text):
    """Extract all placeholders from the template text"""
    # Match placeholders in format [PLACEHOLDER_NAME]
    placeholders = re.findall(r'\[([A-Z_]+)\]', template_text)
    return list(set(placeholders))  # Remove duplicates

def map_data_to_template(template_text, quotation_data):
    """Map quotation data to template placeholders"""
    # Extract all placeholders from template
    placeholders = extract_template_placeholders(template_text)
    
    # Create a mapping of placeholder to value
    placeholder_map = {}
    
    # Map basic information
    placeholder_map['DATE'] = datetime.now().strftime('%Y-%m-%d')
    placeholder_map['QUOTE_NUMBER'] = f"Q{datetime.now().strftime('%Y%m%d%H%M')}"
    
    # Map company information if available
    if 'company_info' in quotation_data:
        placeholder_map['COMPANY_NAME'] = quotation_data['company_info'].get('name', '')
        placeholder_map['ADDRESS'] = quotation_data['company_info'].get('address', '')
        placeholder_map['CONTACT_INFO'] = quotation_data['company_info'].get('contact', '')
    
    # Map client information if available
    if 'client_info' in quotation_data:
        placeholder_map['CLIENT_NAME'] = quotation_data['client_info'].get('name', '')
        placeholder_map['PROJECT_NAME'] = quotation_data['client_info'].get('project_name', '')
    
    # Map project details
    project_details = []
    for section in quotation_data['sections']:
        section_text = f"\n{section['name']}:\n"
        for subsection in section['subsections']:
            section_text += f"\n{subsection['name']}:\n"
            for item in subsection['items']:
                if 'description' in item:
                    section_text += f"- {item['description']}: {item['quantity']} x ${item['unit_price']} = ${item['total_price']}\n"
                elif 'role' in item:
                    section_text += f"- {item['role']} ({item['task']}): {item['hours']}hrs x ${item['rate']} = ${item['total_price']}\n"
        project_details.append(section_text)
    
    placeholder_map['PROJECT_DETAILS'] = "\n".join(project_details)
    placeholder_map['AMOUNT'] = f"${quotation_data['total_amount']:,.2f}"
    
    # Map payment terms and validity if available
    if 'terms' in quotation_data:
        placeholder_map['PAYMENT_TERMS'] = quotation_data['terms'].get('payment_terms', 'Net 30')
        placeholder_map['VALIDITY_PERIOD'] = quotation_data['terms'].get('validity', '30 days')
    
    # Replace all placeholders in template
    filled_template = template_text
    for placeholder in placeholders:
        value = placeholder_map.get(placeholder, f'[{placeholder}]')
        filled_template = filled_template.replace(f'[{placeholder}]', value)
    
    return filled_template

def retry_with_backoff(func, max_retries=3, initial_delay=1, max_delay=10):
    """Retry a function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            
            # Check if it's a model overload or timeout error
            if "503" in str(e) or "504" in str(e) or "overload" in str(e).lower():
                delay = min(initial_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                time.sleep(delay)
                continue
            else:
                raise e

def generate_quotation_with_retry(model, prompt):
    """Generate quotation with retry logic"""
    def _generate():
        response = model.generate_content(prompt)
        if not response or not response.text:
            raise ValueError("Empty response from model")
        return response.text
    
    return retry_with_backoff(_generate)


def validate_quotation_line_count(quotation_data):
    """Validate that the quotation has sufficient line items"""
    total_lines = 0
    for section in quotation_data['sections']:
        for subsection in section['subsections']:
            total_lines += len(subsection['items'])
    return total_lines >= 150


def display_chat_message(message):
    """Display a chat message with appropriate styling"""
    if message['role'] == 'user':
        st.write(f"👤 You: {message['content']}")
    else:
        st.write(f"🤖 Assistant: {message['content']}")

def display_examples():
    """Display the list of examples with expandable details"""
    if st.session_state.examples:
        st.subheader("📚 Example Pairs")
        for i, example in enumerate(st.session_state.examples, 1):
            with st.expander(f"Example {i}"):
                st.write("Input Files:")
                for file_name in example['input_file_names']:
                    st.write(f"- {file_name}")
                
                st.write("Output File:")
                st.write(f"- {example['output_file_name']}")
                
                if st.checkbox(f"Show extracted text for Example {i}", key=f"show_text_{i}"):
                    st.write("Input Text:")
                    st.text_area("", example['input_text'], height=200, key=f"input_text_{i}")
                    st.write("Output Text:")
                    st.text_area("", example['output_text'], height=200, key=f"output_text_{i}")

def main():
    # st.title("Quotation Generator")
    
    # Template settings in sidebar
    st.sidebar.header("Template Settings")
    template_mode = st.sidebar.radio(
        "Select Template Mode",
        ["Default Template", "Custom Template"]
    )
    
    if template_mode == "Custom Template":
        template_file = st.sidebar.file_uploader(
            "Upload Template File",
            type=['xlsx', 'xls'],
            key="template_uploader"
        )
        if template_file:
            # Store the template file in session state
            st.session_state.template = template_file
            try:
                # Create a temporary file to read the Excel
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                    tmp_file.write(template_file.getvalue())
                    tmp_file.flush()
                    df = pd.read_excel(tmp_file.name)
                    os.unlink(tmp_file.name)
                st.sidebar.write("Template Columns:")
                st.sidebar.write(df.columns.tolist())
            except Exception as e:
                st.error(f"Error reading template: {str(e)}")
    else:
        st.session_state.template = None
    
    # Main chat interface container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(message)
    
    # Example overlay/modal
    if st.session_state.show_example_overlay:
        with st.container():
            st.markdown("---")
            st.subheader("Add Example Pair")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Input Files (PDF, DOCX)")
                example_input_files = st.file_uploader(
                    "Upload input files",
                    type=['pdf', 'docx'],
                    accept_multiple_files=True,
                    key="example_input_uploader_modal"
                )
            with col2:
                st.write("Output File (XLSX)")
                example_output_file = st.file_uploader(
                    "Upload output file",
                    type=['xlsx'],
                    key="example_output_uploader_modal"
                )
            save_col, cancel_col = st.columns([1,1])
            with save_col:
                if st.button("Save Example", key="save_example_btn_modal"):
                    if example_input_files and example_output_file:
                        # Process input files
                        input_texts = []
                        for file in example_input_files:
                            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                                tmp_file.write(file.getvalue())
                                text = extract_text_from_file(tmp_file.name, file.name)
                                input_texts.append(text)
                            os.unlink(tmp_file.name)
                        # Process output file
                        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                            tmp_file.write(example_output_file.getvalue())
                            output_text = extract_text_from_file(tmp_file.name, example_output_file.name)
                            os.unlink(tmp_file.name)
                        # Check if parsing was successful
                        if not any(input_texts) or not output_text:
                            st.error("Failed to parse one or more files. Please check your uploads.")
                        else:
                            new_example = {
                                'input_files': example_input_files,
                                'output_file': example_output_file,
                                'input_text': "\n\n".join(input_texts),
                                'output_text': output_text,
                                'input_file_names': [file.name for file in example_input_files],
                                'output_file_name': example_output_file.name
                            }
                            print("Example parsed:", new_example['input_file_names'], new_example['output_file_name'])
                            st.session_state.examples.append(new_example)
                            # Update vector store
                            st.session_state.vector_store = create_vector_store(st.session_state.examples)
                            st.success("Example parsed and indexed successfully! Ready for use in generation.")
                            st.session_state.show_example_overlay = False
                            st.rerun()
            with cancel_col:
                if st.button("Cancel", key="cancel_example_btn_modal"):
                    st.session_state.show_example_overlay = False
                    st.rerun()
    
    # Bottom input area for chat and add example
    st.markdown("---")
    chat_col, add_example_col = st.columns([4, 1])
    with chat_col:
        uploaded_files = st.file_uploader(
            "Upload files to generate quotation",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            key="current_input_uploader_chat"
        )
        
        # Check if files have changed
        if uploaded_files != st.session_state.current_uploaded_files:
            st.session_state.current_uploaded_files = uploaded_files
            st.session_state.current_quotation_data = None
            st.session_state.current_excel_file = None
        
        if uploaded_files and st.button("Generate Quotation", key="generate_quotation_btn_chat"):
            # Create tabs for different views immediately
            tab1, tab2, tab3 = st.tabs(["Excel Preview", "JSON Data", "Download"])
            
            # Initialize progress container
            progress_container = st.empty()
            
            try:
                # Process current input files
                with progress_container:
                    with st.spinner("Processing input files..."):
                        input_texts = []
                        for file in uploaded_files:
                            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                                tmp_file.write(file.getvalue())
                                text = extract_text_from_file(tmp_file.name, file.name)
                                if not text or len(text.strip()) == 0:
                                    st.error(f"Could not extract text from {file.name}. Please check if the file is valid.")
                                    return
                                input_texts.append(text)
                            os.unlink(tmp_file.name)
                        
                        if not input_texts:
                            st.error("No text could be extracted from the uploaded files.")
                            return
                
                # Find similar examples using RAG
                with progress_container:
                    with st.spinner("Finding similar examples..."):
                        combined_input = "\n\n".join(input_texts)
                        similar_examples = find_similar_examples(combined_input) if st.session_state.vector_store else None
                        
                        if similar_examples:
                            st.info(f"Using the following examples for context: {', '.join([f'Example {i+1}' for i in range(len(similar_examples))])}")
                
                # Prepare the actual text of the most relevant examples
                relevant_examples = []
                if similar_examples:
                    for sim in similar_examples:
                        for ex in st.session_state.examples:
                            if sim.page_content in ex['input_text'] or sim.page_content in ex['output_text']:
                                relevant_examples.append(ex)
                                break
                else:
                    relevant_examples = st.session_state.examples
                
                # Generate quotation with retry logic
                with progress_container:
                    with st.spinner("Generating quotation (this may take a few attempts if the model is busy)..."):
                        model = initialize_model()
                        prompt = generate_quotation_prompt(
                            relevant_examples,
                            combined_input,
                            similar_examples,
                            st.session_state.template
                        )
                        
                        try:
                            response_text = generate_quotation_with_retry(model, prompt)
                            if not response_text:
                                st.error("Received empty response from model. Please try again.")
                                return
                        except Exception as e:
                            if "503" in str(e) or "overload" in str(e).lower():
                                st.error("The model is currently overloaded. Please try again in a few minutes.")
                            elif "504" in str(e):
                                st.warning("The request timed out. Please try again.")
                            else:
                                st.error(f"An error occurred: {str(e)}")
                            with tab2:
                                st.text_area("Error Details", str(e), height=300)
                            return
                        
                        if not response_text:
                            st.error("Failed to generate quotation. Please try again.")
                            return
                        
                        # Print raw response to terminal for debugging
                        print("\n=== Raw AI Response ===")
                        print(response_text)
                
                # Extract and parse the response
                with progress_container:
                    with st.spinner("Processing response..."):
                        try:
                            quotation_data = extract_json_from_response(response_text)
                            if not quotation_data:
                                st.error("Failed to parse the AI response into a valid quotation format.")
                                with tab2:
                                    st.text_area("Raw AI Response", response_text, height=300)
                                return
                            
                            # Generate Excel file first
                            try:
                                excel_file = create_excel_from_quotation(quotation_data, st.session_state.template)
                                
                                # Only store data if Excel creation succeeds
                                st.session_state.current_quotation_data = quotation_data
                                st.session_state.current_excel_file = excel_file
                                
                                # Show JSON data
                                with tab2:
                                    st.json(quotation_data, expanded=True)
                                
                                # Show Excel preview with proper formatting
                                with tab1:
                                    try:
                                        df = pd.read_excel(excel_file, sheet_name=None)
                                        if not df:
                                            st.error("Failed to read the generated Excel file.")
                                            return
                                        
                                        for sheet_name, sheet_df in df.items():
                                            st.subheader(sheet_name)
                                            if not sheet_df.empty:
                                                # Format numeric columns
                                                for col in sheet_df.columns:
                                                    if 'Price' in col or 'Amount' in col:
                                                        sheet_df[col] = pd.to_numeric(sheet_df[col], errors='coerce')
                                                    elif 'Qty' in col or 'Quantity' in col:
                                                        sheet_df[col] = pd.to_numeric(sheet_df[col], errors='coerce')
                                                
                                                # Display the dataframe with proper formatting
                                                st.dataframe(
                                                    sheet_df,
                                                    use_container_width=True,
                                                    hide_index=True
                                                )
                                            else:
                                                st.warning(f"Sheet '{sheet_name}' is empty.")
                                    except Exception as e:
                                        st.error(f"Error displaying Excel preview: {str(e)}")
                                        st.text_area("Error Details", str(e), height=300)
                                
                                # Add download button with proper styling
                                with tab3:
                                    try:
                                        with open(excel_file, 'rb') as f:
                                            excel_data = f.read()
                                            # Use a unique key for the download button to prevent reload
                                            download_key = f"download_btn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                            st.download_button(
                                                label="📥 Download Quotation (Excel)",
                                                data=excel_data,
                                                file_name=f"quotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                use_container_width=True,
                                                key=download_key
                                            )
                                    except Exception as e:
                                        st.error(f"Error creating download button: {str(e)}")
                                        st.text_area("Error Details", str(e), height=300)
                            except Exception as e:
                                st.error(f"Error creating Excel file: {str(e)}")
                                with tab2:
                                    st.text_area("Raw AI Response", response_text, height=300)
                        except Exception as e:
                            st.error(f"Error parsing the AI response: {str(e)}")
                            with tab2:
                                st.text_area("Raw AI Response", response_text, height=300)
                            return
                
                # Clear progress container
                progress_container.empty()
                
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                if 'response_text' in locals():
                    with tab2:
                        st.text_area("Raw AI Response", response_text, height=300)
                else:
                    with tab2:
                        st.text_area("Raw AI Response", "No response generated", height=300)
        elif st.session_state.current_quotation_data and st.session_state.current_excel_file:
            # Display existing quotation if available
            tab1, tab2, tab3 = st.tabs(["Excel Preview", "JSON Data", "Download"])
            
            # Show JSON data
            with tab2:
                st.json(st.session_state.current_quotation_data, expanded=True)
            
            # Show Excel preview
            with tab1:
                try:
                    df = pd.read_excel(st.session_state.current_excel_file, sheet_name=None)
                    if not df:
                        st.error("Failed to read the generated Excel file.")
                        return
                    
                    for sheet_name, sheet_df in df.items():
                        st.subheader(sheet_name)
                        if not sheet_df.empty:
                            # Format numeric columns
                            for col in sheet_df.columns:
                                if 'Price' in col or 'Amount' in col:
                                    sheet_df[col] = pd.to_numeric(sheet_df[col], errors='coerce')
                                elif 'Qty' in col or 'Quantity' in col:
                                    sheet_df[col] = pd.to_numeric(sheet_df[col], errors='coerce')
                            
                            # Display the dataframe with proper formatting
                            st.dataframe(
                                sheet_df,
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.warning(f"Sheet '{sheet_name}' is empty.")
                except Exception as e:
                    st.error(f"Error displaying Excel preview: {str(e)}")
            
            # Show download button
            with tab3:
                try:
                    with open(st.session_state.current_excel_file, 'rb') as f:
                        excel_data = f.read()
                        # Use a unique key for the download button to prevent reload
                        download_key = f"download_btn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        st.download_button(
                            label="📥 Download Quotation (Excel)",
                            data=excel_data,
                            file_name=f"quotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key=download_key
                        )
                except Exception as e:
                    st.error(f"Error creating download button: {str(e)}")
    
    with add_example_col:
        if st.button("Add Example", key="add_example_btn_fab"):
            st.session_state.show_example_overlay = True
            st.rerun()

if __name__ == "__main__":
    main() 