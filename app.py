import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from utils import extract_text_from_file
import tempfile
import pandas as pd
from datetime import datetime
import json
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import numpy as np
import re

# Load environment variables
load_dotenv()

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Set page config
st.set_page_config(
    page_title="Proposal Quotation Generator",
    page_icon="📄",
    layout="wide"
)

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

def generate_quotation_prompt(examples, input_docs_text, similar_examples=None, template=None):
    # Format examples
    examples_text = ""
    for i, example in enumerate(examples, 1):
        examples_text += f"\nExample {i}:\n"
        examples_text += f"Input: {example['input_text']}\n"
        examples_text += f"Output: {example['output_text']}\n"
    
    # Format similar examples if available
    similar_examples_text = ""
    if similar_examples:
        similar_examples_text = "\nSimilar Examples from Previous Cases:\n"
        for i, example in enumerate(similar_examples, 1):
            similar_examples_text += f"\nSimilar Example {i}:\n{example.page_content}\n"
    
    # Chain of thought reasoning prompt
    chain_of_thought = """
Follow these steps carefully:

1. Template Analysis:
   - Identify all placeholders in the template
   - Understand the structure and formatting requirements
   - Note any specific sections or categories needed

2. Content Extraction:
   - Extract relevant information from input documents
   - Match extracted information to template placeholders
   - Identify any missing information that needs to be inferred

3. Validation:
   - Ensure all required sections are present
   - Verify that all placeholders are filled
   - Check for consistency in formatting and style
   - Validate numerical calculations and units

4. Final Review:
   - Cross-reference with example quotations
   - Ensure all pricing follows established patterns
   - Verify source references are accurate
   - Check for any missing or incomplete information
"""

    json_structure = '''{
    "sections": [
        {
            "name": "AV Equipment",
            "items": [
                {
                    "description": "string",
                    "quantity": number,
                    "unit_price": number,
                    "total_price": number,
                    "source": {
                        "document": "string",
                        "line_reference": "string"
                    }
                }
            ]
        }
    ],
    "total_amount": number
}'''
    
    prompt = f"""You are a smart proposal quotation document creator. I will provide you with examples and new input documents.
    
Previous Examples:
{examples_text}

{similar_examples_text}

New Input Documents:
{input_docs_text}

{chain_of_thought}

Please analyze both the examples and new input documents to create a detailed quotation that follows the same style and structure as the examples, but with content specific to the new input.

The quotation should include the following sections:
1. AV Equipment
2. Labor
3. Custom Fabrication
4. Creative Services

For each section, provide:
- Item description
- Quantity
- Unit price
- Total price
- Source reference (which document and line number the item was derived from)

Format the response as a JSON object with the following structure:
{json_structure}

Ensure all prices are in USD and include appropriate units for quantities. Follow the pricing patterns and item categorization from the examples while adapting to the specific requirements in the new input documents. For each item, include a reference to the source document and line number where the information was derived from."""
    
    return prompt

def create_excel_from_quotation(quotation_data):
    # Create a DataFrame for each section
    dfs = []
    for section in quotation_data['sections']:
        df = pd.DataFrame(section['items'])
        df['Section'] = section['name']
        # Add source information
        df['Source'] = df.apply(lambda row: f"{row['source']['document']} (Line: {row['source']['line_reference']})", axis=1)
        dfs.append(df)
    
    # Combine all sections
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Reorder columns
    final_df = final_df[['Section', 'description', 'quantity', 'unit_price', 'total_price', 'Source']]
    
    # Add total row
    total_row = pd.DataFrame({
        'Section': ['TOTAL'],
        'description': [''],
        'quantity': [''],
        'unit_price': [''],
        'total_price': [quotation_data['total_amount']],
        'Source': ['']
    })
    final_df = pd.concat([final_df, total_row], ignore_index=True)
    
    return final_df

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
    st.title("Quotation Generator")
    
    # Template settings in sidebar
    st.sidebar.header("Template Settings")
    template_mode = st.sidebar.radio(
        "Select Template Mode",
        ["Default Template", "Custom Template"]
    )
    
    if template_mode == "Custom Template":
        st.session_state.template = st.sidebar.text_area(
            "Enter your template (use placeholders like [COMPANY_NAME], [PROJECT_DETAILS], etc.)",
            height=200
        )
    else:
        st.session_state.template = """[COMPANY_NAME]
[ADDRESS]
[CONTACT_INFO]

QUOTATION

Date: [DATE]
Quotation #: [QUOTE_NUMBER]

Dear [CLIENT_NAME],

Thank you for your interest in our services. Please find below our quotation for [PROJECT_NAME]:

[PROJECT_DETAILS]

Total Amount: [AMOUNT]
Payment Terms: [PAYMENT_TERMS]
Validity: [VALIDITY_PERIOD]

Best regards,
[YOUR_NAME]
[YOUR_POSITION]"""
    
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
        if uploaded_files and st.button("Generate Quotation", key="generate_quotation_btn_chat"):
            with st.spinner("Generating quotation..."):
                # Process current input files
                input_texts = []
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(file.getvalue())
                        text = extract_text_from_file(tmp_file.name, file.name)
                        input_texts.append(text)
                    os.unlink(tmp_file.name)
                # Find similar examples using RAG
                combined_input = "\n\n".join(input_texts)
                similar_examples = find_similar_examples(combined_input) if st.session_state.vector_store else None
                # Display which examples are being used
                if similar_examples:
                    st.info(f"Using the following examples for context: {', '.join([f'Example {i+1}' for i in range(len(similar_examples))])}")
                # Prepare the actual text of the most relevant examples
                relevant_examples = []
                if similar_examples:
                    for sim in similar_examples:
                        # Find the original example whose text matches
                        for ex in st.session_state.examples:
                            if sim.page_content in ex['input_text'] or sim.page_content in ex['output_text']:
                                relevant_examples.append(ex)
                                break
                else:
                    relevant_examples = st.session_state.examples
                # Generate quotation
                model = initialize_model()
                prompt = generate_quotation_prompt(
                    relevant_examples,
                    combined_input,
                    similar_examples,
                    st.session_state.template
                )
                response = model.generate_content(prompt)
                try:
                    # Parse the JSON response
                    try:
                        quotation_data = json.loads(response.text)
                    except json.JSONDecodeError:
                        # Try to extract JSON block from the response
                        match = re.search(r'\{[\s\S]*\}', response.text)
                        if match:
                            try:
                                quotation_data = json.loads(match.group(0))
                            except Exception:
                                quotation_data = None
                        else:
                            quotation_data = None
                    if quotation_data:
                        # Create and display the Excel data
                        df = create_excel_from_quotation(quotation_data)
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'role': 'user',
                            'content': f"Generated quotation for {len(uploaded_files)} files"
                        })
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': "Here's the generated quotation:"
                        })
                        # Create Excel file
                        excel_buffer = pd.ExcelWriter('quotation.xlsx', engine='xlsxwriter')
                        df.to_excel(excel_buffer, index=False, sheet_name='Quotation')
                        worksheet = excel_buffer.sheets['Quotation']
                        for i, col in enumerate(df.columns):
                            max_length = max(df[col].astype(str).apply(len).max(), len(col)) + 2
                            worksheet.set_column(i, i, max_length)
                        excel_buffer.close()
                        with open('quotation.xlsx', 'rb') as f:
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': st.download_button(
                                    label="Download Quotation (Excel)",
                                    data=f,
                                    file_name=f"quotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            })
                    else:
                        st.error("Error parsing the AI response. Please try again.")
                        st.text_area("Raw AI Response", response.text, height=300)
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    st.text_area("Raw AI Response", response.text, height=300)
    with add_example_col:
        if st.button("Add Example", key="add_example_btn_fab"):
            st.session_state.show_example_overlay = True
            st.rerun()

if __name__ == "__main__":
    main() 