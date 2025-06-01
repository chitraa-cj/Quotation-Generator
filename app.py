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
import base64
import time
import random

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
    
    # Extract template placeholders if template is provided
    template_analysis = ""
    if template and hasattr(template, 'name') and template.name.endswith('.xlsx'):
        try:
            # Handle UploadedFile object
            if hasattr(template, 'getvalue'):
                # Create a temporary file to read the Excel
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                    tmp_file.write(template.getvalue())
                    tmp_file.flush()
                    df = pd.read_excel(tmp_file.name)
                    os.unlink(tmp_file.name)
            else:
                df = pd.read_excel(template)
            
            # Extract column names as placeholders
            placeholders = list(df.columns)
            template_analysis = f"\nTemplate Analysis:\nThe provided template contains the following columns: {', '.join(placeholders)}\n"
            template_analysis += "Ensure all these columns are properly filled with relevant information from the input documents.\n"
        except Exception as e:
            print(f"Error analyzing template: {str(e)}")
            template_analysis = "\nTemplate Analysis:\nCould not analyze template structure. Proceeding with default format.\n"
    
    # Chain of thought reasoning prompt
    chain_of_thought = f"""
Follow these steps carefully:

1. Template Analysis:
   - Identify all columns in the template
   - Understand the structure and formatting requirements
   - Note any specific sections or categories needed
   {template_analysis}

2. Content Extraction:
   - Extract relevant information from input documents
   - Match extracted information to template columns
   - Identify any missing information that needs to be inferred

3. Granular Decomposition:
   - Break down each major item into its smallest possible components
   - For AV equipment:
     * Main equipment (cameras, projectors, etc.)
     * Each camera needs: body, lens, battery, memory card, case
     * Each projector needs: unit, mount, cables, remote, spare lamp
     * Each screen needs: screen, frame, mounting hardware
     * Each audio system needs: mixer, speakers, microphones, stands, cables
     * Each lighting system needs: fixtures, controllers, cables, stands, gels
   - For labor:
     * Break down by role, time block, and specific task
     * Include setup, operation, and teardown for each item
     * Add prep time for each piece of equipment
     * Include travel time between locations
     * Add buffer time for contingencies
   - For rooms/areas:
     * Break down each room's requirements separately
     * Include all equipment needed for each room
     * Add setup and teardown labor for each room
     * Include room-specific consumables
   - For supporting items:
     * Include all necessary cables and connectors
     * Add mounting hardware and brackets
     * Include cases and protective equipment
     * Add spare parts and consumables
     * Include testing and calibration equipment

4. Validation:
   - Ensure all required sections are present
   - Verify that all template columns are filled
   - Check for consistency in formatting and style
   - Validate numerical calculations and units
   - Confirm all components are properly categorized
   - Verify minimum line item count (150-200 lines)
   - Ensure no major components are missing

5. Final Review:
   - Cross-reference with example quotations
   - Ensure all pricing follows established patterns
   - Verify source references are accurate
   - Check for any missing or incomplete information
   - Confirm proper grouping and sub-categorization
   - Verify template synchronization
   - Count total line items to ensure sufficient detail
"""

    json_structure = '''{
    "company_info": {
        "name": "string",
        "address": "string",
        "contact": "string"
    },
    "client_info": {
        "name": "string",
        "project_name": "string"
    },
    "sections": [
        {
            "name": "AV Equipment",
            "subsections": [
                {
                    "name": "Main Equipment",
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
            ]
        }
    ],
    "terms": {
        "payment_terms": "string",
        "validity": "string"
    },
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

The quotation should be highly detailed and granular, breaking down each major item into its smallest possible components. For example:

1. AV Equipment should be broken down into:
   - Main equipment (cameras, projectors, etc.)
   - Each camera needs: body, lens, battery, memory card, case
   - Each projector needs: unit, mount, cables, remote, spare lamp
   - Each screen needs: screen, frame, mounting hardware
   - Each audio system needs: mixer, speakers, microphones, stands, cables
   - Each lighting system needs: fixtures, controllers, cables, stands, gels
   - All necessary cables and connectors
   - All mounting hardware and brackets
   - All cases and protective equipment
   - All spare parts and consumables
   - All testing and calibration equipment

2. Labor should be detailed by:
   - Role (Camera Op, A1, Stagehand, etc.)
   - Time block (Load-in, Day 1, Day 2, etc.)
   - Specific task
   - Hours and rates
   - Setup time for each piece of equipment
   - Operation time for each piece of equipment
   - Teardown time for each piece of equipment
   - Prep time for each piece of equipment
   - Travel time between locations
   - Buffer time for contingencies
   - Per diem and meal buyouts

3. Each room or area should have its own detailed breakdown:
   - All required equipment
   - Setup and teardown labor
   - Supporting items
   - Consumables
   - Room-specific testing and calibration

4. Include all supporting services:
   - Freight and shipping
   - Equipment prep and testing
   - Warehouse pulls
   - Local union charges
   - Miscellaneous consumables
   - Insurance and permits
   - Quality control and testing

Format the response as a JSON object with the following structure:
{json_structure}

Ensure all prices are in USD and include appropriate units for quantities. Follow the pricing patterns and item categorization from the examples while adapting to the specific requirements in the new input documents. For each item, include a reference to the source document and line number where the information was derived from.

IMPORTANT: The quotation MUST contain at least 150-200 detailed line items. Break down every component into its smallest possible parts to achieve this level of detail. Do not combine items that should be separate line items.

{template_analysis}"""
    
    return prompt

def validate_quotation_line_count(quotation_data):
    """Validate that the quotation has sufficient line items"""
    total_lines = 0
    for section in quotation_data['sections']:
        for subsection in section['subsections']:
            total_lines += len(subsection['items'])
    return total_lines >= 150

def extract_json_from_response(response_text):
    """Extract and parse JSON from the AI response with multiple fallback mechanisms"""
    print("\n=== Raw Response Text ===")
    print(response_text)
    print("=== End Raw Response Text ===\n")
    
    def sanitize_text(text):
        """Sanitize text to ensure it's valid for JSON parsing"""
        # Remove any non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char in '\n\r\t')
        # Remove any BOM or special characters at the start
        text = text.lstrip('\ufeff')
        # Remove any leading/trailing whitespace
        text = text.strip()
        return text
    
    def extract_json_block(text):
        """Extract the largest valid JSON block from text"""
        # Find all potential JSON blocks
        json_blocks = []
        depth = 0
        start = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0 and start != -1:
                        json_blocks.append(text[start:i+1])
                        start = -1
        
        # Return the largest block
        if json_blocks:
            return max(json_blocks, key=len)
        return None
    
    def fix_missing_commas(text):
        """Fix missing commas in JSON text"""
        # Fix missing commas between properties
        text = re.sub(r'"\s*"', '","', text)  # Between strings
        text = re.sub(r'}\s*{', '},{', text)  # Between objects
        text = re.sub(r']\s*\[', '],[', text)  # Between arrays
        text = re.sub(r'"\s*{', '":{', text)  # After property name
        text = re.sub(r'"\s*\[', '":[', text)  # After property name before array
        
        # Fix missing commas after values
        text = re.sub(r'"\s*"', '","', text)  # After string
        text = re.sub(r'(\d+)\s*"', r'\1,"', text)  # After number
        text = re.sub(r'true\s*"', r'true,"', text)  # After true
        text = re.sub(r'false\s*"', r'false,"', text)  # After false
        text = re.sub(r'null\s*"', r'null,"', text)  # After null
        
        # Fix missing commas in arrays and objects
        text = re.sub(r'"\s*]', '"]', text)  # Before array end
        text = re.sub(r'"\s*}', '"}', text)  # Before object end
        text = re.sub(r'(\d+)\s*]', r'\1]', text)  # Number before array end
        text = re.sub(r'(\d+)\s*}', r'\1}', text)  # Number before object end
        
        # Fix specific delimiter issues
        text = re.sub(r'([^,{])\s*"([^"]+)"\s*:', r'\1,"\2":', text)  # Missing comma before property
        text = re.sub(r'([^,{])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1,"\2":', text)  # Missing comma before unquoted property
        
        return text
    
    def clean_json_text(text):
        """Clean and fix JSON text with multiple passes"""
        # First pass: Basic cleaning
        text = sanitize_text(text)
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Second pass: Extract JSON block
        json_block = extract_json_block(text)
        if json_block:
            text = json_block
        
        # Third pass: Fix common JSON issues
        fixes = [
            # Fix property names
            (r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":'),
            # Fix missing colons
            (r'"\s*{', '":{'),
            (r'"\s*\[', ':['),
            # Fix string values
            (r':\s*([a-zA-Z_][a-zA-Z0-9_]*)([,\s}])', r':"\1"\2'),
            # Fix boolean and null
            (r':\s*true([,\s}])', r':true\1'),
            (r':\s*false([,\s}])', r':false\1'),
            (r':\s*null([,\s}])', r':null\1'),
            # Fix numeric values
            (r':\s*(\d+\.?\d*)([,\s}])', r':\1\2'),
            # Fix missing commas
            (r'"\s*"', '","'),
            (r'}\s*{', '},{'),
            (r']\s*\[', '],['),
            (r'(\d+)\s*"', r'\1,"'),
            (r'true\s*"', r'true,"'),
            (r'false\s*"', r'false,"'),
            (r'null\s*"', r'null,"'),
            # Fix nested structures
            (r'}\s*]', '}]'),
            (r']\s*}', ']}'),
            # Remove trailing commas
            (r',\s*}', '}'),
            (r',\s*]', ']')
        ]
        
        for pattern, replacement in fixes:
            text = re.sub(pattern, replacement, text)
        
        # Fourth pass: Fix missing commas
        text = fix_missing_commas(text)
        
        # Fifth pass: Ensure proper JSON structure
        if not text.startswith('{'):
            text = '{' + text
        if not text.endswith('}'):
            text = text + '}'
        
        return text.strip()
    
    def fix_json_structure(text):
        """Fix JSON structure with balanced brackets and proper nesting"""
        stack = []
        fixed_text = ""
        in_string = False
        escape_next = False
        
        for char in text:
            if escape_next:
                fixed_text += char
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                fixed_text += char
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                fixed_text += char
                continue
            
            if not in_string:
                if char in '{[':
                    stack.append(char)
                    fixed_text += char
                elif char in '}]':
                    if stack:
                        if (char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '['):
                            stack.pop()
                            fixed_text += char
                        else:
                            # Fix mismatched brackets
                            if char == '}':
                                fixed_text += ']'
                            else:
                                fixed_text += '}'
                    else:
                        # Add missing opening bracket
                        if char == '}':
                            fixed_text = '{' + fixed_text + char
                        else:
                            fixed_text = '[' + fixed_text + char
                else:
                    fixed_text += char
            else:
                fixed_text += char
        
        # Close any remaining open brackets
        while stack:
            if stack[-1] == '{':
                fixed_text += '}'
            else:
                fixed_text += ']'
            stack.pop()
        
        return fixed_text
    
    def validate_and_fix_data(data):
        """Validate and fix data structure with multiple fallbacks"""
        try:
            # Ensure data is a dictionary
            if not isinstance(data, dict):
                data = {"sections": [], "company_info": {}, "client_info": {}, "terms": {}, "total_amount": 0}
            
            # Ensure required fields exist
            required_fields = {
                "sections": [],
                "company_info": {"name": "", "address": "", "contact": ""},
                "client_info": {"name": "", "project_name": ""},
                "terms": {"payment_terms": "Net 30", "validity": "30 days"},
                "total_amount": 0
            }
            
            for field, default_value in required_fields.items():
                if field not in data:
                    data[field] = default_value
                elif isinstance(default_value, dict) and isinstance(data[field], dict):
                    for subfield, subdefault in default_value.items():
                        if subfield not in data[field]:
                            data[field][subfield] = subdefault
            
            # Validate and fix sections
            if not isinstance(data["sections"], list):
                data["sections"] = []
            
            for section in data["sections"]:
                if not isinstance(section, dict):
                    continue
                
                if "subsections" not in section or not isinstance(section["subsections"], list):
                    section["subsections"] = []
                
                for subsection in section["subsections"]:
                    if not isinstance(subsection, dict):
                        continue
                    
                    if "items" not in subsection or not isinstance(subsection["items"], list):
                        subsection["items"] = []
                    
                    for item in subsection["items"]:
                        if not isinstance(item, dict):
                            continue
                        
                        # Ensure source field exists
                        if "source" not in item or not isinstance(item["source"], dict):
                            item["source"] = {"document": "", "line_reference": ""}
                        
                        # Convert numeric values
                        numeric_fields = ["quantity", "unit_price", "total_price", "rate", "hours"]
                        for field in numeric_fields:
                            if field in item:
                                try:
                                    item[field] = float(item[field])
                                except (ValueError, TypeError):
                                    item[field] = 0
            
            # Convert total_amount to float
            try:
                data["total_amount"] = float(data["total_amount"])
            except (ValueError, TypeError):
                data["total_amount"] = 0
            
            return data
            
        except Exception as e:
            print(f"Error in validate_and_fix_data: {str(e)}")
            # Return a valid default structure
            return {
                "sections": [],
                "company_info": {"name": "", "address": "", "contact": ""},
                "client_info": {"name": "", "project_name": ""},
                "terms": {"payment_terms": "Net 30", "validity": "30 days"},
                "total_amount": 0
            }
    
    # Multiple parsing attempts with different strategies
    parsing_strategies = [
        # Strategy 1: Direct parsing with full cleaning
        lambda: json.loads(fix_json_structure(clean_json_text(response_text))),
        
        # Strategy 2: Extract JSON block and parse
        lambda: json.loads(fix_json_structure(clean_json_text(extract_json_block(response_text) or response_text))),
        
        # Strategy 3: Line-by-line parsing
        lambda: json.loads(fix_json_structure(clean_json_text(''.join([line.strip() for line in response_text.split('\n') if line.strip()])))),
        
        # Strategy 4: Try to fix common JSON issues
        lambda: json.loads(fix_json_structure(clean_json_text(re.sub(r'([^"])\'([^"]*)\'', r'\1"\2"', response_text)))),
        
        # Strategy 5: Last resort - try to extract any valid JSON structure
        lambda: json.loads(fix_json_structure(clean_json_text(re.sub(r'[^{]*(\{[\s\S]*\})[^}]*', r'\1', response_text))))
    ]
    
    # Try each strategy
    for i, strategy in enumerate(parsing_strategies, 1):
        try:
            print(f"\n=== Trying Parsing Strategy {i} ===")
            data = strategy()
            print("Strategy successful!")
            print(json.dumps(data, indent=2))
            
            # Validate and fix the data
            data = validate_and_fix_data(data)
            return data
            
        except Exception as e:
            print(f"Strategy {i} failed: {str(e)}")
            continue
    
    print("\n=== All Parsing Strategies Failed ===")
    # Return a valid default structure as last resort
    return {
        "sections": [],
        "company_info": {"name": "", "address": "", "contact": ""},
        "client_info": {"name": "", "project_name": ""},
        "terms": {"payment_terms": "Net 30", "validity": "30 days"},
        "total_amount": 0
    }

def extract_template_columns(template_file):
    """Extract column structure from Excel template"""
    try:
        # Handle UploadedFile object
        if hasattr(template_file, 'getvalue'):
            # Create a temporary file to read the Excel
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                tmp_file.write(template_file.getvalue())
                tmp_file.flush()
                df = pd.read_excel(tmp_file.name)
                os.unlink(tmp_file.name)
        else:
            # Handle file path
            df = pd.read_excel(template_file)
        return list(df.columns)
    except Exception as e:
        print(f"Error reading template columns: {str(e)}")
        return None

def map_data_to_template_columns(quotation_data, template_columns):
    """Map quotation data to template columns"""
    mapped_data = []
    
    # Helper function to get value from nested dict
    def get_nested_value(data, path):
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return ""
        return str(current)
    
    # Process each section and subsection
    for section in quotation_data.get('sections', []):
        for subsection in section.get('subsections', []):
            for item in subsection.get('items', []):
                row_data = {}
                
                # Map each template column to data
                for col in template_columns:
                    col_lower = col.lower()
                    
                    # Map common fields
                    if 'quantity' in col_lower or 'qty' in col_lower:
                        row_data[col] = item.get('quantity', 0)
                    elif 'description' in col_lower or 'desc' in col_lower:
                        row_data[col] = item.get('description', item.get('task', ''))
                    elif 'product' in col_lower:
                        # Extract product name from description or task
                        description = item.get('description', item.get('task', ''))
                        # Try to extract the main product name (usually the first part before any details)
                        if description:
                            # Split by common separators and take the first meaningful part
                            parts = re.split(r'[,;:()]', description)
                            product_name = parts[0].strip()
                            # Remove common prefixes like "1x", "2x", etc.
                            product_name = re.sub(r'^\d+\s*[xX]\s*', '', product_name)
                            # Remove any remaining numbers at the start
                            product_name = re.sub(r'^\d+\s*', '', product_name)
                            # Capitalize first letter of each word
                            product_name = ' '.join(word.capitalize() for word in product_name.split())
                            row_data[col] = product_name
                        else:
                            row_data[col] = item.get('role', '')  # Fallback to role for labor items
                    elif 'unit price' in col_lower or 'rate' in col_lower:
                        row_data[col] = item.get('unit_price', item.get('rate', 0))
                    elif 'total' in col_lower or 'price' in col_lower:
                        row_data[col] = item.get('total_price', 0)
                    elif 'source' in col_lower:
                        row_data[col] = get_nested_value(item, ['source', 'document'])
                    elif 'reference' in col_lower or 'line' in col_lower:
                        row_data[col] = get_nested_value(item, ['source', 'line_reference'])
                    elif 'role' in col_lower:
                        row_data[col] = item.get('role', '')
                    elif 'task' in col_lower:
                        row_data[col] = item.get('task', '')
                    elif 'hours' in col_lower:
                        row_data[col] = item.get('hours', 0)
                    elif 'section' in col_lower:
                        row_data[col] = section.get('name', '')
                    elif 'subsection' in col_lower:
                        row_data[col] = subsection.get('name', '')
                    else:
                        # Try to find the value in the item
                        row_data[col] = item.get(col, '')
                
                mapped_data.append(row_data)
    
    return mapped_data

def create_excel_from_quotation(quotation_data, template=None):
    """Create an Excel file from quotation data, maintaining template structure if provided"""
    print("\n=== Starting Excel Generation ===")
    print("Input data:", json.dumps(quotation_data, indent=2))
    
    # Validate quotation data structure
    if not quotation_data or not isinstance(quotation_data, dict):
        raise ValueError("Invalid quotation data: must be a non-empty dictionary")
    
    if 'sections' not in quotation_data:
        print("No sections found in data, creating default structure")
        quotation_data['sections'] = []
    
    # Create a new Excel writer
    excel_buffer = pd.ExcelWriter('quotation.xlsx', engine='xlsxwriter')
    workbook = excel_buffer.book
    
    # Add some basic formatting
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#D9E1F2',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'font_size': 12
    })
    subheader_format = workbook.add_format({
        'bold': True,
        'bg_color': '#E2EFDA',
        'border': 1,
        'align': 'left',
        'valign': 'vcenter',
        'font_size': 11
    })
    currency_format = workbook.add_format({
        'num_format': '$#,##0.00',
        'border': 1,
        'align': 'right',
        'valign': 'vcenter',
        'font_size': 11
    })
    text_format = workbook.add_format({
        'border': 1,
        'align': 'left',
        'valign': 'vcenter',
        'text_wrap': True,
        'font_size': 11
    })
    number_format = workbook.add_format({
        'num_format': '#,##0',
        'border': 1,
        'align': 'right',
        'valign': 'vcenter',
        'font_size': 11
    })
    
    # Define default columns
    default_columns = ['Qty', 'Product', 'Description', 'Price Level', 'Unit Price', 'Price', 'Source Document', 'Line Reference']
    
    # If template is provided, use its structure
    if template and hasattr(template, 'name') and template.name.endswith('.xlsx'):
        print("Using custom Excel template structure")
        try:
            # Handle UploadedFile object
            if hasattr(template, 'getvalue'):
                # Create a temporary file to read the Excel
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                    tmp_file.write(template.getvalue())
                    tmp_file.flush()
                    template_columns = extract_template_columns(tmp_file.name)
                    os.unlink(tmp_file.name)
            else:
                template_columns = extract_template_columns(template)
            
            if template_columns:
                # Map data to template columns
                mapped_data = map_data_to_template_columns(quotation_data, template_columns)
                
                # Create DataFrame with template columns
                df = pd.DataFrame(mapped_data, columns=template_columns)
                
                # Write to Excel
                df.to_excel(excel_buffer, sheet_name='Quotation', index=False)
                
                # Get the worksheet
                worksheet = excel_buffer.sheets['Quotation']
                
                # Apply formatting
                for col_num, col_name in enumerate(template_columns):
                    # Set column width
                    worksheet.set_column(col_num, col_num, 20)
                    
                    # Apply header format
                    worksheet.write(0, col_num, col_name, header_format)
                    
                    # Apply appropriate format to data
                    for row_num, value in enumerate(df[col_name], start=1):
                        if 'price' in col_name.lower() or 'total' in col_name.lower() or 'amount' in col_name.lower():
                            try:
                                value = float(value)
                                worksheet.write(row_num, col_num, value, currency_format)
                            except (ValueError, TypeError):
                                worksheet.write(row_num, col_num, value, text_format)
                        elif 'quantity' in col_name.lower() or 'qty' in col_name.lower() or 'hours' in col_name.lower():
                            try:
                                value = float(value)
                                worksheet.write(row_num, col_num, value, number_format)
                            except (ValueError, TypeError):
                                worksheet.write(row_num, col_num, value, text_format)
                        else:
                            worksheet.write(row_num, col_num, value, text_format)
                
                # Add summary information
                summary_row = len(mapped_data) + 2
                worksheet.write(summary_row, 0, 'Total Amount:', header_format)
                worksheet.write(summary_row, 1, f"${quotation_data.get('total_amount', 0):,.2f}", currency_format)
                
                # Add terms if available
                if 'terms' in quotation_data:
                    terms = quotation_data['terms']
                    worksheet.write(summary_row + 1, 0, 'Payment Terms:', header_format)
                    worksheet.write(summary_row + 1, 1, terms.get('payment_terms', 'Net 30'), text_format)
                    worksheet.write(summary_row + 2, 0, 'Validity:', header_format)
                    worksheet.write(summary_row + 2, 1, terms.get('validity', '30 days'), text_format)
        except Exception as e:
            print(f"Error using custom template: {str(e)}")
            print("Falling back to default structure")
            template = None
    
    # Use default structure if no template or template processing failed
    if not template or not hasattr(template, 'name') or not template.name.endswith('.xlsx'):
        print("Using default Excel structure")
        # Create detailed sheets for each section
        print(f"Processing {len(quotation_data['sections'])} sections")
        for section in quotation_data['sections']:
            print(f"\nProcessing section: {section.get('name', 'Unnamed Section')}")
            # Create a list to store all rows for this section
            section_rows = []
            
            # Add section header
            section_rows.append([str(section.get('name', 'Unnamed Section'))] + [''] * (len(default_columns) - 1))
            
            # Process each subsection
            for subsection in section.get('subsections', []):
                print(f"Processing subsection: {subsection.get('name', 'Unnamed Subsection')}")
                # Add subsection header
                section_rows.append([f"  {str(subsection.get('name', 'Unnamed Subsection'))}"] + [''] * (len(default_columns) - 1))
                
                # Add column headers
                section_rows.append(default_columns)
                
                # Add items
                for item in subsection.get('items', []):
                    try:
                        print(f"Processing item: {item.get('description', item.get('role', 'Unnamed Item'))}")
                        if 'description' in item:
                            # Map item data to default columns
                            row_data = {
                                'Qty': str(item.get('quantity', 0)),
                                'Product': str(item.get('product', '')),
                                'Description': str(item.get('description', '')),
                                'Price Level': str(item.get('price_level', 'Standard')),
                                'Unit Price': str(item.get('unit_price', 0)),
                                'Price': str(item.get('total_price', 0)),
                                'Source Document': str(item.get('source', {}).get('document', '')),
                                'Line Reference': str(item.get('source', {}).get('line_reference', ''))
                            }
                        else:
                            # For labor items, map to appropriate columns
                            row_data = {
                                'Qty': str(item.get('hours', 0)),
                                'Product': str(item.get('role', '')),
                                'Description': str(item.get('task', '')),
                                'Price Level': str(item.get('price_level', 'Standard')),
                                'Unit Price': str(item.get('rate', 0)),
                                'Price': str(item.get('total_price', 0)),
                                'Source Document': str(item.get('source', {}).get('document', '')),
                                'Line Reference': str(item.get('source', {}).get('line_reference', ''))
                            }
                        section_rows.append([row_data[col] for col in default_columns])
                    except Exception as e:
                        print(f"Error processing item: {str(e)}")
                        continue
                
                # Add blank row after subsection
                section_rows.append([''] * len(default_columns))
            
            if len(section_rows) <= 3:  # Only headers and no data
                print(f"Warning: Section '{section.get('name', 'Unnamed Section')}' has no valid items")
                continue
            
            # Create DataFrame for this section
            section_df = pd.DataFrame(section_rows)
            
            # Write to Excel
            sheet_name = str(section.get('name', 'Unnamed Section'))[:31]  # Excel sheet names limited to 31 chars
            section_df.to_excel(excel_buffer, sheet_name=sheet_name, index=False, header=False)
            
            # Get the worksheet
            worksheet = excel_buffer.sheets[sheet_name]
            
            # Set column widths based on content type
            column_widths = {
                'Qty': 10,
                'Product': 30,
                'Description': 50,
                'Price Level': 15,
                'Unit Price': 15,
                'Price': 15,
                'Source Document': 25,
                'Line Reference': 15
            }
            
            for i, col in enumerate(default_columns):
                worksheet.set_column(i, i, column_widths[col])
            
            # Apply formatting
            for row in range(len(section_rows)):
                if row == 0:  # Section header
                    worksheet.write(row, 0, section_rows[row][0], header_format)
                elif row == 1:  # Subsection header
                    worksheet.write(row, 0, section_rows[row][0], subheader_format)
                elif row == 2:  # Column headers
                    for col in range(len(default_columns)):
                        worksheet.write(row, col, section_rows[row][col], header_format)
                elif row < len(section_rows) - 1:  # Data rows
                    for col in range(len(default_columns)):
                        if col in [4, 5]:  # Price columns
                            try:
                                value = float(section_rows[row][col])
                                worksheet.write(row, col, value, currency_format)
                            except (ValueError, TypeError):
                                worksheet.write(row, col, section_rows[row][col], text_format)
                        elif col == 0:  # Quantity column
                            try:
                                value = float(section_rows[row][col])
                                worksheet.write(row, col, value, number_format)
                            except (ValueError, TypeError):
                                worksheet.write(row, col, section_rows[row][col], text_format)
                        else:
                            worksheet.write(row, col, section_rows[row][col], text_format)
        
        # Add terms and total
        if 'terms' in quotation_data:
            print("\nAdding terms and total")
            terms = quotation_data['terms']
            if not isinstance(terms, dict):
                terms = {}
            terms_data = [
                ['Terms and Conditions', ''],
                ['Payment Terms', str(terms.get('payment_terms', 'Net 30'))],
                ['Validity Period', str(terms.get('validity', '30 days'))],
                ['', ''],
                ['Total Amount', f"${quotation_data.get('total_amount', 0):,.2f}"]
            ]
            terms_df = pd.DataFrame(terms_data)
            terms_df.to_excel(excel_buffer, sheet_name='Terms', index=False, header=False)
            
            # Format terms sheet
            terms_sheet = excel_buffer.sheets['Terms']
            terms_sheet.set_column('A:A', 25)
            terms_sheet.set_column('B:B', 50)
            
            # Apply formatting to terms sheet
            for row in range(len(terms_data)):
                if row == 0:
                    terms_sheet.write(row, 0, terms_data[row][0], header_format)
                    terms_sheet.write(row, 1, terms_data[row][1], header_format)
                elif row == 4:  # Total Amount row
                    terms_sheet.write(row, 0, terms_data[row][0], header_format)
                    terms_sheet.write(row, 1, terms_data[row][1], currency_format)
                else:
                    terms_sheet.write(row, 0, terms_data[row][0], text_format)
                    terms_sheet.write(row, 1, terms_data[row][1], text_format)
    
    # Save the Excel file
    print("\nSaving Excel file")
    try:
        excel_buffer.close()
    except Exception as e:
        print(f"Error closing Excel buffer: {str(e)}")
        raise
    
    # Verify the file was created and has content
    if not os.path.exists('quotation.xlsx'):
        raise ValueError("Failed to create Excel file")
    
    file_size = os.path.getsize('quotation.xlsx')
    if file_size == 0:
        raise ValueError("Created Excel file is empty")
    
    print(f"Excel file created successfully (size: {file_size} bytes)")
    
    # Verify the file can be read
    try:
        df = pd.read_excel('quotation.xlsx', sheet_name=None)
        if not df:
            raise ValueError("Excel file has no sheets")
        print(f"Successfully verified Excel file with {len(df)} sheets")
    except Exception as e:
        print(f"Error verifying Excel file: {str(e)}")
        raise
    
    return 'quotation.xlsx'

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
                            
                            # Show JSON data immediately in a formatted way
                            with tab2:
                                st.json(quotation_data, expanded=True)
                            
                            # Generate Excel file
                            try:
                                excel_file = create_excel_from_quotation(quotation_data, st.session_state.template)
                                
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
    
    with add_example_col:
        if st.button("Add Example", key="add_example_btn_fab"):
            st.session_state.show_example_overlay = True
            st.rerun()

if __name__ == "__main__":
    main() 