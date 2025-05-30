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
    if template:
        placeholders = extract_template_placeholders(template)
        template_analysis = f"\nTemplate Analysis:\nThe provided template contains the following placeholders: {', '.join(placeholders)}\n"
        template_analysis += "Ensure all these placeholders are properly filled with relevant information from the input documents.\n"
    
    # Chain of thought reasoning prompt
    chain_of_thought = f"""
Follow these steps carefully:

1. Template Analysis:
   - Identify all placeholders in the template
   - Understand the structure and formatting requirements
   - Note any specific sections or categories needed
   {template_analysis}

2. Content Extraction:
   - Extract relevant information from input documents
   - Match extracted information to template placeholders
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
   - Verify that all placeholders are filled
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
                },
                {
                    "name": "Accessories",
                    "items": [...]
                },
                {
                    "name": "Cables and Connectors",
                    "items": [...]
                },
                {
                    "name": "Mounts and Supports",
                    "items": [...]
                },
                {
                    "name": "Cases and Protection",
                    "items": [...]
                },
                {
                    "name": "Spare Parts",
                    "items": [...]
                },
                {
                    "name": "Testing Equipment",
                    "items": [...]
                }
            ]
        },
        {
            "name": "Labor",
            "subsections": [
                {
                    "name": "Technical Staff",
                    "items": [
                        {
                            "role": "string",
                            "task": "string",
                            "time_block": "string",
                            "hours": number,
                            "rate": number,
                            "total_price": number,
                            "source": {
                                "document": "string",
                                "line_reference": "string"
                            }
                        }
                    ]
                },
                {
                    "name": "Support Staff",
                    "items": [...]
                },
                {
                    "name": "Setup Labor",
                    "items": [...]
                },
                {
                    "name": "Operation Labor",
                    "items": [...]
                },
                {
                    "name": "Teardown Labor",
                    "items": [...]
                },
                {
                    "name": "Prep Time",
                    "items": [...]
                },
                {
                    "name": "Travel Time",
                    "items": [...]
                },
                {
                    "name": "Buffer Time",
                    "items": [...]
                },
                {
                    "name": "Per Diem and Meals",
                    "items": [...]
                }
            ]
        },
        {
            "name": "Custom Fabrication",
            "subsections": [
                {
                    "name": "Materials",
                    "items": [...]
                },
                {
                    "name": "Labor",
                    "items": [...]
                },
                {
                    "name": "Finishing",
                    "items": [...]
                },
                {
                    "name": "Hardware",
                    "items": [...]
                },
                {
                    "name": "Protective Coatings",
                    "items": [...]
                }
            ]
        },
        {
            "name": "Creative Services",
            "subsections": [
                {
                    "name": "Design",
                    "items": [...]
                },
                {
                    "name": "Content Creation",
                    "items": [...]
                },
                {
                    "name": "Technical Direction",
                    "items": [...]
                },
                {
                    "name": "Rehearsal Time",
                    "items": [...]
                },
                {
                    "name": "Quality Control",
                    "items": [...]
                }
            ]
        },
        {
            "name": "Supporting Services",
            "subsections": [
                {
                    "name": "Freight and Shipping",
                    "items": [...]
                },
                {
                    "name": "Equipment Prep",
                    "items": [...]
                },
                {
                    "name": "Testing and Quality Control",
                    "items": [...]
                },
                {
                    "name": "Consumables",
                    "items": [...]
                },
                {
                    "name": "Warehouse Services",
                    "items": [...]
                },
                {
                    "name": "Local Services",
                    "items": [...]
                },
                {
                    "name": "Insurance and Permits",
                    "items": [...]
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
    """Extract and parse JSON from the AI response"""
    try:
        # First try direct JSON parsing
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            # Try to find JSON block in the response
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        
        # If JSON parsing fails, try to structure the response
        try:
            # Split the response into sections
            sections = response_text.split('\n\n')
            structured_data = {
                "company_info": {
                    "name": "",
                    "address": "",
                    "contact": ""
                },
                "client_info": {
                    "name": "",
                    "project_name": ""
                },
                "sections": [],
                "terms": {
                    "payment_terms": "Net 30",
                    "validity": "30 days"
                },
                "total_amount": 0
            }
            
            current_section = None
            current_subsection = None
            
            for section in sections:
                if section.strip():
                    # Check if this is a main section
                    if section.upper() in ["AV EQUIPMENT", "LABOR", "CUSTOM FABRICATION", "CREATIVE SERVICES", "SUPPORTING SERVICES"]:
                        current_section = {
                            "name": section.strip(),
                            "subsections": []
                        }
                        structured_data["sections"].append(current_section)
                    # Check if this is a subsection
                    elif current_section and section.strip().startswith("  "):
                        current_subsection = {
                            "name": section.strip(),
                            "items": []
                        }
                        current_section["subsections"].append(current_subsection)
                    # This should be an item
                    elif current_subsection:
                        # Try to parse the item
                        try:
                            if ":" in section:
                                parts = section.split(":")
                                description = parts[0].strip()
                                details = parts[1].strip()
                                
                                # Try to extract quantities and prices
                                quantity_match = re.search(r'(\d+)\s*x\s*\$(\d+\.?\d*)', details)
                                if quantity_match:
                                    quantity = int(quantity_match.group(1))
                                    unit_price = float(quantity_match.group(2))
                                    total_price = quantity * unit_price
                                    
                                    item = {
                                        "description": description,
                                        "quantity": quantity,
                                        "unit_price": unit_price,
                                        "total_price": total_price,
                                        "source": {
                                            "document": "Input Document",
                                            "line_reference": "1"
                                        }
                                    }
                                    current_subsection["items"].append(item)
                                    structured_data["total_amount"] += total_price
                        except Exception:
                            continue
            
            return structured_data
        except Exception:
            return None

def create_excel_from_quotation(quotation_data, template=None):
    """Create an Excel file from quotation data, maintaining template structure if provided"""
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
    
    # If template is provided, create a summary sheet with template structure
    if template:
        # Extract template placeholders
        placeholders = extract_template_placeholders(template)
        
        # Create summary sheet
        summary_data = []
        
        # Add company information
        if 'company_info' in quotation_data:
            summary_data.extend([
                ['Company Information', ''],
                ['Company Name', quotation_data['company_info'].get('name', '')],
                ['Address', quotation_data['company_info'].get('address', '')],
                ['Contact', quotation_data['company_info'].get('contact', '')],
                ['', '']
            ])
        
        # Add quotation details
        summary_data.extend([
            ['Quotation Details', ''],
            ['Date', datetime.now().strftime('%Y-%m-%d')],
            ['Quotation Number', f"Q{datetime.now().strftime('%Y%m%d%H%M')}"],
            ['', '']
        ])
        
        # Add client information
        if 'client_info' in quotation_data:
            summary_data.extend([
                ['Client Information', ''],
                ['Client Name', quotation_data['client_info'].get('name', '')],
                ['Project Name', quotation_data['client_info'].get('project_name', '')],
                ['', '']
            ])
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(excel_buffer, sheet_name='Summary', index=False, header=False)
        
        # Format summary sheet
        summary_sheet = excel_buffer.sheets['Summary']
        summary_sheet.set_column('A:A', 25)
        summary_sheet.set_column('B:B', 50)
        
        # Apply formatting to summary sheet
        for row in range(len(summary_data)):
            if row == 0 or summary_data[row][0] in ['Company Information', 'Quotation Details', 'Client Information']:
                summary_sheet.write(row, 0, summary_data[row][0], header_format)
                summary_sheet.write(row, 1, summary_data[row][1], header_format)
            else:
                summary_sheet.write(row, 0, summary_data[row][0], text_format)
                summary_sheet.write(row, 1, summary_data[row][1], text_format)
    
    # Create detailed sheets for each section
    for section in quotation_data['sections']:
        # Create a list to store all rows for this section
        section_rows = []
        
        # Add section header
        section_rows.append([section['name']] + [''] * (len(default_columns) - 1))
        
        # Process each subsection
        for subsection in section['subsections']:
            # Add subsection header
            section_rows.append([f"  {subsection['name']}"] + [''] * (len(default_columns) - 1))
            
            # Add column headers
            section_rows.append(default_columns)
            
            # Add items
            for item in subsection['items']:
                if 'description' in item:
                    # Map item data to default columns
                    row_data = {
                        'Qty': item['quantity'],
                        'Product': item.get('product', ''),
                        'Description': item['description'],
                        'Price Level': item.get('price_level', 'Standard'),
                        'Unit Price': item['unit_price'],
                        'Price': item['total_price'],
                        'Source Document': item['source']['document'],
                        'Line Reference': item['source']['line_reference']
                    }
                    section_rows.append([row_data[col] for col in default_columns])
                else:
                    # For labor items, map to appropriate columns
                    row_data = {
                        'Qty': item['hours'],
                        'Product': item['role'],
                        'Description': item['task'],
                        'Price Level': item.get('price_level', 'Standard'),
                        'Unit Price': item['rate'],
                        'Price': item['total_price'],
                        'Source Document': item['source']['document'],
                        'Line Reference': item['source']['line_reference']
                    }
                    section_rows.append([row_data[col] for col in default_columns])
            
            # Add blank row after subsection
            section_rows.append([''] * len(default_columns))
        
        # Create DataFrame for this section
        section_df = pd.DataFrame(section_rows)
        
        # Write to Excel
        sheet_name = section['name'][:31]  # Excel sheet names limited to 31 chars
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
                        worksheet.write(row, col, section_rows[row][col], currency_format)
                    elif col == 0:  # Quantity column
                        worksheet.write(row, col, section_rows[row][col], number_format)
                    else:
                        worksheet.write(row, col, section_rows[row][col], text_format)
    
    # Add terms and total
    if 'terms' in quotation_data:
        terms_data = [
            ['Terms and Conditions', ''],
            ['Payment Terms', quotation_data['terms'].get('payment_terms', 'Net 30')],
            ['Validity Period', quotation_data['terms'].get('validity', '30 days')],
            ['', ''],
            ['Total Amount', f"${quotation_data['total_amount']:,.2f}"]
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
    excel_buffer.close()
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
    st.title("Quotation Generator")
    
    # Template settings in sidebar
    st.sidebar.header("Template Settings")
    template_mode = st.sidebar.radio(
        "Select Template Mode",
        ["Default Template", "Custom Template"]
    )
    
    if template_mode == "Custom Template":
        template_file = st.sidebar.file_uploader(
            "Upload Template File",
            type=['txt', 'docx'],
            key="template_uploader"
        )
        if template_file:
            template_text = extract_text_from_file(template_file, template_file.name)
            st.session_state.template = template_text
            st.sidebar.text_area(
                "Template Preview",
                template_text,
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
                    # Extract and parse the response
                    quotation_data = extract_json_from_response(response.text)
                    
                    # Print raw response to terminal
                    print("\n=== Raw AI Response ===")
                    print(response.text)
                    print("\n=== Structured Quotation Data ===")
                    print(json.dumps(quotation_data, indent=2))
                    print("==============================\n")
                    
                    if quotation_data:
                        # Validate line count
                        if not validate_quotation_line_count(quotation_data):
                            # st.warning("Generated quotation has fewer than 150 line items. Regenerating with more detail...")
                            # Regenerate with emphasis on more detail
                            response = model.generate_content(prompt + "\n\nIMPORTANT: The previous response did not have enough line items. Please generate a more detailed quotation with at least 150 line items.")
                            quotation_data = extract_json_from_response(response.text)
                            
                            # Print regenerated response to terminal
                            print("\n=== Regenerated Raw AI Response ===")
                            print(response.text)
                            print("\n=== Regenerated Structured Quotation Data ===")
                            print(json.dumps(quotation_data, indent=2))
                            print("=======================================\n")
                        
                        if quotation_data:
                            # Create Excel file with template structure
                            excel_file = create_excel_from_quotation(quotation_data, st.session_state.template)
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                'role': 'user',
                                'content': f"Generated quotation for {len(uploaded_files)} files"
                            })
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': "Here's the generated quotation:"
                            })
                            
                            # Create tabs for different views
                            tab1, tab2, tab3 = st.tabs(["Excel Preview", "JSON Data", "Download"])
                            
                            with tab1:
                                # Read the Excel file and display preview
                                df = pd.read_excel(excel_file, sheet_name=None)
                                for sheet_name, sheet_df in df.items():
                                    st.subheader(sheet_name)
                                    st.dataframe(sheet_df, use_container_width=True)
                            
                            with tab2:
                                # Display formatted JSON
                                st.json(quotation_data)
                            
                            with tab3:
                                # Add Excel download button
                                with open(excel_file, 'rb') as f:
                                    st.download_button(
                                        label="Download Quotation (Excel)",
                                        data=f,
                                        file_name=f"quotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                        else:
                            st.error("Error structuring the AI response. Please try again.")
                            st.text_area("Raw AI Response", response.text, height=300)
                    else:
                        st.error("Error structuring the AI response. Please try again.")
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