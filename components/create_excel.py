import json5
import pandas as pd
import tempfile
import os
import re
import logging

# Configure logging
logger = logging.getLogger(__name__)

def sanitize_sheet_name(name: str) -> str:
    """Sanitize sheet name to comply with Excel restrictions"""
    if not name or not isinstance(name, str):
        logger.warning("Invalid sheet name provided, using default name")
        return "Unnamed Section"

    # Remove invalid characters: []:*?/\
    sanitized = re.sub(r'[\[\]:*?/\\]', '', name)

    # Ensure the name is not "History" (case-insensitive)
    if sanitized.lower() == "history":
        sanitized = "HistoryRenamed"

    # Truncate to 31 characters (Excel limit)
    if len(sanitized) > 31:
        logger.warning(f"Sheet name '{sanitized}' exceeds 31 characters, truncating")
        sanitized = sanitized[:31].rstrip()

    # Ensure the name is not empty after sanitization
    if not sanitized.strip():
        logger.warning("Sheet name is empty after sanitization, using default name")
        return "Unnamed Section"

    return sanitized

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
        logger.error(f"Error reading template columns: {str(e)}")
        return None
    
def attempt_fix_json(raw_text):
    """Attempts to fix common JSON format errors"""
    # Fix unquoted keys
    raw_text = re.sub(r'([{,])(\s*)([a-zA-Z0-9_]+)(\s*):', r'\1"\3":', raw_text)
    
    # Remove trailing commas before closing braces/brackets
    raw_text = re.sub(r',(\s*[}\]])', r'\1', raw_text)
    
    # Fix unterminated strings
    raw_text = re.sub(r'([^\\])"([^"]*?)(?=\s*[,}\]])', r'\1"\2"', raw_text)
    
    # Attempt to auto-close brackets
    open_brackets = raw_text.count('[')
    close_brackets = raw_text.count(']')
    if open_brackets > close_brackets:
        raw_text += ']' * (open_brackets - close_brackets)
    
    open_braces = raw_text.count('{')
    close_braces = raw_text.count('}')
    if open_braces > close_braces:
        raw_text += '}' * (open_braces - close_braces)
    
    return raw_text

def create_excel_from_quotation(quotation_data, template=None):
    """Create an Excel file from quotation data with robust JSON handling"""
    logger.info("\n=== Starting Excel Generation ===")
    
    # Convert quotation_data to JSON string if it's a dict
    if isinstance(quotation_data, dict):
        try:
            json_str = json5.dumps(quotation_data)
        except Exception as e:
            logger.error(f"Error converting dict to JSON: {str(e)}")
            json_str = str(quotation_data)
    else:
        json_str = str(quotation_data)
    
    # Attempt to fix JSON if needed
    try:
        data = json5.loads(json_str)
    except json5.Json5DecodeError as e:
        logger.warning(f"Initial JSON5 parse failed: {e}")
        fixed_json = attempt_fix_json(json_str)
        try:
            data = json5.loads(fixed_json)
            logger.info("Recovered from malformed JSON")
        except Exception as e2:
            logger.error(f"Failed to fix JSON: {e2}")
            raise ValueError("Invalid JSON data structure")
    
    # Ensure data has required structure
    if not isinstance(data, dict):
        raise ValueError("Quotation data must be a dictionary")
    
    # Create default structure if missing
    data.setdefault("sections", [])
    data.setdefault("company_info", {"name": "", "address": "", "contact": ""})
    data.setdefault("client_info", {"name": "", "project_name": ""})
    data.setdefault("terms", {"payment_terms": "Net 30", "validity": "30 days"})
    data.setdefault("total_amount", 0)
    
    # Create Excel writer
    excel_buffer = pd.ExcelWriter('quotation.xlsx', engine='xlsxwriter')
    workbook = excel_buffer.book
    
    # Add formatting
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#D9E1F2',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'font_size': 12
    })
    
    # Define column names
    column_names = ['Qty', 'Product', 'Description', 'Price Level', 'Unit Price', 'Price', 'Source', 'Reference']
    
    # Process sections
    for section in data.get('sections', []):
        section_name = section.get('name', 'Unnamed Section')
        # Sanitize the sheet name
        sanitized_sheet_name = sanitize_sheet_name(section_name)
        logger.debug(f"Original sheet name: '{section_name}', Sanitized sheet name: '{sanitized_sheet_name}'")
        rows = []
        
        # Add section header
        rows.append([section_name] + [''] * (len(column_names) - 1))
        
        # Process subsections
        for subsection in section.get('subsections', []):
            rows.append([f"  {subsection.get('name', 'Unnamed Subsection')}"] + [''] * (len(column_names) - 1))
            rows.append(column_names)  # Add column headers
            
            # Process items
            for item in subsection.get('items', []):
                # Convert all text fields to strings to prevent Arrow serialization issues
                row = [
                    item.get('quantity', 0),
                    str(item.get('product', '')),
                    str(item.get('description', '')),
                    str(item.get('price_level', 'Standard')),
                    item.get('unit_price', 0),
                    item.get('total_price', 0),
                    str(item.get('source', {}).get('document', '')),
                    str(item.get('source', {}).get('line_reference', ''))
                ]
                rows.append(row)
        
        if rows:
            # Create DataFrame with proper column names
            df = pd.DataFrame(rows, columns=column_names)
            
            # Convert all object columns to string type to prevent Arrow serialization issues
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
            
            # Write to Excel using sanitized sheet name
            df.to_excel(excel_buffer, sheet_name=sanitized_sheet_name, index=False)
            
            # Get worksheet and apply formatting
            worksheet = excel_buffer.sheets[sanitized_sheet_name]
            
            # Apply header format to the first row
            for col_num, value in enumerate(df.columns):
                worksheet.write(0, col_num, value, header_format)
            
            # Set column widths
            for i, col in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']):
                worksheet.set_column(f'{col}:{col}', 20)
    
    # Add summary sheet
    summary_data = [
        ['Company Information', ''],
        ['Name', data['company_info']['name']],
        ['Address', data['company_info']['address']],
        ['Contact', data['company_info']['contact']],
        ['', ''],
        ['Client Information', ''],
        ['Name', data['client_info']['name']],
        ['Project', data['client_info']['project_name']],
        ['', ''],
        ['Terms', ''],
        ['Payment Terms', data['terms']['payment_terms']],
        ['Validity', data['terms']['validity']],
        ['', ''],
        ['Total Amount', f"${data['total_amount']:,.2f}"]
    ]
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(excel_buffer, sheet_name='Summary', index=False, header=False)
    
    # Format summary sheet
    summary_sheet = excel_buffer.sheets['Summary']
    summary_sheet.set_column('A:A', 20)
    summary_sheet.set_column('B:B', 40)
    
    # Save and verify
    try:
        excel_buffer.close()
        if not os.path.exists('quotation.xlsx'):
            raise ValueError("Failed to create Excel file")
        
        # Verify file can be read
        pd.read_excel('quotation.xlsx', sheet_name=None)
        return 'quotation.xlsx'
        
    except Exception as e:
        logger.error(f"Error saving Excel file: {str(e)}")
        raise