import json
import pandas as pd
import tempfile
import os
import re


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
        print(f"Error reading template columns: {str(e)}")
        return None
    



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
                
                # Create DataFrame with template columns and ensure proper types
                df = pd.DataFrame(mapped_data, columns=template_columns)
                
                # Convert numeric columns to appropriate types
                for col in df.columns:
                    col_lower = col.lower()
                    if 'price' in col_lower or 'total' in col_lower or 'amount' in col_lower:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    elif 'quantity' in col_lower or 'qty' in col_lower or 'hours' in col_lower:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    else:
                        df[col] = df[col].astype(str)
                
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
                        col_lower = col_name.lower()
                        if 'price' in col_lower or 'total' in col_lower or 'amount' in col_lower:
                            try:
                                value = float(value)
                                worksheet.write(row_num, col_num, value, currency_format)
                            except (ValueError, TypeError):
                                worksheet.write(row_num, col_num, str(value), text_format)
                        elif 'quantity' in col_lower or 'qty' in col_lower or 'hours' in col_lower:
                            try:
                                value = float(value)
                                worksheet.write(row_num, col_num, value, number_format)
                            except (ValueError, TypeError):
                                worksheet.write(row_num, col_num, str(value), text_format)
                        else:
                            worksheet.write(row_num, col_num, str(value), text_format)
                
                # Add summary information
                summary_row = len(mapped_data) + 2
                worksheet.write(summary_row, 0, 'Total Amount:', header_format)
                worksheet.write(summary_row, 1, f"${quotation_data.get('total_amount', 0):,.2f}", currency_format)
                
                # Add terms if available
                if 'terms' in quotation_data:
                    terms = quotation_data['terms']
                    worksheet.write(summary_row + 1, 0, 'Payment Terms:', header_format)
                    worksheet.write(summary_row + 1, 1, str(terms.get('payment_terms', 'Net 30')), text_format)
                    worksheet.write(summary_row + 2, 0, 'Validity:', header_format)
                    worksheet.write(summary_row + 2, 1, str(terms.get('validity', '30 days')), text_format)
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
                            # Map item data to default columns with proper type conversion
                            row_data = {
                                'Qty': float(item.get('quantity', 0)),
                                'Product': str(item.get('product', '')),
                                'Description': str(item.get('description', '')),
                                'Price Level': str(item.get('price_level', 'Standard')),
                                'Unit Price': float(item.get('unit_price', 0)),
                                'Price': float(item.get('total_price', 0)),
                                'Source Document': str(item.get('source', {}).get('document', '')),
                                'Line Reference': str(item.get('source', {}).get('line_reference', ''))
                            }
                        else:
                            # For labor items, map to appropriate columns with proper type conversion
                            row_data = {
                                'Qty': float(item.get('hours', 0)),
                                'Product': str(item.get('role', '')),
                                'Description': str(item.get('task', '')),
                                'Price Level': str(item.get('price_level', 'Standard')),
                                'Unit Price': float(item.get('rate', 0)),
                                'Price': float(item.get('total_price', 0)),
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
            
            # Create DataFrame for this section with proper type conversion
            section_df = pd.DataFrame(section_rows)
            
            # Convert numeric columns to appropriate types
            for col in section_df.columns:
                col_lower = col.lower()
                if 'price' in col_lower or 'total' in col_lower or 'amount' in col_lower:
                    section_df[col] = pd.to_numeric(section_df[col], errors='coerce').fillna(0)
                elif 'quantity' in col_lower or 'qty' in col_lower or 'hours' in col_lower:
                    section_df[col] = pd.to_numeric(section_df[col], errors='coerce').fillna(0)
                else:
                    section_df[col] = section_df[col].astype(str)
            
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
                    worksheet.write(row, 0, str(section_rows[row][0]), header_format)
                elif row == 1:  # Subsection header
                    worksheet.write(row, 0, str(section_rows[row][0]), subheader_format)
                elif row == 2:  # Column headers
                    for col in range(len(default_columns)):
                        worksheet.write(row, col, str(section_rows[row][col]), header_format)
                elif row < len(section_rows) - 1:  # Data rows
                    for col in range(len(default_columns)):
                        col_name = default_columns[col].lower()
                        if 'price' in col_name or 'total' in col_name or 'amount' in col_name:
                            try:
                                value = float(section_rows[row][col])
                                worksheet.write(row, col, value, currency_format)
                            except (ValueError, TypeError):
                                worksheet.write(row, col, str(section_rows[row][col]), text_format)
                        elif 'quantity' in col_name or 'qty' in col_name or 'hours' in col_name:
                            try:
                                value = float(section_rows[row][col])
                                worksheet.write(row, col, value, number_format)
                            except (ValueError, TypeError):
                                worksheet.write(row, col, str(section_rows[row][col]), text_format)
                        else:
                            worksheet.write(row, col, str(section_rows[row][col]), text_format)
        
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
                    terms_sheet.write(row, 0, str(terms_data[row][0]), header_format)
                    terms_sheet.write(row, 1, str(terms_data[row][1]), header_format)
                elif row == 4:  # Total Amount row
                    terms_sheet.write(row, 0, str(terms_data[row][0]), header_format)
                    terms_sheet.write(row, 1, str(terms_data[row][1]), currency_format)
                else:
                    terms_sheet.write(row, 0, str(terms_data[row][0]), text_format)
                    terms_sheet.write(row, 1, str(terms_data[row][1]), text_format)
    
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