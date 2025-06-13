import PyPDF2
from docx import Document
import pandas as pd
import os
import re
import logging

logger = logging.getLogger(__name__)

def extract_text_from_file(file_path, file_name):
    """
    Extract text from various file types (PDF, DOCX, XLSX, TXT)
    """
    file_extension = os.path.splitext(file_name)[1].lower()
    
    try:
        if file_extension == '.pdf':
            return extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return extract_text_from_docx(file_path)
        elif file_extension == '.xlsx':
            return extract_text_from_excel(file_path)
        elif file_extension == '.txt':
            return extract_text_from_txt(file_path)
        else:
            return f"Unsupported file type: {file_extension}"
    except Exception as e:
        return f"Error processing file: {str(e)}"

def extract_text_from_pdf(file_path):
    """Extract text from PDF file with improved cleaning"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Check if PDF is encrypted
            if pdf_reader.is_encrypted:
                return "Error: PDF is encrypted/password protected"
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    # Clean up the text
                    page_text = page_text.replace('\r', ' ')  # Replace carriage returns
                    page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace
                    page_text = page_text.strip()
                    
                    # Fix common OCR issues
                    page_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', page_text)  # Add space between camelCase
                    page_text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', page_text)  # Add space between number and letter
                    page_text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', page_text)  # Add space between letter and number
                    
                    # Fix common word breaks
                    word_fixes = {
                        'M eeting': 'Meeting',
                        'Tom orrow': 'Tomorrow',
                        'Q uote': 'Quote',
                        'M ay': 'May',
                        'W e': 'We',
                        'w ould': 'would',
                        'subm it': 'submit',
                        'th.': 'th',
                        'Univ ersity': 'University',
                        'Manuf actur ers': 'Manufacturers',
                        'Stand Alone': 'Stand-Alone',
                    }
                    for wrong, correct in word_fixes.items():
                        page_text = page_text.replace(wrong, correct)
                    
                    # Fix date formatting
                    page_text = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', r'\1/\2/\3', page_text)
                    page_text = re.sub(r'(\d{1,2})(?:st|nd|rd|th)?,?\s+([A-Za-z]+),?\s+(\d{4})', r'\1 \2 \3', page_text)
                    
                    # Fix bullet points and lists
                    page_text = re.sub(r'[•●◆■]', '•', page_text)
                    page_text = re.sub(r'(\d+)\)', r'\1.', page_text)
                    
                    # Fix quotes and special characters
                    page_text = re.sub(r'["""]', '"', page_text)
                    page_text = re.sub(r"[''']", "'", page_text)
                    page_text = re.sub(r'[–—]', '-', page_text)
                    
                    text += page_text + "\n"
            
            # Final cleanup
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace again
            text = text.strip()
            
            return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_excel(file_path):
    """Extract text from Excel file"""
    df = pd.read_excel(file_path)
    return df.to_string()

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def process_document(file_path, file_name):
    """
    Process document and return extracted text
    """
    return extract_text_from_file(file_path, file_name)

def clean_text(text: str) -> str:
    """Clean and normalize text content with improved number handling"""
    if not text:
        return ""
    
    try:
        # Replace special characters and normalize spaces
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        text = text.replace('\n', ' ')
        
        # Fix common OCR issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # camelCase
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # number+letter
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # letter+number
        text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)  # punctuation+letter
        
        # Fix common OCR mistakes
        ocr_fixes = {
            'l': '1',  # lowercase L to 1
            'O': '0',  # uppercase O to 0
            '|': '1',  # vertical bar to 1
            'I': '1',  # uppercase I to 1
            'i': '1',  # lowercase i to 1
        }
        for wrong, correct in ocr_fixes.items():
            text = re.sub(f'\\b{wrong}\\b', correct, text)
        
        # Fix number and unit spacing with improved handling
        text = re.sub(r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*([A-Za-z]+)', r'\1 \2', text)  # Add space between number and unit
        text = re.sub(r'([A-Za-z]+)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', r'\1 \2', text)  # Add space between unit and number
        
        # Fix common unit abbreviations with proper handling
        unit_fixes = {
            r'\bB\b': 'Billion',
            r'\bM\b': 'Million',
            r'\bK\b': 'Thousand',
            r'\bhrs\b': 'hours',
            r'\bhr\b': 'hour',
            r'\bmin\b': 'minutes',
            r'\bsec\b': 'seconds',
        }
        for unit, full in unit_fixes.items():
            text = re.sub(unit, full, text)
        
        # Fix price formatting
        text = re.sub(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', r'$\1', text)  # Preserve existing decimal places
        text = re.sub(r'\$(\d+)(?!\.)', r'$\1.00', text)  # Add cents to whole dollar amounts
        text = re.sub(r'\$(\d+)\.(\d)$', r'$\1.\20', text)  # Add trailing zero to single decimal
        text = re.sub(r'\$(\d+)\.(\d{3,})', r'$\1.\2', text)  # Fix more than 2 decimal places
        
        # Fix date formatting
        text = re.sub(r'(\d{1,2})/(\d{1,2})-(\d{1,2})/(\d{1,2})', r'\1/\2 to \3/\4', text)  # Fix date ranges
        text = re.sub(r'(\d{1,2})/(\d{1,2})', r'\1/\2', text)  # Ensure proper date format
        
        # Fix bullet points and lists
        text = re.sub(r'[•●◆■]', '•', text)  # Normalize bullet points
        text = re.sub(r'(\d+)\)', r'\1.', text)  # Convert numbered lists to consistent format
        
        # Fix common word spacing issues
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'(\w)\s+(\w)', r'\1 \2', text)  # Fix double spaces between words
        
        # Fix common OCR word breaks
        word_fixes = {
            'Univ ersity': 'University',
            'Manuf actur ers': 'Manufacturers',
            'Stand Alone': 'Stand-Alone',
        }
        for wrong, correct in word_fixes.items():
            text = text.replace(wrong, correct)
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return text.strip() 