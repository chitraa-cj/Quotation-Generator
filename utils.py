import PyPDF2
from docx import Document
import pandas as pd
import os

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
    """Extract text from PDF file"""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

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