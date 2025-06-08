from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any
import json

# Comment out these imports temporarily to test basic FastAPI functionality
# from components.extract_json import extract_json_from_response
# from components.create_excel import create_excel_from_quotation
# from utils import extract_text_from_file

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chat.google.com",
        "https://chat.googleapis.com", 
        "https://*.google.com",
        "https://*.googleapis.com",
        "https://quotebot-bluelabel.xoidlabs.com",  # Your Streamlit app
        "https://*.xoidlabs.com",  # Allow all xoidlabs.com subdomains
        "http://localhost:3000",  # For local development
        "http://localhost:8501",  # Default Streamlit port
        "http://localhost:8000",  # FastAPI local development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store for temporary files (use Redis/database in production)
temp_files = {}
@app.post("/chat-webhook")
async def chat_webhook(request: Request):
    """Handle Google Chat webhook events"""
    try:
        event = await request.json()
        logger.info(f"Received Google Chat event: {event}")
        
        event_type = event.get("type")
        
        if event_type == "ADDED_TO_SPACE":
            return handle_added_to_space(event)
        elif event_type == "REMOVED_FROM_SPACE":
            return handle_removed_from_space(event)
        elif event_type == "MESSAGE":
            return await handle_message(event)
        else:
            return {"text": "Unknown event type"}
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return {
            "text": "❌ Sorry, there was an error processing your request.",
            "cards": [{
                "sections": [{
                    "widgets": [{
                        "textParagraph": {
                            "text": f"Error: {str(e)}"
                        }
                    }]
                }]
            }]
        }

def handle_added_to_space(event: Dict[Any, Any]) -> Dict[str, Any]:
    """Handle bot being added to space"""
    space = event.get("space", {})
    user = event.get("user", {})
    
    if space.get("singleUserBotDm"):
        message = f"Thank you for adding me to a DM, {user.get('displayName', 'User')}!"
    else:
        space_name = space.get("displayName", "this chat")
        message = f"Thank you for adding me to {space_name}!"
    
    message += "\n\n📄 **How to use me:**\n"
    message += "• Upload documents (PDF, DOCX, TXT, RTF)\n"
    message += "• I'll extract text and generate quotations\n"
    message += "• You'll get an Excel file with the quotation\n"
    message += "• Just attach files and mention me!"
    
    return {"text": message}

def handle_removed_from_space(event: Dict[Any, Any]) -> Dict[str, Any]:
    """Handle bot being removed from space"""
    logger.info(f"Bot removed from space: {event.get('space', {}).get('name', 'unknown')}")
    return {}  # No response needed for removal

async def handle_message(event: Dict[Any, Any]) -> Dict[str, Any]:
    """Handle message events"""
    message = event.get("message", {})
    attachments = message.get("attachment", [])
    text = message.get("text", "").strip()
    user = event.get("user", {})
    
    # If no attachments, provide instructions
    if not attachments:
        return {
            "text": "👋 Hello! To generate a quotation, please attach your documents to this message.\n\n"
                   "📎 **Supported formats:** PDF, DOCX, DOC, TXT, RTF\n"
                   "📊 I'll analyze your documents and create an Excel quotation file."
        }
    
    try:
        # Process attachments
        extracted_texts = []
        processed_files = []
        
        for attachment in attachments:
            attachment_name = attachment.get("name", "unknown_file")
            drive_file_id = attachment.get("driveFile", {}).get("id")
            
            if not drive_file_id:
                continue
                
            # Download file from Google Drive
            file_content = await download_drive_file(drive_file_id, event)
            if not file_content:
                continue
            
            # Save temporarily and extract text
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(attachment_name).suffix) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                if is_valid_file_type(attachment_name):
                    text_content = extract_text_from_file(temp_file_path, attachment_name)
                    if text_content.strip():
                        extracted_texts.append(text_content)
                        processed_files.append(attachment_name)
            except Exception as e:
                logger.error(f"Error processing {attachment_name}: {str(e)}")
            finally:
                os.unlink(temp_file_path)
        
        if not extracted_texts:
            return {
                "text": "❌ No valid text could be extracted from the attached files.\n"
                       "Please ensure files are not corrupted and are in supported formats."
            }
        
        # Generate quotation
        combined_text = "\n".join(extracted_texts)
        quotation_data = extract_json_from_response(combined_text)
        excel_file_path = create_excel_from_quotation(quotation_data)
        
        # Store file temporarily
        file_id = str(uuid.uuid4())
        temp_files[file_id] = excel_file_path
        
        # Create response with download link
        download_url = f"https://your-api-domain.com/download/{file_id}"  # Replace with your actual domain
        
        return {
            "text": f"✅ **Quotation Generated Successfully!**\n\n"
                   f"📁 Processed files: {', '.join(processed_files)}\n"
                   f"📊 Download your Excel quotation: [Click here]({download_url})\n\n"
                   f"⏰ Download link expires in 1 hour.",
            "cards": [{
                "sections": [{
                    "widgets": [{
                        "buttons": [{
                            "textButton": {
                                "text": "📥 DOWNLOAD QUOTATION",
                                "onClick": {
                                    "openLink": {
                                        "url": download_url
                                    }
                                }
                            }
                        }]
                    }]
                }]
            }]
        }
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return {
            "text": f"❌ Error generating quotation: {str(e)}\n"
                   "Please try again or contact support."
        }

async def download_drive_file(file_id: str, event: Dict[Any, Any]) -> bytes:
    """Download file from Google Drive using the bot's token"""
    try:
        # You'll need to implement OAuth2 token handling for your bot
        # This is a simplified version - you'll need proper token management
        
        # For now, return None - you'll need to implement Drive API access
        # with proper authentication for your Google Chat bot
        logger.warning("Drive file download not implemented - requires OAuth2 setup")
        return None
        
    except Exception as e:
        logger.error(f"Error downloading Drive file {file_id}: {str(e)}")
        return None

@app.post("/process-documents")
async def process_documents(files: List[UploadFile] = File(...)):
    """Main endpoint for direct file upload and processing"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Extract text from all uploaded files
        extracted_texts = []
        
        for file in files:
            # Validate file type
            if not is_valid_file_type(file.filename):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file.filename}"
                )
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Extract text from file
                text = extract_text_from_file(temp_file_path, file.filename)
                extracted_texts.append(text)
            finally:
                # Clean up temp file
                os.unlink(temp_file_path)
        
        # Combine all extracted text
        input_docs_text = "\n".join(extracted_texts)
        
        if not input_docs_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the files")
        
        # Generate quotation JSON
        quotation_data = extract_json_from_response(input_docs_text)
        
        # Create Excel file
        excel_file_path = create_excel_from_quotation(quotation_data)
        
        # Store file temporarily with unique ID
        file_id = str(uuid.uuid4())
        temp_files[file_id] = excel_file_path
        
        return {
            "message": "Quotation generated successfully!",
            "download_url": f"/download/{file_id}",
            "file_id": file_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """Download generated Excel file"""
    if file_id not in temp_files:
        raise HTTPException(status_code=404, detail="File not found or expired")
    
    file_path = temp_files[file_id]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Clean up after download
    def cleanup():
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
            temp_files.pop(file_id, None)
        except Exception as e:
            logger.error(f"Error cleaning up file: {str(e)}")
    
    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="quotation.xlsx",
        background=cleanup
    )

def is_valid_file_type(filename: str) -> bool:
    """Validate file types"""
    if not filename:
        return False
    
    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.rtf'}
    file_ext = Path(filename).suffix.lower()
    return file_ext in allowed_extensions

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.get("/")
async def root():
    return {
        "message": "Document Quotation API with Google Chat Integration",
        "endpoints": {
            "upload": "/process-documents",
            "download": "/download/{file_id}",
            "webhook": "/chat-webhook"
        }
    }