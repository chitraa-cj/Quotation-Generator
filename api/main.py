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
import httpx
import base64

# Comment out these imports temporarily to test basic FastAPI functionality
from components.extract_json import extract_json_from_response
from components.create_excel import create_excel_from_quotation
from utils import extract_text_from_file

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporarily allow all origins for debugging
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more details
logger = logging.getLogger(__name__)

# Store for temporary files (use Redis/database in production)
temp_files = {}

@app.post("/chat-webhook")
async def chat_webhook(request: Request):
    """Handle Google Chat webhook events"""
    try:
        event = await request.json()
        logger.info(f"=== RECEIVED GOOGLE CHAT EVENT ===")
        logger.info(f"Event type: {event.get('type')}")
        logger.info(f"Full event: {json.dumps(event, indent=2)}")
        
        event_type = event.get("type")
        
        if event_type == "ADDED_TO_SPACE":
            return handle_added_to_space(event)
        elif event_type == "REMOVED_FROM_SPACE":
            return handle_removed_from_space(event)
        elif event_type == "MESSAGE":
            return await handle_message(event)
        else:
            logger.warning(f"Unknown event type: {event_type}")
            return {"text": "Unknown event type"}
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}", exc_info=True)
        return {
            "text": "❌ Sorry, there was an error processing your request.",
            "cards": [{
                "sections": [{
                    "widgets": [{
                        "textParagraph": {
                            "text": f"Debug Error: {str(e)}"
                        }
                    }]
                }]
            }]
        }

def handle_added_to_space(event: Dict[Any, Any]) -> Dict[str, Any]:
    """Handle bot being added to space"""
    space = event.get("space", {})
    user = event.get("user", {})
    
    logger.info(f"Bot added to space: {space.get('name', 'unknown')}")
    
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
    return {}

async def handle_message(event: Dict[Any, Any]) -> Dict[str, Any]:
    """Handle message events with detailed debugging"""
    message = event.get("message", {})
    attachments = message.get("attachment", [])
    text = message.get("text", "").strip()
    user = event.get("user", {})
    
    logger.info(f"=== PROCESSING MESSAGE ===")
    logger.info(f"Message text: {text}")
    logger.info(f"Attachments count: {len(attachments)}")
    logger.info(f"User: {user.get('displayName', 'Unknown')}")
    
    # Debug: Print attachment details
    for i, attachment in enumerate(attachments):
        logger.info(f"Attachment {i}: {json.dumps(attachment, indent=2)}")
    
    # If no attachments, provide instructions
    if not attachments:
        return {
            "text": "👋 Hello! To generate a quotation, please attach your documents to this message.\n\n"
                   "📎 **Supported formats:** PDF, DOCX, DOC, TXT, RTF\n"
                   "📊 I'll analyze your documents and create an Excel quotation file.\n\n"
                   "🔧 **Debug Mode:** This response means no attachments were detected."
        }
    
    try:
        # Process attachments with detailed logging
        extracted_texts = []
        processed_files = []
        debug_info = []
        
        for i, attachment in enumerate(attachments):
            attachment_name = attachment.get("name", f"unknown_file_{i}")
            drive_file_id = attachment.get("driveFile", {}).get("id")
            
            logger.info(f"Processing attachment: {attachment_name}")
            logger.info(f"Drive file ID: {drive_file_id}")
            
            debug_info.append(f"📎 File: {attachment_name}")
            
            if not drive_file_id:
                logger.warning(f"No drive file ID for {attachment_name}")
                debug_info.append(f"   ❌ No Drive file ID")
                continue
            
            # Try to download file from Google Drive
            logger.info(f"Attempting to download file: {drive_file_id}")
            file_content = await download_drive_file_debug(drive_file_id, event, attachment_name)
            
            if not file_content:
                logger.error(f"Failed to download file: {attachment_name}")
                debug_info.append(f"   ❌ Download failed")
                continue
            
            debug_info.append(f"   ✅ Downloaded ({len(file_content)} bytes)")
            
            # Save temporarily and extract text
            file_suffix = Path(attachment_name).suffix.lower()
            logger.info(f"File suffix: {file_suffix}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                if is_valid_file_type(attachment_name):
                    logger.info(f"Extracting text from: {temp_file_path}")
                    text_content = extract_text_from_file(temp_file_path, attachment_name)
                    
                    if text_content and text_content.strip():
                        extracted_texts.append(text_content)
                        processed_files.append(attachment_name)
                        debug_info.append(f"   ✅ Text extracted ({len(text_content)} chars)")
                        logger.info(f"Successfully extracted {len(text_content)} characters from {attachment_name}")
                    else:
                        logger.warning(f"No text extracted from {attachment_name}")
                        debug_info.append(f"   ❌ No text content")
                else:
                    logger.warning(f"Invalid file type: {attachment_name}")
                    debug_info.append(f"   ❌ Unsupported file type")
            except Exception as e:
                logger.error(f"Error processing {attachment_name}: {str(e)}", exc_info=True)
                debug_info.append(f"   ❌ Processing error: {str(e)}")
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        # Create debug response
        debug_message = "\n".join(debug_info)
        
        if not extracted_texts:
            return {
                "text": f"❌ No valid text could be extracted from the attached files.\n\n"
                       f"**Debug Information:**\n{debug_message}\n\n"
                       f"**Supported formats:** PDF, DOCX, DOC, TXT, RTF\n"
                       f"**Common issues:**\n"
                       f"• Scanned PDFs (need OCR)\n"
                       f"• Password-protected files\n"
                       f"• Corrupted files\n"
                       f"• Empty files\n"
                       f"• Authentication issues with Google Drive"
            }
        
        # Generate quotation
        combined_text = "\n".join(extracted_texts)
        logger.info(f"Combined text length: {len(combined_text)} characters")
        
        quotation_data = extract_json_from_response(combined_text)
        excel_file_path = create_excel_from_quotation(quotation_data)
        
        # Store file temporarily
        file_id = str(uuid.uuid4())
        temp_files[file_id] = excel_file_path
        
        # Use your actual EC2 public IP/domain
        download_url = f"http://51.21.187.10:8000/download/{file_id}"
        
        return {
            "text": f"✅ **Quotation Generated Successfully!**\n\n"
                   f"📁 Processed files: {', '.join(processed_files)}\n"
                   f"📊 Download your Excel quotation: [Click here]({download_url})\n\n"
                   f"⏰ Download link expires in 1 hour.\n\n"
                   f"**Debug Info:**\n{debug_message}",
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
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        return {
            "text": f"❌ Error generating quotation: {str(e)}\n"
                   f"Please try again or contact support.\n\n"
                   f"**Debug Info:**\n{debug_message if 'debug_message' in locals() else 'No debug info available'}"
        }

async def download_drive_file_debug(file_id: str, event: Dict[Any, Any], filename: str) -> bytes:
    """Download file from Google Drive with detailed debugging"""
    logger.info(f"=== DOWNLOADING DRIVE FILE ===")
    logger.info(f"File ID: {file_id}")
    logger.info(f"Filename: {filename}")
    
    try:
        # Get the token from the event
        token = None
        if "token" in event:
            token = event["token"]
        elif "message" in event and "token" in event["message"]:
            token = event["message"]["token"]
        
        if not token:
            logger.error("No authentication token found in event")
            return None
            
        # Get the download URL from the attachment
        message = event.get("message", {})
        attachments = message.get("attachment", [])
        
        target_attachment = None
        for attachment in attachments:
            if attachment.get("driveFile", {}).get("id") == file_id:
                target_attachment = attachment
                break
        
        if not target_attachment:
            logger.error(f"Could not find attachment with file ID: {file_id}")
            return None
            
        drive_file = target_attachment.get("driveFile", {})
        download_url = drive_file.get("downloadUrl")
        
        if not download_url:
            logger.error("No download URL found in attachment")
            return None
            
        # Download the file using the URL and token
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "*/*"
            }
            
            response = await client.get(download_url, headers=headers)
            
            if response.status_code == 200:
                logger.info(f"Successfully downloaded file: {len(response.content)} bytes")
                return response.content
            else:
                logger.error(f"Failed to download file. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"Error downloading Drive file {file_id}: {str(e)}", exc_info=True)
        return None

@app.post("/process-documents")
async def process_documents(files: List[UploadFile] = File(...)):
    """Main endpoint for direct file upload and processing"""
    try:
        logger.info(f"=== PROCESSING DIRECT UPLOAD ===")
        logger.info(f"Number of files: {len(files)}")
        
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Extract text from all uploaded files
        extracted_texts = []
        processed_files = []
        
        for file in files:
            logger.info(f"Processing file: {file.filename}")
            
            # Validate file type
            if not is_valid_file_type(file.filename):
                logger.error(f"Unsupported file type: {file.filename}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file.filename}"
                )
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
                
            logger.info(f"Saved temp file: {temp_file_path}, size: {len(content)} bytes")
            
            try:
                # Extract text from file
                text = extract_text_from_file(temp_file_path, file.filename)
                logger.info(f"Extracted text length: {len(text) if text else 0}")
                
                if text and text.strip():
                    extracted_texts.append(text)
                    processed_files.append(file.filename)
                    logger.info(f"Successfully processed: {file.filename}")
                else:
                    logger.warning(f"No text extracted from {file.filename}")
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Error processing {file.filename}: {str(e)}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        # Combine all extracted text
        input_docs_text = "\n".join(extracted_texts)
        
        if not input_docs_text.strip():
            raise HTTPException(
                status_code=400, 
                detail="No text could be extracted from the files. Please check if files are corrupted or in unsupported format."
            )
        
        logger.info(f"Combined extracted text length: {len(input_docs_text)} characters")
        
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
            "file_id": file_id,
            "processed_files": processed_files,
            "text_length": len(input_docs_text)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """Download generated Excel file"""
    if file_id not in temp_files:
        raise HTTPException(status_code=404, detail="File not found or expired")
    
    file_path = temp_files[file_id]
    
    if not os.path.exists(file_path):
        temp_files.pop(file_id, None)
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="quotation.xlsx"
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

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check system status"""
    return {
        "status": "debug",
        "temp_files_count": len(temp_files),
        "temp_files_ids": list(temp_files.keys()),
        "supported_extensions": ['.pdf', '.docx', '.doc', '.txt', '.rtf']
    }

@app.get("/")
async def root():
    return {
        "message": "Document Quotation API with Google Chat Integration - DEBUG MODE",
        "endpoints": {
            "upload": "/process-documents",
            "download": "/download/{file_id}",
            "webhook": "/chat-webhook",
            "debug": "/debug",
            "health": "/health"
        }
    }