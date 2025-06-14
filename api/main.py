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
import base64
import time
import random
import google.generativeai as genai
from components.extract_json import RobustJSONExtractor, extract_json_from_response
from components.create_excel import create_excel_from_quotation
from components.generate_quotation import generate_quotation_prompt
from utils import extract_text_from_file

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Store for temporary files (use Redis/database in production)
temp_files = {}

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

def initialize_model():
    """Initialize the Gemini model"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return None

@app.post("/chat-webhook")
async def chat_webhook(request: Request):
    """Handle Google Chat webhook events"""
    try:
        event = await request.json()
        logger.info(f"=== RECEIVED GOOGLE CHAT EVENT ===")
        logger.info(f"Event type: {event.get('type')}")
        
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
        message = f"👋 Hello {user.get('displayName', 'User')}! Welcome to your Document Quotation Assistant!"
    else:
        space_name = space.get("displayName", "this chat")
        message = f"👋 Thank you for adding me to {space_name}!"
    
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
    """Handle message events with processed files"""
    message = event.get("message", {})
    text = message.get("text", "").strip()
    user = event.get("user", {})
    processed_files = event.get("processedFiles", [])
    
    logger.info(f"=== PROCESSING MESSAGE ===")
    logger.info(f"Message text: {text}")
    logger.info(f"Processed files count: {len(processed_files)}")
    logger.info(f"User: {user.get('displayName', 'Unknown')}")
    
    # If no processed files, provide instructions
    if not processed_files:
        # Check if there were attachments that failed to process
        attachments = message.get("attachment", [])
        if attachments:
            return {
                "text": "❌ I couldn't process the attached files.\n\n"
                       "**Common issues:**\n"
                       "• Files may be corrupted or password-protected\n"
                       "• Unsupported file format\n"
                       "• File access permissions\n\n"
                       "**Supported formats:** PDF, DOCX, DOC, TXT, RTF\n"
                       "Please try uploading the files again or check if they're accessible."
            }
        else:
            return {
                "text": "👋 Hello! To generate a quotation, please attach your documents to this message.\n\n"
                       "📎 **Supported formats:** PDF, DOCX, DOC, TXT, RTF\n"
                       "📊 I'll analyze your documents and create an Excel quotation file."
            }
    
    try:
        # Process the files that were downloaded by Apps Script
        extracted_texts = []
        processed_file_names = []
        debug_info = []
        
        for file_data in processed_files:
            file_name = file_data.get("name", "unknown_file")
            file_content_b64 = file_data.get("content", "")
            file_size = file_data.get("size", 0)
            
            logger.info(f"Processing file: {file_name} ({file_size} bytes)")
            debug_info.append(f"📎 {file_name} ({file_size} bytes)")
            
            if not file_content_b64:
                logger.warning(f"No content for {file_name}")
                debug_info.append(f"   ❌ No content")
                continue
            
            try:
                # Decode base64 content
                file_content = base64.b64decode(file_content_b64)
                logger.info(f"Decoded {len(file_content)} bytes for {file_name}")
                
                # Validate file type
                if not is_valid_file_type(file_name):
                    logger.warning(f"Invalid file type: {file_name}")
                    debug_info.append(f"   ❌ Unsupported file type")
                    continue
                
                # Save temporarily and extract text
                file_suffix = Path(file_name).suffix.lower()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name
                
                try:
                    logger.info(f"Extracting text from: {temp_file_path}")
                    text_content = extract_text_from_file(temp_file_path, file_name)
                    
                    if text_content and text_content.strip():
                        extracted_texts.append(text_content)
                        processed_file_names.append(file_name)
                        debug_info.append(f"   ✅ Extracted {len(text_content)} characters")
                        logger.info(f"Successfully extracted {len(text_content)} characters from {file_name}")
                    else:
                        logger.warning(f"No text extracted from {file_name}")
                        debug_info.append(f"   ❌ No text content")
                        
                except Exception as extract_error:
                    logger.error(f"Error extracting text from {file_name}: {str(extract_error)}")
                    debug_info.append(f"   ❌ Extraction error: {str(extract_error)}")
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        
            except Exception as decode_error:
                logger.error(f"Error decoding {file_name}: {str(decode_error)}")
                debug_info.append(f"   ❌ Decode error: {str(decode_error)}")
        
        # Create debug message
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
                       f"• Empty files"
            }
        
        # Generate quotation
        combined_text = "\n".join(extracted_texts)
        logger.info(f"Combined text length: {len(combined_text)} characters")
        
        # Initialize model and generate quotation
        model = initialize_model()
        if not model:
            raise HTTPException(status_code=500, detail="Failed to initialize AI model")
        
        # Generate quotation with retry logic
        try:
            # Generate prompt and get response
            prompt = generate_quotation_prompt([], combined_text)  # Empty examples list for now
            response_text = generate_quotation_with_retry(model, prompt)
            
            if not response_text:
                raise HTTPException(status_code=500, detail="Empty response from AI model")
            
            # Print raw response to terminal for debugging
            logger.info("\n=== Raw AI Response ===")
            logger.info(response_text)
            
            # Extract and parse the response
            try:
                quotation_data = extract_json_from_response(response_text)
                if not quotation_data:
                    raise HTTPException(status_code=500, detail="Failed to parse the AI response into a valid quotation format")
                
                # Create Excel file with proper formatting
                excel_file_path = create_excel_from_quotation(quotation_data)
                
                if not os.path.exists(excel_file_path):
                    raise HTTPException(status_code=500, detail="Failed to create Excel file")
                
                # Store file temporarily with unique ID
                file_id = str(uuid.uuid4())
                temp_files[file_id] = excel_file_path
                
                # Create download URL
                download_url = f"http://51.21.187.10:8000/download/{file_id}"
                
                return {
                    "text": f"✅ **Quotation Generated Successfully!**\n\n"
                           f"📁 **Processed files:** {', '.join(processed_file_names)}\n"
                           f"📊 **Download your Excel quotation:** [Click here]({download_url})\n\n"
                           f"⏰ Download link expires in 1 hour.\n\n"
                           f"**Processing Details:**\n{debug_message}",
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
                logger.error(f"Error parsing AI response: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error parsing AI response: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error generating quotation: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error generating quotation: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        return {
            "text": f"❌ Error generating quotation: {str(e)}\n"
                   f"Please try again or contact support.\n\n"
                   f"**Debug Info:**\n{debug_message if 'debug_message' in locals() else 'No debug info available'}"
        }

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
        
        # Initialize model and generate quotation
        model = initialize_model()
        if not model:
            raise HTTPException(status_code=500, detail="Failed to initialize AI model")
        
        # Generate quotation with retry logic
        try:
            # Generate prompt and get response
            prompt = generate_quotation_prompt([], input_docs_text)  # Empty examples list for now
            response_text = generate_quotation_with_retry(model, prompt)
            
            if not response_text:
                raise HTTPException(status_code=500, detail="Empty response from AI model")
            
            # Print raw response to terminal for debugging
            logger.info("\n=== Raw AI Response ===")
            logger.info(response_text)
            
            # Extract and parse the response
            try:
                quotation_data = extract_json_from_response(response_text)
                if not quotation_data:
                    raise HTTPException(status_code=500, detail="Failed to parse the AI response into a valid quotation format")
                
                # Create Excel file with proper formatting
                excel_file_path = create_excel_from_quotation(quotation_data)
                
                if not os.path.exists(excel_file_path):
                    raise HTTPException(status_code=500, detail="Failed to create Excel file")
                
                # Store file temporarily with unique ID
                file_id = str(uuid.uuid4())
                temp_files[file_id] = excel_file_path
                
                return {
                    "message": "Quotation generated successfully!",
                    "download_url": f"/download/{file_id}",
                    "file_id": file_id,
                    "processed_files": processed_files,
                    "text_length": len(input_docs_text),
                    "extraction_strategy": "direct_parse",  # Since we're using extract_json_from_response
                    "needs_review": False
                }
                
            except Exception as e:
                logger.error(f"Error parsing AI response: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error parsing AI response: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error generating quotation: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error generating quotation: {str(e)}")
        
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
        "message": "Document Quotation API with Google Chat Integration",
        "endpoints": {
            "upload": "/process-documents",
            "download": "/download/{file_id}",
            "webhook": "/chat-webhook",
            "debug": "/debug",
            "health": "/health"
        }
    }