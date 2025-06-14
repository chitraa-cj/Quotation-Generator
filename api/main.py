from fastapi import FastAPI, Request, UploadFile, File, HTTPException, BackgroundTasks
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
import asyncio
import httpx
import google.generativeai as genai
from components.extract_json import (
    RobustJSONExtractor, 
    extract_json_from_response,
    extract_json_content,
    aggressive_json_repair,
    repair_json_with_llm,
    create_fallback_structure
)
from components.create_excel import create_excel_from_quotation
from components.generate_quotation import generate_quotation_prompt
from utils import extract_text_from_file
import json5

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
# Store for processing status
processing_status = {}

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
        # Use the API key directly
        api_key = 'AIzaSyC_6vPDvqyOnJTWl434LScSxh_ndAjmxDI'
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return None

async def send_chat_message(space_name: str, message_data: Dict[str, Any]):
    """Send a message to Google Chat space using webhook or API"""
    try:
        # This is a placeholder - you'll need to implement the actual sending logic
        # using Google Chat API or webhook URL if available
        logger.info(f"Sending message to {space_name}: {message_data}")
        
        # For now, we'll just log it. In production, you'd use:
        # - Google Chat API with service account
        # - Or store the webhook URL from the original request
        pass
    except Exception as e:
        logger.error(f"Failed to send chat message: {str(e)}")

async def process_documents_async(
    space_name: str,
    processed_files: List[Dict],
    processing_id: str,
    user_display_name: str = "User"
):
    """Process documents asynchronously and send result to chat"""
    try:
        processing_status[processing_id] = {"status": "processing", "progress": "Extracting text..."}
        
        # Process the files that were downloaded by Apps Script
        extracted_texts = []
        processed_file_names = []
        debug_info = []
        
        # Process files
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
            processing_status[processing_id] = {"status": "error", "message": "No valid text extracted"}
            
            error_message = {
                "text": f"❌ No valid text could be extracted from the attached files.\n\n"
                       f"**Debug Information:**\n{debug_message}\n\n"
                       f"**Supported formats:** PDF, DOCX, DOC, TXT, RTF\n"
                       f"**Common issues:**\n"
                       f"• Scanned PDFs (need OCR)\n"
                       f"• Password-protected files\n"
                       f"• Corrupted files\n"
                       f"• Empty files"
            }
            
            await send_chat_message(space_name, error_message)
            return
        
        # Update status
        processing_status[processing_id] = {"status": "processing", "progress": "Generating quotation..."}
        
        # Generate quotation
        combined_text = "\n".join(extracted_texts)
        logger.info(f"Combined text length: {len(combined_text)} characters")
        
        # Initialize model and generate quotation
        model = initialize_model()
        if not model:
            processing_status[processing_id] = {"status": "error", "message": "Failed to initialize AI model"}
            
            error_message = {
                "text": "❌ Failed to initialize AI model. Please try again later."
            }
            await send_chat_message(space_name, error_message)
            return
        
        # Generate quotation with retry logic
        try:
            # Generate prompt and get response
            prompt = generate_quotation_prompt([], combined_text)
            response_text = generate_quotation_with_retry(model, prompt)
            
            if not response_text:
                raise Exception("Empty response from AI model")
            
            # Print raw response to terminal for debugging
            logger.info("\n=== Raw AI Response ===")
            logger.info(response_text)
            
            # Update status
            processing_status[processing_id] = {"status": "processing", "progress": "Creating Excel file..."}
            
            # Extract and parse the response using the same method as Streamlit
            quotation_data = None
            
            # First try direct JSON parsing
            try:
                json_content = extract_json_content(response_text)
                quotation_data = json.loads(json_content)
                logger.info("Successfully parsed JSON with standard parser")
            except Exception as e:
                logger.warning(f"Standard JSON parsing failed: {str(e)}")
                # Try json5 parsing
                try:
                    json_content = extract_json_content(response_text)
                    quotation_data = json5.loads(json_content)
                    logger.info("Successfully parsed JSON with json5")
                except Exception as e:
                    logger.warning(f"JSON5 parsing failed: {str(e)}")
                    # Try aggressive repair
                    try:
                        repaired_text = aggressive_json_repair(response_text)
                        logger.info("Repaired text: " + (repaired_text[:200] + "..." if len(repaired_text) > 200 else repaired_text))
                        
                        # Try both json and json5
                        for parser in [json.loads, json5.loads]:
                            try:
                                quotation_data = parser(repaired_text)
                                logger.info("Successfully parsed JSON with aggressive repair")
                                break
                            except Exception:
                                continue
                    except Exception as e:
                        logger.warning(f"Aggressive repair failed: {str(e)}")
                        # Try LLM repair if available
                        try:
                            llm_repaired_text = repair_json_with_llm(response_text, model)
                            if llm_repaired_text:
                                for parser in [json.loads, json5.loads]:
                                    try:
                                        quotation_data = parser(llm_repaired_text)
                                        logger.info("Successfully parsed JSON with LLM repair")
                                        break
                                    except Exception:
                                        continue
                        except Exception as e:
                            logger.warning(f"LLM repair failed: {str(e)}")
                            # Last resort: create fallback structure
                            logger.warning("All parsing methods failed, creating fallback structure")
                            quotation_data = create_fallback_structure(response_text)
            
            # If all parsing attempts failed, use fallback structure
            if not quotation_data:
                logger.warning("All parsing methods failed, using fallback structure")
                quotation_data = create_fallback_structure(response_text)
            
            # Create Excel file with proper formatting
            excel_file_path = create_excel_from_quotation(quotation_data)
            
            if not os.path.exists(excel_file_path):
                raise Exception("Failed to create Excel file")
            
            # Store file temporarily with unique ID
            file_id = str(uuid.uuid4())
            temp_files[file_id] = excel_file_path
            
            # Update status to completed
            processing_status[processing_id] = {"status": "completed", "file_id": file_id}
            
            # Create download URL
            download_url = f"https://51.21.187.10:8000/download/{file_id}"
            
            # Send success message
            success_message = {
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
            
            await send_chat_message(space_name, success_message)
            
        except Exception as e:
            logger.error(f"Error generating quotation: {str(e)}", exc_info=True)
            processing_status[processing_id] = {"status": "error", "message": str(e)}
            
            error_message = {
                "text": f"❌ Error generating quotation: {str(e)}\n"
                       f"Please try again or contact support.\n\n"
                       f"**Debug Info:**\n{debug_message}"
            }
            await send_chat_message(space_name, error_message)
        
    except Exception as e:
        logger.error(f"Error in async processing: {str(e)}", exc_info=True)
        processing_status[processing_id] = {"status": "error", "message": str(e)}
        
        error_message = {
            "text": f"❌ Unexpected error during processing: {str(e)}\n"
                   f"Please try again or contact support."
        }
        await send_chat_message(space_name, error_message)

@app.post("/chat-webhook")
async def chat_webhook(request: Request, background_tasks: BackgroundTasks):
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
            message = event.get("message", {})
            text = message.get("text", "").strip()
            processed_files = event.get("processedFiles", [])
            space = event.get("space", {})
            user = event.get("user", {})
            
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
                               "• Unsupported file type\n"
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
            
            # If there are files, send immediate response and process in background
            processing_id = str(uuid.uuid4())
            processing_status[processing_id] = {"status": "starting", "progress": "Initializing..."}
            
            # Start background processing
            background_tasks.add_task(
                process_documents_async,
                space.get("name", "unknown_space"),
                processed_files,
                processing_id,
                user.get("displayName", "User")
            )
            
            # Return immediate response (must be within 30 seconds)
            return {
                "text": f"🔄 **Processing your documents...**\n\n"
                       f"📄 Found {len(processed_files)} file(s) to process\n"
                       f"⏳ This may take a few minutes. I'll send you the results when ready!\n\n"
                       f"🆔 Processing ID: `{processing_id}`\n"
                       f"💡 You can check status at: `/status {processing_id}`"
            }
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
    message += "• Just attach files and mention me!\n\n"
    message += "⚡ **Note:** Processing may take a few minutes for large files."
    
    return {"text": message}

def handle_removed_from_space(event: Dict[Any, Any]) -> Dict[str, Any]:
    """Handle bot being removed from space"""
    logger.info(f"Bot removed from space: {event.get('space', {}).get('name', 'unknown')}")
    return {}

@app.get("/status/{processing_id}")
async def get_processing_status(processing_id: str):
    """Get processing status"""
    if processing_id not in processing_status:
        raise HTTPException(status_code=404, detail="Processing ID not found")
    
    status_info = processing_status[processing_id]
    
    if status_info["status"] == "completed" and "file_id" in status_info:
        download_url = f"https://51.21.187.10:8000/download/{status_info['file_id']}"
        return {
            "status": "completed",
            "message": "Processing completed successfully!",
            "download_url": download_url
        }
    
    return status_info

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
        "processing_jobs": len(processing_status),
        "processing_ids": list(processing_status.keys()),
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
            "status": "/status/{processing_id}",
            "debug": "/debug",
            "health": "/health"
        }
    }