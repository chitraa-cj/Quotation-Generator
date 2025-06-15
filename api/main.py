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
from unittest.mock import Mock
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
from datetime import datetime, timedelta

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

# Constants
GOOGLE_CHAT_TIMEOUT = 30  # seconds - hard limit from Google
MAX_RESPONSE_TIME = 25    # seconds - our safe limit
CLEANUP_INTERVAL = 3600   # 1 hour

def cleanup_old_files():
    """Clean up old temporary files and processing status"""
    current_time = time.time()
    cutoff_time = current_time - CLEANUP_INTERVAL
    
    # Clean up old temp files
    expired_files = []
    for file_id, file_path in temp_files.items():
        try:
            if os.path.exists(file_path):
                file_stat = os.stat(file_path)
                if file_stat.st_mtime < cutoff_time:
                    os.unlink(file_path)
                    expired_files.append(file_id)
            else:
                expired_files.append(file_id)
        except Exception as e:
            logger.warning(f"Error cleaning up file {file_id}: {e}")
            expired_files.append(file_id)
    
    for file_id in expired_files:
        temp_files.pop(file_id, None)
    
    # Clean up old processing status
    expired_status = []
    for processing_id, status_info in processing_status.items():
        if status_info.get('created_at', current_time) < cutoff_time:
            expired_status.append(processing_id)
    
    for processing_id in expired_status:
        processing_status.pop(processing_id, None)
    
    logger.info(f"Cleaned up {len(expired_files)} files and {len(expired_status)} processing records")

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

async def send_chat_message_with_retry(space_name: str, message_data: Dict[str, Any], max_retries: int = 3):
    """Send a message to Google Chat space with retry logic"""
    for attempt in range(max_retries):
        try:
            # For now, just log the message - you'll need to implement actual sending
            logger.info(f"=== SENDING TO CHAT (Attempt {attempt + 1}) ===")
            logger.info(f"Space: {space_name}")
            logger.info(f"Message: {json.dumps(message_data, indent=2)[:500]}...")
            
            # TODO: Implement actual message sending with your preferred method:
            # Option 1: Use Google Chat API with service account credentials
            # Option 2: Use stored webhook URL from space registration
            # Option 3: Use Google Chat REST API
            
            # Simulate network delay for testing
            await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send chat message (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                return False

async def process_documents_async(
    space_name: str,
    processed_files: List[Dict],
    file_id: str,
    user_display_name: str = "User"
):
    """Process documents asynchronously and send result to chat"""
    start_time = time.time()
    
    try:
        logger.info(f"Starting async processing for {file_id}")
        
        # Send initial detailed status
        await send_chat_message_with_retry(space_name, {
            "text": f"🔄 **Processing Started**\n\n"
                   f"📊 **Files:** {len(processed_files)} document(s)\n"
                   f"⏱️ **Started:** {datetime.now().strftime('%H:%M:%S')}\n"
                   f"📥 **Download:** [Click here]({processing_status[file_id]['download_url']})\n\n"
                   f"**Status:** Extracting text from files..."
        })
        
        processing_status[file_id].update({
            "status": "processing", 
            "progress": "Extracting text...",
            "created_at": start_time,
            "current_step": "text_extraction"
        })
        
        # Process files with progress tracking
        extracted_texts = []
        processed_file_names = []
        debug_info = []
        
        for i, file_data in enumerate(processed_files, 1):
            file_name = file_data.get("name", f"file_{i}")
            file_content_b64 = file_data.get("content", "")
            file_size = file_data.get("size", 0)
            
            logger.info(f"Processing file {i}/{len(processed_files)}: {file_name} ({file_size} bytes)")
            debug_info.append(f"📎 **{file_name}** ({file_size:,} bytes)")
            
            if not file_content_b64:
                logger.warning(f"No content for {file_name}")
                debug_info.append(f"   ❌ No content received")
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
                        debug_info.append(f"   ✅ Extracted {len(text_content):,} characters")
                        logger.info(f"Successfully extracted {len(text_content)} characters from {file_name}")
                    else:
                        logger.warning(f"No text extracted from {file_name}")
                        debug_info.append(f"   ⚠️ No readable text found")
                        
                except Exception as extract_error:
                    logger.error(f"Error extracting text from {file_name}: {str(extract_error)}")
                    debug_info.append(f"   ❌ Extraction failed: {str(extract_error)[:50]}...")
                    
                finally:
                    # Clean up temp file
                    try:
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
                        
            except Exception as decode_error:
                logger.error(f"Error decoding {file_name}: {str(decode_error)}")
                debug_info.append(f"   ❌ Decode error: {str(decode_error)[:50]}...")
        
        # Check if we have any extracted text
        if not extracted_texts:
            processing_status[file_id] = {
                "status": "error", 
                "message": "No valid text extracted",
                "created_at": start_time,
                "completed_at": time.time()
            }
            
            error_message = {
                "text": f"❌ **Processing Failed**\n\n"
                       f"No readable text could be extracted from the files.\n\n"
                       f"**Files Processed:**\n" + "\n".join(debug_info[:10]) + 
                       (f"\n... and {len(debug_info) - 10} more" if len(debug_info) > 10 else "") +
                       f"\n\n**Supported Formats:** PDF, DOCX, DOC, TXT, RTF\n"
                       f"**Common Issues:**\n"
                       f"• Scanned PDFs without OCR\n"
                       f"• Password-protected files\n"
                       f"• Corrupted/empty files\n"
                       f"• Image-only documents\n\n"
                       f"🆔 ID: `{file_id}`"
            }
            
            await send_chat_message_with_retry(space_name, error_message)
            return
        
        # Send progress update with more details
        combined_length = sum(len(text) for text in extracted_texts)
        await send_chat_message_with_retry(space_name, {
            "text": f"🔄 **Processing Update**\n\n"
                   f"✅ **Text Extraction Complete**\n"
                   f"📄 Processed: {len(processed_file_names)} file(s)\n"
                   f"📝 Total text: {combined_length:,} characters\n"
                   f"🤖 Generating AI quotation...\n\n"
                   f"🆔 ID: `{file_id}`"
        })
        
        # Update status
        processing_status[file_id].update({
            "status": "processing", 
            "progress": "Generating quotation...",
            "current_step": "ai_generation",
            "files_processed": len(processed_file_names),
            "text_length": combined_length
        })
        
        # Generate quotation
        combined_text = "\n".join(extracted_texts)
        logger.info(f"Combined text length: {len(combined_text)} characters")
        
        # Initialize model
        model = initialize_model()
        if not model:
            processing_status[file_id] = {
                "status": "error", 
                "message": "Failed to initialize AI model",
                "created_at": start_time,
                "completed_at": time.time()
            }
            
            await send_chat_message_with_retry(space_name, {
                "text": f"❌ **AI Model Error**\n\n"
                       f"Failed to initialize the AI model.\n"
                       f"Please try again in a few minutes.\n\n"
                       f"🆔 ID: `{file_id}`"
            })
            return
        
        # Generate quotation with timeout protection
        try:
            # Generate prompt and get response
            prompt = generate_quotation_prompt([], combined_text)
            
            # Add timeout to AI generation
            ai_start_time = time.time()
            logger.info("Starting AI quotation generation...")
            
            response_text = generate_quotation_with_retry(model, prompt)
            
            ai_duration = time.time() - ai_start_time
            logger.info(f"AI generation completed in {ai_duration:.1f} seconds")
            
            if not response_text:
                raise Exception("Empty response from AI model")
            
            # Print raw response to terminal for debugging
            logger.info("\n=== Raw AI Response (First 500 chars) ===")
            logger.info(response_text[:500] + ("..." if len(response_text) > 500 else ""))
            
            # Send Excel generation update
            await send_chat_message_with_retry(space_name, {
                "text": f"🔄 **Almost Done!**\n\n"
                       f"✅ AI quotation generated\n"
                       f"📊 Creating Excel spreadsheet...\n\n"
                       f"🆔 ID: `{file_id}`"
            })
            
            # Update status
            processing_status[file_id].update({
                "status": "processing", 
                "progress": "Creating Excel file...",
                "current_step": "excel_generation"
            })
            
            # Parse AI response with comprehensive error handling
            quotation_data = None
            parsing_method = "unknown"
            
            # Try multiple parsing strategies
            try:
                json_content = extract_json_content(response_text)
                quotation_data = json.loads(json_content)
                parsing_method = "standard_json"
                logger.info("Successfully parsed with standard JSON")
            except Exception as e:
                logger.warning(f"Standard JSON parsing failed: {str(e)}")
                try:
                    json_content = extract_json_content(response_text)
                    quotation_data = json5.loads(json_content)
                    parsing_method = "json5"
                    logger.info("Successfully parsed with JSON5")
                except Exception as e:
                    logger.warning(f"JSON5 parsing failed: {str(e)}")
                    try:
                        repaired_text = aggressive_json_repair(response_text)
                        quotation_data = json.loads(repaired_text)
                        parsing_method = "aggressive_repair"
                        logger.info("Successfully parsed with aggressive repair")
                    except Exception as e:
                        logger.warning(f"Aggressive repair failed: {str(e)}")
                        try:
                            llm_repaired_text = repair_json_with_llm(response_text, model)
                            if llm_repaired_text:
                                quotation_data = json.loads(llm_repaired_text)
                                parsing_method = "llm_repair"
                                logger.info("Successfully parsed with LLM repair")
                        except Exception as e:
                            logger.warning(f"LLM repair failed: {str(e)}")
                            quotation_data = create_fallback_structure(response_text)
                            parsing_method = "fallback_structure"
                            logger.info("Using fallback structure")
            
            if not quotation_data:
                quotation_data = create_fallback_structure(response_text)
                parsing_method = "emergency_fallback"
                logger.warning("Using emergency fallback structure")
            
            # Create Excel file
            excel_start_time = time.time()
            excel_file_path = create_excel_from_quotation(quotation_data)
            excel_duration = time.time() - excel_start_time
            
            if not os.path.exists(excel_file_path):
                raise Exception("Failed to create Excel file")
            
            # Store file with metadata
            file_id = str(uuid.uuid4())
            temp_files[file_id] = excel_file_path
            
            # Update status to completed
            total_duration = time.time() - start_time
            processing_status[file_id].update({
                "status": "completed", 
                "created_at": start_time,
                "completed_at": time.time(),
                "total_duration": total_duration,
                "files_processed": len(processed_file_names),
                "text_length": combined_length,
                "parsing_method": parsing_method,
                "ai_duration": ai_duration,
                "excel_duration": excel_duration
            })
            
            # Create download URL
            download_url = f"https://quotebot-bluelabel.xoidlabs.com/api/download/{file_id}"
            
            # Send comprehensive success message
            success_message = {
                "text": f"✅ **Quotation Generated Successfully!**\n\n"
                       f"📊 **Excel file ready for download**\n"
                       f"📁 **Files processed:** {len(processed_file_names)}\n"
                       f"📝 **Text extracted:** {combined_length:,} characters\n"
                       f"⏱️ **Processing time:** {total_duration:.1f} seconds\n"
                       f"🤖 **AI method:** {parsing_method.replace('_', ' ').title()}\n\n"
                       f"**📥 [Download Excel Quotation]({download_url})**\n\n"
                       f"⏰ *Download link expires in 1 hour*",
                       
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
            
            await send_chat_message_with_retry(space_name, success_message)
            logger.info(f"Processing completed successfully for {file_id} in {total_duration:.1f}s")
            
        except Exception as e:
            error_duration = time.time() - start_time
            logger.error(f"Error in quotation generation: {str(e)}", exc_info=True)
            
            processing_status[file_id] = {
                "status": "error", 
                "message": str(e),
                "created_at": start_time,
                "completed_at": time.time(),
                "error_duration": error_duration
            }
            
            error_message = {
                "text": f"❌ **Quotation Generation Failed**\n\n"
                       f"**Error:** {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}\n"
                       f"**Duration:** {error_duration:.1f} seconds\n\n"
                       f"Please try again or contact support if the issue persists.\n\n"
                       f"🆔 ID: `{file_id}`"
            }
            await send_chat_message_with_retry(space_name, error_message)
        
    except Exception as e:
        error_duration = time.time() - start_time
        logger.error(f"Critical error in async processing: {str(e)}", exc_info=True)
        
        processing_status[file_id] = {
            "status": "error", 
            "message": f"Critical processing error: {str(e)}",
            "created_at": start_time,
            "completed_at": time.time(),
            "error_duration": error_duration
        }
        
        error_message = {
            "text": f"❌ **System Error**\n\n"
                   f"A critical error occurred during processing.\n"
                   f"**Error:** {str(e)[:150]}{'...' if len(str(e)) > 150 else ''}\n"
                   f"**Duration:** {error_duration:.1f} seconds\n\n"
                   f"Please try again or contact support.\n\n"
                   f"🆔 ID: `{file_id}`"
        }
        await send_chat_message_with_retry(space_name, error_message)

@app.post("/chat-webhook")
async def chat_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle Google Chat webhook events with optimized response time"""
    webhook_start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    try:
        logger.info(f"=== WEBHOOK REQUEST {request_id} RECEIVED ===")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        # Parse request with timeout protection
        try:
            # Set a reasonable timeout for request parsing
            request_body = await asyncio.wait_for(request.body(), timeout=5.0)
            event = json.loads(request_body.decode('utf-8'))
        except asyncio.TimeoutError:
            logger.error(f"Request parsing timeout for {request_id}")
            return {"text": "⏱️ Request timeout. Please try again."}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {request_id}: {e}")
            return {"text": "❌ Invalid request format. Please try again."}
        except Exception as e:
            logger.error(f"Request parsing error for {request_id}: {e}")
            return {"text": "❌ Request processing error. Please try again."}
        
        event_type = event.get("type", "UNKNOWN")
        logger.info(f"Event type: {event_type} for request {request_id}")
        
        # Handle different event types with ultra-fast responses
        if event_type == "ADDED_TO_SPACE":
            response = handle_added_to_space(event)
            response_time = time.time() - webhook_start_time
            logger.info(f"ADDED_TO_SPACE response time: {response_time:.2f}s")
            return response
            
        elif event_type == "REMOVED_FROM_SPACE":
            response = handle_removed_from_space(event)
            response_time = time.time() - webhook_start_time
            logger.info(f"REMOVED_FROM_SPACE response time: {response_time:.2f}s")
            return response
            
        elif event_type == "MESSAGE":
            message = event.get("message", {})
            text = message.get("text", "").strip()
            processed_files = event.get("processedFiles", [])
            space = event.get("space", {})
            user = event.get("user", {})
            
            space_name = space.get("name", "unknown_space")
            user_name = user.get("displayName", "User")
            
            logger.info(f"MESSAGE from {user_name} in {space_name}: '{text[:100]}...'")
            logger.info(f"Files: {len(processed_files)} for request {request_id}")
            
            # Quick validation and immediate response
            if not processed_files:
                attachments = message.get("attachment", [])
                if attachments:
                    response = {
                        "text": "❌ **File Processing Issue**\n\n"
                               "I couldn't access the attached files. This might be due to:\n"
                               "• File format not supported\n"
                               "• File size too large\n"
                               "• Temporary server issue\n\n"
                               "**Supported formats:** PDF, DOCX, DOC, TXT, RTF\n"
                               "**Max size:** 10MB per file\n\n"
                               "Please try uploading again or contact support."
                    }
                else:
                    response = {
                        "text": "👋 **Welcome to Quotation Generator!**\n\n"
                               "To create a quotation, please:\n"
                               "📎 **Attach your documents** to this message\n\n"
                               "**Supported formats:**\n"
                               "• PDF documents\n"
                               "• Word files (DOCX, DOC)\n"
                               "• Text files (TXT, RTF)\n\n"
                               "**Features:**\n"
                               "✅ AI-powered text extraction\n"
                               "✅ Professional Excel quotations\n"
                               "✅ Fast processing (2-5 minutes)\n"
                               "✅ Secure file handling\n\n"
                               "*Just attach your files and I'll get started!*"
                    }
                
                response_time = time.time() - webhook_start_time
                logger.info(f"No files response time: {response_time:.2f}s")
                return response
            
            # Generate processing ID and queue background task
            file_id = str(uuid.uuid4())
            download_url = f"https://quotebot-bluelabel.xoidlabs.com/api/download/{file_id}"
            
            # Initialize processing status immediately
            processing_status[file_id] = {
                "status": "queued", 
                "progress": "Queued for processing...",
                "created_at": webhook_start_time,
                "request_id": request_id,
                "user_name": user_name,
                "space_name": space_name,
                "file_count": len(processed_files),
                "download_url": download_url
            }
            
            # Queue background processing - this happens AFTER response
            background_tasks.add_task(
                process_documents_async,
                space_name,
                processed_files,
                file_id,
                user_name
            )
            
            # Return immediate acknowledgment response with download link
            response = {
                "text": f"🚀 **Processing Started!**\n\n"
                       f"📊 **Files received:** {len(processed_files)} document(s)\n"
                       f"⏱️ **Estimated time:** 2-5 minutes\n"
                       f"🤖 **AI Model:** Gemini 2.0 Flash\n\n"
                       f"**What happens next:**\n"
                       f"1. ⚡ Text extraction from files\n"
                       f"2. 🤖 AI quotation generation\n"
                       f"3. 📊 Excel spreadsheet creation\n"
                       f"4. 📥 Download link delivery\n\n"
                       f"**📥 [Download Link]({download_url})**\n"
                       f"*This link will be active once processing is complete*\n\n"
                       f"*I'll update you with progress messages!*"
            }
            
            response_time = time.time() - webhook_start_time
            logger.info(f"MESSAGE processing queued in {response_time:.2f}s for {request_id}")
            
            # Cleanup old files periodically
            if random.random() < 0.1:  # 10% chance
                background_tasks.add_task(cleanup_old_files)
            
            return response
            
        else:
            logger.warning(f"Unknown event type: {event_type} for request {request_id}")
            response = {
                "text": f"❓ **Unknown Event Type**\n\n"
                       f"Received: `{event_type}`\n"
                       f"I'm designed to handle file uploads and generate quotations.\n\n"
                       f"Please attach documents to get started!"
            }
            response_time = time.time() - webhook_start_time
            logger.info(f"Unknown event response time: {response_time:.2f}s")
            return response
        
    except Exception as e:
        error_time = time.time() - webhook_start_time
        logger.error(f"CRITICAL WEBHOOK ERROR for {request_id}: {str(e)}", exc_info=True)
        
        # Return minimal error response to stay under timeout
        response = {
            "text": f"❌ **Temporary System Error**\n\n"
                   f"Something went wrong on our end.\n"
                   f"Please try again in a moment.\n\n"
                   f"*If the issue persists, contact support*\n"
                   f"Error ID: `{request_id}`"
        }
        
        logger.info(f"Error response time: {error_time:.2f}s for {request_id}")
        return response

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

# Add this endpoint for manual status checking
@app.get("/status/{file_id}")
async def get_processing_status(file_id: str):
    """Get processing status"""
    if file_id not in processing_status:
        raise HTTPException(status_code=404, detail="File ID not found")
    
    status_info = processing_status[file_id]
    
    if status_info["status"] == "completed":
        return {
            "status": "completed",
            "message": "Processing completed successfully!",
            "download_url": status_info.get("download_url"),
            "file_id": file_id
        }
    elif status_info["status"] == "error":
        return {
            "status": "error",
            "message": status_info.get("message", "Unknown error occurred"),
            "file_id": file_id
        }
    else:
        return {
            "status": status_info["status"],
            "progress": status_info.get("progress", "Processing..."),
            "file_id": file_id,
            "download_url": status_info.get("download_url"),
            "elapsed_time": f"{time.time() - status_info.get('created_at', time.time()):.1f}s"
        }

# Add this endpoint to check if bot is working
@app.get("/ping")
async def ping():
    """Simple ping endpoint for testing"""
    return {
        "status": "pong",
        "timestamp": time.time(),
        "message": "Bot is alive and responding"
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
        "timestamp": time.time(),
        "temp_files_count": len(temp_files),
        "temp_files_ids": list(temp_files.keys()),
        "processing_jobs_count": len(processing_status),
        "processing_jobs": {
            pid: {
                "status": info["status"],
                "progress": info.get("progress", "N/A"),
                "age_seconds": time.time() - info.get("created_at", time.time())
            }
            for pid, info in processing_status.items()
        },
        "supported_extensions": ['.pdf', '.docx', '.doc', '.txt', '.rtf'],
        "server_info": {
            "python_version": "3.x",
            "fastapi_running": True
        }
    }

# Add endpoint to test chat webhook
@app.post("/test-webhook")
async def test_webhook():
    """Test endpoint to simulate a chat webhook"""
    test_event = {
        "type": "MESSAGE",
        "message": {
            "text": "test message",
            "attachment": []
        },
        "user": {
            "displayName": "Test User"
        },
        "space": {
            "name": "spaces/test_space"
        },
        "processedFiles": []
    }
    
    # Simulate the webhook call
    import json
    from fastapi import Request
    from unittest.mock import Mock
    
    # Create a mock request
    mock_request = Mock()
    mock_request.json = lambda: test_event
    mock_request.body = lambda: json.dumps(test_event).encode()
    mock_request.headers = {"content-type": "application/json"}
    
    try:
        result = await chat_webhook(mock_request, BackgroundTasks())
        return {
            "test_result": "success",
            "response": result,
            "message": "Webhook test completed successfully"
        }
    except Exception as e:
        return {
            "test_result": "error",
            "error": str(e),
            "message": "Webhook test failed"
        }

@app.get("/")
async def root():
    return {
        "message": "Document Quotation API with Google Chat Integration",
        "endpoints": {
            "upload": "/process-documents",
            "download": "/download/{file_id}",
            "webhook": "/chat-webhook",
            "status": "/status/{file_id}",
            "debug": "/debug",
            "health": "/health"
        }
    }