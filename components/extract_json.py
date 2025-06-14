import json5
import json
import re
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
import google.generativeai as genai
from components.repair_json_llm import repair_json_with_llm
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ExtractionStrategy(Enum):
    DIRECT_PARSE = "direct_parse"
    MARKDOWN_BLOCK = "markdown_block"
    PARTIAL_REPAIR = "partial_repair"
    LLM_REPAIR = "llm_repair"
    AGGRESSIVE_REPAIR = "aggressive_repair"
    STRUCTURED_PARSE = "structured_parse"
    FALLBACK = "fallback"

@dataclass
class ExtractionResult:
    data: Dict[str, Any]
    strategy_used: ExtractionStrategy
    needs_review: bool
    original_text: str
    error_message: Optional[str] = None

class QuotationJSONGenerator:
    def __init__(self, model=None):
        self.model = model or initialize_model()
        self.prompt_templates = {
            "structured": """
            Convert the following text into a valid JSON object. Follow these requirements:
            1. All property names must be in quotes
            2. All string values must be in quotes
            3. Use proper JSON syntax for arrays and objects
            4. Include all relevant information from the text
            
            Text to convert:
            {text}
            
            Expected format:
            {{
                "property": "value",
                "array": ["item1", "item2"],
                "nested": {{
                    "key": "value"
                }}
            }}
            """,
            
            "example_based": """
            Here's an example of how to convert text to JSON:
            
            Input text:
            Company: Acme Corp
            Address: 123 Main St
            Projects: Web Dev, Mobile App
            
            Expected output:
            {{
                "company": "Acme Corp",
                "address": "123 Main St",
                "projects": ["Web Dev", "Mobile App"]
            }}
            
            Now convert this text:
            {text}
            """,
            
            "simple": """
            Convert this text to JSON:
            {text}
            """
        }
    
    def generate_json(self, text: str, max_retries: int = 3) -> Dict[str, Any]:
        for attempt in range(max_retries):
            try:
                # Try different prompt strategies
                if attempt == 0:
                    prompt = self.prompt_templates["structured"].format(text=text)
                elif attempt == 1:
                    prompt = self.prompt_templates["example_based"].format(text=text)
                else:
                    prompt = self.prompt_templates["simple"].format(text=text)
                
                response = self.model.generate_content(prompt)
                result = extract_json_from_response(response.text)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                continue
        
        return create_fallback_structure(text)

class RobustJSONExtractor:
    def __init__(self, model=None):
        self.json_generator = QuotationJSONGenerator(model)
        self.model = model
    
    def extract(self, text: str) -> ExtractionResult:
        """Try multiple strategies to extract JSON from text"""
        if not text:
            return ExtractionResult(
                data={},
                strategy_used=ExtractionStrategy.FALLBACK,
                needs_review=False,
                original_text=text,
                error_message="Empty input text"
            )
        
        # Strategy 1: Direct JSON parsing
        try:
            data = json.loads(text)
            return ExtractionResult(
                data=data,
                strategy_used=ExtractionStrategy.DIRECT_PARSE,
                needs_review=False,
                original_text=text
            )
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Markdown code block extraction
        try:
            cleaned_text = clean_json_text(text)
            data = json.loads(cleaned_text)
            return ExtractionResult(
                data=data,
                strategy_used=ExtractionStrategy.MARKDOWN_BLOCK,
                needs_review=False,
                original_text=text
            )
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Partial JSON repair
        try:
            repaired_text = fix_json_syntax(text)
            data = json.loads(repaired_text)
            return ExtractionResult(
                data=data,
                strategy_used=ExtractionStrategy.PARTIAL_REPAIR,
                needs_review=False,
                original_text=text
            )
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: LLM-based repair
        try:
            if self.model:
                repaired_text = repair_json_with_llm(text, self.model)
                data = json.loads(repaired_text)
                return ExtractionResult(
                    data=data,
                    strategy_used=ExtractionStrategy.LLM_REPAIR,
                    needs_review=True,
                    original_text=text
                )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM repair failed: {str(e)}")
        
        # Strategy 5: Aggressive text repair
        try:
            repaired_text = aggressive_json_repair(text)
            data = json.loads(repaired_text)
            return ExtractionResult(
                data=data,
                strategy_used=ExtractionStrategy.AGGRESSIVE_REPAIR,
                needs_review=True,
                original_text=text
            )
        except json.JSONDecodeError:
            pass
        
        # Strategy 6: Structured text parsing
        try:
            data = self._parse_structured_text(text)
            return ExtractionResult(
                data=data,
                strategy_used=ExtractionStrategy.STRUCTURED_PARSE,
                needs_review=True,
                original_text=text
            )
        except Exception as e:
            logger.warning(f"Structured parsing failed: {str(e)}")
        
        # Strategy 7: Smart fallback
        fallback_data = create_fallback_structure(text)
        return ExtractionResult(
            data=fallback_data,
            strategy_used=ExtractionStrategy.FALLBACK,
            needs_review=True,
            original_text=text,
            error_message="All extraction strategies failed, using fallback structure"
        )
    
    def _parse_structured_text(self, text: str) -> Dict[str, Any]:
        """Parse text that appears to be structured but not in JSON format"""
        result = {}
        current_section = None
        current_items = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            if line.endswith(':'):
                if current_section and current_items:
                    result[current_section] = current_items
                current_section = line[:-1].lower().replace(' ', '_')
                current_items = []
                continue
            
            # Check for key-value pairs
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Handle multiple values
                if ',' in value:
                    values = [v.strip() for v in value.split(',')]
                    result[key] = values
                else:
                    result[key] = value
            else:
                current_items.append(line)
        
        # Add the last section
        if current_section and current_items:
            result[current_section] = current_items
        
        return result

def clean_json_text(text: str) -> str:
    """Clean and normalize JSON text with robust error handling"""
    if not text:
        return "{}"
    
    try:
        # Remove markdown code blocks
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
        # Remove any non-printable characters except newlines, tabs, and carriage returns
        text = ''.join(char for char in text if char.isprintable() or char in '\n\r\t')
        
        # Remove BOM and special characters
        text = text.lstrip('\ufeff')
        
        # Fix common JSON formatting issues
        text = re.sub(r'""([^"]*)""', r'"\1"', text)  # Fix double quoted strings
        text = re.sub(r'"\s*"', '"', text)  # Remove empty quotes
        text = re.sub(r',\s*}', '}', text)  # Remove trailing commas
        text = re.sub(r',\s*]', ']', text)  # Remove trailing commas in arrays
        
        # Fix unquoted property names
        text = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', text)
        
        # Fix unquoted string values
        text = re.sub(r':\s*([^"\'{\[\]}\s][^,}\]]*?)([,}\]])', r': "\1"\2', text)
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning JSON text: {str(e)}")
        return "{}"

def fix_broken_string_values(text: str) -> str:
    """Fix broken string values in JSON text with robust error handling"""
    try:
        lines = text.split('\n')
        fixed_lines = []
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                value = value.strip()
                
                # Clean up any double quotes
                value = re.sub(r'""([^"]*)""', r'"\1"', value)
                value = re.sub(r'"\s*"', '"', value)
                
                # Handle different value types
                if not (value.startswith('"') and value.endswith('"')) and \
                   not (value.startswith("'") and value.endswith("'")) and \
                   not value.isdigit() and \
                   value.lower() not in ['true', 'false', 'null'] and \
                   not value.startswith('{') and \
                   not value.startswith('['):
                    
                    # Handle multiple values
                    if ',' in value:
                        values = [v.strip() for v in value.split(',')]
                        values = [re.sub(r'^"|"$', '', v) for v in values]
                        value = '", "'.join(values)
                        value = f'"{value}"'
                    else:
                        value = re.sub(r'^"|"$', '', value)
                        value = f'"{value}"'
                
                fixed_lines.append(f'{key}: {value}')
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    except Exception as e:
        logger.error(f"Error fixing string values: {str(e)}")
        return text

def fix_json_syntax(text: str) -> str:
    """Fix common JSON syntax issues with robust error handling"""
    try:
        # Fix missing quotes around property names
        text = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', text)
        
        # Fix missing quotes around string values
        text = re.sub(r':\s*([^"\'{\[\]}\s][^,}\]]*?)([,}\]])', r': "\1"\2', text)
        
        # Fix trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Fix missing commas between properties
        text = re.sub(r'"\s*}\s*"', '", "', text)
        
        # Fix multiple values in a single field
        text = re.sub(r':\s*([^"\'{\[\]}\s][^,}\]]*?),\s*([^"\'{\[\]}\s][^,}\]]*?)([,}\]])', r': ["\1", "\2"]\3', text)
        
        # Fix double quotes
        text = re.sub(r'""([^"]*)""', r'"\1"', text)
        text = re.sub(r'"\s*"', '"', text)
        
        # Fix unterminated strings
        text = re.sub(r':\s*"([^"]*?)(?=\s*[,}\]])', r':"\1"', text)
        
        return text
    except Exception as e:
        logger.error(f"Error fixing JSON syntax: {str(e)}")
        return text

def find_json_boundaries(text: str) -> tuple[Optional[int], Optional[int]]:
    """Find the start and end of JSON content with robust error handling"""
    try:
        # Look for opening brace
        start_match = re.search(r'{', text)
        if not start_match:
            return None, None
        
        start_pos = start_match.start()
        
        # Find matching closing brace
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start_pos:], start_pos):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return start_pos, i + 1
        
        return start_pos, len(text)
    except Exception as e:
        logger.error(f"Error finding JSON boundaries: {str(e)}")
        return None, None

def extract_json_content(text: str) -> str:
    """Extract JSON content from text with robust error handling"""
    try:
        start, end = find_json_boundaries(text)
        if start is None:
            return text
        return text[start:end]
    except Exception as e:
        logger.error(f"Error extracting JSON content: {str(e)}")
        return text

def aggressive_json_repair(text: str) -> str:
    """Aggressively repair any text to make it valid JSON with robust error handling"""
    try:
        # Clean the text first
        text = clean_json_text(text)
        
        # Extract JSON content
        json_content = extract_json_content(text)
        
        # Fix broken string values first
        json_content = fix_broken_string_values(json_content)
        
        # Fix syntax issues
        json_content = fix_json_syntax(json_content)
        
        # Ensure proper brace matching
        open_braces = json_content.count('{')
        close_braces = json_content.count('}')
        
        if open_braces > close_braces:
            json_content += '}' * (open_braces - close_braces)
        elif close_braces > open_braces:
            # Remove extra closing braces
            extra_braces = close_braces - open_braces
            for _ in range(extra_braces):
                json_content = json_content.rsplit('}', 1)[0] + json_content.rsplit('}', 1)[1]
        
        # Ensure proper array bracket matching
        open_brackets = json_content.count('[')
        close_brackets = json_content.count(']')
        
        if open_brackets > close_brackets:
            json_content += ']' * (open_brackets - close_brackets)
        elif close_brackets > open_brackets:
            extra_brackets = close_brackets - open_brackets
            for _ in range(extra_brackets):
                json_content = json_content.rsplit(']', 1)[0] + json_content.rsplit(']', 1)[1]
        
        # Fix any remaining unquoted values in arrays
        json_content = re.sub(r'\[([^"\'{\[\]}\s][^,}\]]*?)\]', r'["\1"]', json_content)
        
        # Final cleanup of any remaining double quotes
        json_content = re.sub(r'""([^"]*)""', r'"\1"', json_content)
        json_content = re.sub(r'"\s*"', '"', json_content)
        
        return json_content
    except Exception as e:
        logger.error(f"Error in aggressive JSON repair: {str(e)}")
        return "{}"

def create_fallback_structure(text: str) -> Dict[str, Any]:
    """Create a valid JSON structure from any text with improved handling"""
    try:
        logger.info("Creating fallback JSON structure")
        
        # Try to extract any useful information from the text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Look for patterns that might indicate data
        sections = []
        items = []
        current_section = None
        
        # Common patterns to ignore (addresses, dates, etc.)
        ignore_patterns = [
            r'^\d+\s+[A-Za-z]+,\s*[A-Z]{2}\s+\d{5}$',  # Address pattern
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # Date pattern
            r'^[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,\s*\d{4}$',  # Written date
            r'^[A-Za-z]+\s+\d{1,2}$',  # Month and day
            r'^\d{1,2}:\d{2}\s*(?:AM|PM)?$',  # Time
            r'^[A-Za-z]+\.com$',  # Website
            r'^[A-Za-z]+@[A-Za-z]+\.[A-Za-z]+$',  # Email
        ]
        
        # Extract company and client info first
        company_info = {"name": "", "address": "", "contact": ""}
        client_info = {"name": "", "project_name": ""}
        
        # First pass: extract company and client info
        for line in lines:
            # Look for company/client information
            if "company" in line.lower() or "organization" in line.lower():
                company_info["name"] = line.strip()
            elif "address" in line.lower():
                company_info["address"] = line.strip()
            elif "contact" in line.lower() or "phone" in line.lower():
                company_info["contact"] = line.strip()
            elif "client" in line.lower():
                client_info["name"] = line.strip()
            elif "project" in line.lower():
                client_info["project_name"] = line.strip()
        
        # Second pass: extract sections and items
        for line in lines:
            # Skip lines matching ignore patterns
            if any(re.match(pattern, line) for pattern in ignore_patterns):
                continue
            
            # Look for section headers with improved pattern matching
            if (re.match(r'^[A-Z][a-zA-Z\s]+:', line) or 
                re.match(r'^[A-Z][a-zA-Z\s]+$', line) or
                re.match(r'^[A-Z][a-zA-Z\s]+Project', line) or
                re.match(r'^[A-Z][a-zA-Z\s]+Brief', line) or
                re.match(r'^[A-Z][a-zA-Z\s]+Date', line) or
                re.match(r'^[A-Z][a-zA-Z\s]+Due', line)):
                
                if current_section and items:
                    current_section["items"] = items
                    sections.append(current_section)
                    items = []
                
                # Clean up section name
                section_name = line.strip(':')
                section_name = re.sub(r'\s+', ' ', section_name)
                section_name = section_name.replace('Project Brief', '')
                section_name = section_name.replace('Issue Date', '')
                section_name = section_name.replace('Due Date', '')
                section_name = section_name.strip()
                
                # Skip empty section names
                if not section_name:
                    continue
                    
                current_section = {"name": section_name, "items": []}
                continue
            
            # Look for bullet points or numbered items
            if re.match(r'^[•●◆■]', line) or re.match(r'^\d+[\.\)]', line) or re.match(r'^[A-Z][a-z]+\s+\d+', line):
                # Extract description
                description = re.sub(r'^[•●◆■]\s*', '', line)
                description = re.sub(r'^\d+[\.\)]\s*', '', description)
                description = description.strip()
                
                # Look for price in the description
                price_match = re.search(r'(?:\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)|(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:B|M|K|Billion|Million|Thousand)?)', description)
                if price_match:
                    price_str = price_match.group(1) or price_match.group(2)
                    if price_str:
                        try:
                            # Clean up price string
                            price_str = price_str.replace(',', '')
                            
                            # Skip if this looks like an address or date
                            if re.match(r'^\d{5}$', price_str) or re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', price_str):
                                continue
                                
                            # Handle unit multipliers
                            multiplier = 1
                            if 'B' in description or 'Billion' in description:
                                multiplier = 1000000000
                            elif 'M' in description or 'Million' in description:
                                multiplier = 1000000
                            elif 'K' in description or 'Thousand' in description:
                                multiplier = 1000
                            
                            price = float(price_str) * multiplier
                            
                            # Remove price from description
                            description = re.sub(r'\$?\s*\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:B|M|K|Billion|Million|Thousand)?', '', description)
                            description = description.strip()
                            
                            # Skip if description is empty or just punctuation
                            if not description or description.isspace() or all(c in '.,;:!?' for c in description):
                                continue
                                
                            items.append({
                                "description": description or "Unnamed Item",
                                "quantity": 1,
                                "unit_price": price,
                                "total_price": price
                            })
                        except ValueError:
                            pass
                else:
                    # Add item without price
                    items.append({
                        "description": description,
                        "quantity": 1,
                        "unit_price": 0,
                        "total_price": 0
                    })
        
        # Add the last section if it exists
        if current_section and items:
            current_section["items"] = items
            sections.append(current_section)
        
        # If no sections were found but we have items, create a default section
        if not sections and items:
            sections.append({
                "name": "Extracted Items",
                "items": items
            })
        
        # Calculate total
        total_amount = sum(item.get("total_price", 0) for section in sections for item in section.get("items", []))
        
        return {
            "sections": sections,
            "company_info": company_info,
            "client_info": client_info,
            "terms": {
                "payment_terms": "Net 30",
                "validity": "30 days"
            },
            "total_amount": total_amount
        }
    except Exception as e:
        logger.error(f"Error creating fallback structure: {str(e)}")
        return {
            "sections": [],
            "company_info": {"name": "", "address": "", "contact": ""},
            "client_info": {"name": "", "project_name": ""},
            "terms": {"payment_terms": "Net 30", "validity": "30 days"},
            "total_amount": 0
        }

def initialize_model() -> Optional[Any]:
    """Initialize the Gemini model with robust error handling"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return None

def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Extract and parse JSON from the AI response with guaranteed success"""
    logger.info("\n=== Starting JSON Extraction ===")
    
    if not response_text or not response_text.strip():
        logger.warning("Empty response text, creating default structure")
        return create_fallback_structure("")
    
    # Try standard JSON parsing first
    try:
        json_content = extract_json_content(response_text)
        data = json.loads(json_content)
        logger.info("Successfully parsed JSON with standard parser")
        return validate_and_fix_data(data)
    except Exception as e:
        logger.warning(f"Standard JSON parsing failed: {str(e)}")
    
    # Try json5 parsing
    try:
        json_content = extract_json_content(response_text)
        data = json5.loads(json_content)
        logger.info("Successfully parsed JSON with json5")
        return validate_and_fix_data(data)
    except Exception as e:
        logger.warning(f"JSON5 parsing failed: {str(e)}")
    
    # Try aggressive repair
    try:
        repaired_text = aggressive_json_repair(response_text)
        logger.info("Repaired text: " + (repaired_text[:200] + "..." if len(repaired_text) > 200 else repaired_text))
        
        # Try both json and json5
        for parser in [json.loads, json5.loads]:
            try:
                data = parser(repaired_text)
                logger.info("Successfully parsed JSON with aggressive repair")
                return validate_and_fix_data(data)
            except Exception:
                continue
                
    except Exception as e:
        logger.warning(f"Aggressive repair failed: {str(e)}")
    
    # Try LLM repair if available
    try:
        model = initialize_model()
        if model:
            llm_repaired_text = repair_json_with_llm(response_text, model)
            if llm_repaired_text:
                for parser in [json.loads, json5.loads]:
                    try:
                        data = parser(llm_repaired_text)
                        logger.info("Successfully parsed JSON with LLM repair")
                        return validate_and_fix_data(data)
                    except Exception:
                        continue
    except Exception as e:
        logger.warning(f"LLM repair failed: {str(e)}")
    
    # Last resort: create fallback structure
    logger.warning("All parsing methods failed, creating fallback structure")
    return validate_and_fix_data(create_fallback_structure(response_text))

def validate_and_fix_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fix quotation data structure with robust error handling"""
    try:
        if not isinstance(data, dict):
            data = {"sections": []}
        
        # Ensure required fields exist
        if "sections" not in data:
            data["sections"] = []
        
        if "company_info" not in data:
            data["company_info"] = {
                "name": "",
                "address": "",
                "contact": ""
            }
        
        if "client_info" not in data:
            data["client_info"] = {
                "name": "",
                "project_name": ""
            }
        
        if "terms" not in data:
            data["terms"] = {
                "payment_terms": "Net 30",
                "validity": "30 days"
            }
        
        # Validate and fix sections
        valid_sections = []
        total_amount = 0
        
        for section in data.get("sections", []):
            if not isinstance(section, dict):
                continue
                
            # Ensure section has required fields
            if "name" not in section:
                section["name"] = "Unnamed Section"
                
            if "items" not in section:
                section["items"] = []
                
            # Validate and fix items
            valid_items = []
            section_total = 0
            
            for item in section.get("items", []):
                if not isinstance(item, dict):
                    continue
                    
                # Ensure item has required fields
                if "description" not in item:
                    item["description"] = "Unnamed Item"
                    
                # Convert quantity to float
                try:
                    item["quantity"] = float(item.get("quantity", 1))
                except (ValueError, TypeError):
                    item["quantity"] = 1
                    
                # Convert unit price to float
                try:
                    item["unit_price"] = float(item.get("unit_price", 0))
                except (ValueError, TypeError):
                    item["unit_price"] = 0
                    
                # Calculate total price
                item["total_price"] = item["quantity"] * item["unit_price"]
                section_total += item["total_price"]
                
                valid_items.append(item)
                
            section["items"] = valid_items
            section["total"] = section_total
            total_amount += section_total
            
            valid_sections.append(section)
        
        # Update total amount
        data["total_amount"] = total_amount
        
        # Print debug info
        debug_info = {
            "has_sections": bool(data.get("sections")),
            "section_count": len(data.get("sections", [])),
            "has_company_info": bool(data.get("company_info")),
            "has_client_info": bool(data.get("client_info")),
            "has_terms": bool(data.get("terms")),
            "has_total": "total_amount" in data,
            "total_amount": data.get("total_amount", 0)
        }
        logger.info("\nData validation results:")
        logger.info(json.dumps(debug_info, indent=2))
        
        return data
    except Exception as e:
        logger.error(f"Error in validate_and_fix_data: {str(e)}")
        return create_fallback_structure("")