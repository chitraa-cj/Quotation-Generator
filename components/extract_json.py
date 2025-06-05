import json5
import json
import re
import google.generativeai as genai

from components.repair_json_llm import repair_json_with_llm

def clean_json_text(text):
    """Clean and normalize JSON text"""
    # Remove markdown code blocks
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    
    # Remove any non-printable characters except newlines, tabs, and carriage returns
    text = ''.join(char for char in text if char.isprintable() or char in '\n\r\t')
    
    # Remove BOM and special characters
    text = text.lstrip('\ufeff')
    
    # Fix double quotes
    text = re.sub(r'""([^"]*)""', r'"\1"', text)  # Fix double quoted strings
    text = re.sub(r'"\s*"', '"', text)  # Remove empty quotes
    
    return text.strip()

def fix_broken_string_values(text):
    """Fix broken string values in JSON text"""
    # Split into lines for processing
    lines = text.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Fix unquoted string values
        if ':' in line:
            key, value = line.split(':', 1)
            value = value.strip()
            
            # Clean up any double quotes
            value = re.sub(r'""([^"]*)""', r'"\1"', value)
            value = re.sub(r'"\s*"', '"', value)
            
            # If value is not already quoted and not a number/boolean/null
            if not (value.startswith('"') and value.endswith('"')) and \
               not (value.startswith("'") and value.endswith("'")) and \
               not value.isdigit() and \
               value.lower() not in ['true', 'false', 'null'] and \
               not value.startswith('{') and \
               not value.startswith('['):
                
                # Handle multiple values separated by commas
                if ',' in value:
                    values = [v.strip() for v in value.split(',')]
                    # Clean each value
                    values = [re.sub(r'^"|"$', '', v) for v in values]  # Remove any existing quotes
                    value = '", "'.join(values)
                    value = f'"{value}"'
                else:
                    # Remove any existing quotes before adding new ones
                    value = re.sub(r'^"|"$', '', value)
                    value = f'"{value}"'
            
            fixed_lines.append(f'{key}: {value}')
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_json_syntax(text):
    """Fix common JSON syntax issues"""
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

def find_json_boundaries(text):
    """Find the start and end of JSON content"""
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

def extract_json_content(text):
    """Extract JSON content from text"""
    start, end = find_json_boundaries(text)
    if start is None:
        return text
    return text[start:end]

def aggressive_json_repair(text):
    """Aggressively repair any text to make it valid JSON"""
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

def create_fallback_structure(text):
    """Create a valid JSON structure from any text"""
    print("Creating fallback JSON structure")
    
    # Try to extract any useful information from the text
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Look for patterns that might indicate data
    sections = []
    items = []
    
    for line in lines:
        # Look for price patterns
        price_match = re.search(r'\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', line)
        if price_match:
            price_str = price_match.group(1).replace(',', '')
            try:
                price = float(price_str)
                items.append({
                    "description": line,
                    "quantity": 1,
                    "unit_price": price,
                    "total_price": price
                })
            except ValueError:
                pass
    
    # If we found items, create a section for them
    if items:
        sections.append({
            "name": "Extracted Items",
            "items": items
        })
    
    # Calculate total
    total_amount = sum(item.get("total_price", 0) for section in sections for item in section.get("items", []))
    
    return {
        "sections": sections,
        "company_info": {
            "name": "Meeting Tomorrow",
            "address": "",
            "contact": ""
        },
        "client_info": {
            "name": "",
            "project_name": ""
        },
        "terms": {
            "payment_terms": "Net 30",
            "validity": "30 days"
        },
        "total_amount": total_amount
    }

def initialize_model():
    """Initialize the Gemini model"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        return model
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return None

def extract_json_from_response(response_text):
    """Extract and parse JSON from the AI response with guaranteed success"""
    print("\n=== Starting JSON Extraction ===")
    
    if not response_text or not response_text.strip():
        print("Empty response text, creating default structure")
        return create_fallback_structure("")
    
    # Try standard JSON parsing first
    try:
        # Look for JSON content
        json_content = extract_json_content(response_text)
        data = json.loads(json_content)
        print("Successfully parsed JSON with standard parser")
        return validate_and_fix_data(data)
    except Exception as e:
        print(f"Standard JSON parsing failed: {str(e)}")
    
    # Try json5 parsing
    try:
        json_content = extract_json_content(response_text)
        data = json5.loads(json_content)
        print("Successfully parsed JSON with json5")
        return validate_and_fix_data(data)
    except Exception as e:
        print(f"JSON5 parsing failed: {str(e)}")
    
    # Try aggressive repair
    try:
        repaired_text = aggressive_json_repair(response_text)
        print("Repaired text:", repaired_text[:200] + "..." if len(repaired_text) > 200 else repaired_text)
        
        # Try both json and json5
        for parser in [json.loads, json5.loads]:
            try:
                data = parser(repaired_text)
                print("Successfully parsed JSON with aggressive repair")
                return validate_and_fix_data(data)
            except Exception:
                continue
                
    except Exception as e:
        print(f"Aggressive repair failed: {str(e)}")
    
    # Try LLM repair if available
    try:
        model = initialize_model()
        if model:
            llm_repaired_text = repair_json_with_llm(response_text, model)
            if llm_repaired_text:
                for parser in [json.loads, json5.loads]:
                    try:
                        data = parser(llm_repaired_text)
                        print("Successfully parsed JSON with LLM repair")
                        return validate_and_fix_data(data)
                    except Exception:
                        continue
    except Exception as e:
        print(f"LLM repair failed: {str(e)}")
    
    # Last resort: create fallback structure
    print("All parsing methods failed, creating fallback structure")
    return validate_and_fix_data(create_fallback_structure(response_text))

def validate_and_fix_data(data):
    """Validate and fix the data structure"""
    if not isinstance(data, dict):
        print("Data is not a dictionary, creating default structure")
        return create_fallback_structure("")
    
    # Ensure all required fields exist with proper defaults
    if "sections" not in data:
        data["sections"] = []
    
    if "company_info" not in data:
        data["company_info"] = {"name": "", "address": "", "contact": ""}
    elif not isinstance(data["company_info"], dict):
        data["company_info"] = {"name": "", "address": "", "contact": ""}
    
    if "client_info" not in data:
        data["client_info"] = {"name": "", "project_name": ""}
    elif not isinstance(data["client_info"], dict):
        data["client_info"] = {"name": "", "project_name": ""}
    
    if "terms" not in data:
        data["terms"] = {"payment_terms": "Net 30", "validity": "30 days"}
    elif not isinstance(data["terms"], dict):
        data["terms"] = {"payment_terms": "Net 30", "validity": "30 days"}
    
    # Calculate total if not present or invalid
    if "total_amount" not in data or not isinstance(data["total_amount"], (int, float)):
        total = 0
        for section in data.get("sections", []):
            if isinstance(section, dict):
                for item in section.get("items", []):
                    if isinstance(item, dict):
                        total += item.get("total_price", 0)
        data["total_amount"] = total
    
    # Validate sections structure
    if not isinstance(data["sections"], list):
        data["sections"] = []
    
    # Clean up sections
    valid_sections = []
    for section in data["sections"]:
        if isinstance(section, dict) and "name" in section:
            if "items" not in section:
                section["items"] = []
            elif not isinstance(section["items"], list):
                section["items"] = []
            valid_sections.append(section)
    
    data["sections"] = valid_sections
    
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
    print("\nData validation results:")
    print(json.dumps(debug_info, indent=2))
    
    return data