import json5
import demjson3
import ast
import re
import json
import google.generativeai as genai

from components.repair_json_llm import repair_json_with_llm

def validate_and_fix_data(data):
    """Validate and fix the quotation data structure"""
    if not isinstance(data, dict):
        return {
            "sections": [],
            "company_info": {"name": "", "address": "", "contact": ""},
            "client_info": {"name": "", "project_name": ""},
            "terms": {"payment_terms": "Net 30", "validity": "30 days"},
            "total_amount": 0
        }
    
    # Ensure all required fields exist
    data.setdefault("sections", [])
    data.setdefault("company_info", {"name": "", "address": "", "contact": ""})
    data.setdefault("client_info", {"name": "", "project_name": ""})
    data.setdefault("terms", {"payment_terms": "Net 30", "validity": "30 days"})
    data.setdefault("total_amount", 0)
    
    return data

def initialize_model():
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model

def extract_json_from_response(response_text):
    """Extract and parse JSON from the AI response with multiple fallback mechanisms"""
    print("\n=== Raw Response Text ===")
    print(response_text)
    print("=== End Raw Response Text ===\n")
    
    def repair_json(text):
        """Repair malformed JSON text"""
        # First, try to find the largest valid JSON object
        def find_largest_json(text):
            max_valid = ""
            current = ""
            depth = 0
            in_string = False
            escape_next = False
            
            for char in text:
                if escape_next:
                    current += char
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    current += char
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    current += char
                    continue
                
                if not in_string:
                    if char == '{':
                        if depth == 0:
                            current = char
                        else:
                            current += char
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        current += char
                        if depth == 0 and len(current) > len(max_valid):
                            max_valid = current
                    else:
                        current += char
                else:
                    current += char
            
            return max_valid if max_valid else text
        
        # Clean and normalize the text
        text = text.strip()
        
        # Remove any markdown code block markers
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
        # Remove any non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char in '\n\r\t')
        
        # Remove any BOM or special characters at the start
        text = text.lstrip('\ufeff')
        
        # Find the largest valid JSON object
        text = find_largest_json(text)
        
        return text
    
    # Initialize the model for JSON repair
    model = initialize_model()
    
    # First try to repair the JSON using the LLM
    repaired_text = repair_json(response_text)
    if repaired_text:
        llm_repaired_text = repair_json_with_llm(repaired_text, model)
        if llm_repaired_text:
            repaired_text = llm_repaired_text
    
    # Save the original and repaired text for debugging
    debug_info = {
        "original_text": response_text,
        "repaired_text": repaired_text,
        "errors": []
    }
    
    # Multiple parsing attempts with different strategies
    parsing_strategies = [
        # Strategy 1: Try demjson3 with strict=False and allow_omitted=True
        lambda: demjson3.decode(repaired_text, strict=False, allow_omitted=True),
        
        # Strategy 2: Try demjson3 with strict=False and allow_omitted=False
        lambda: demjson3.decode(repaired_text, strict=False, allow_omitted=False),
        
        # Strategy 3: Try json5 with strict=False
        lambda: json5.loads(repaired_text, strict=False),
        
        # Strategy 4: Try standard json with strict=False
        lambda: json.loads(repaired_text, strict=False),
        
        # Strategy 5: Try ast.literal_eval as last resort
        lambda: ast.literal_eval(repaired_text)
    ]
    
    # Try each strategy
    for i, strategy in enumerate(parsing_strategies, 1):
        try:
            print(f"\n=== Trying Parsing Strategy {i} ===")
            data = strategy()
            print("Strategy successful!")
            print(json.dumps(data, indent=2))
            
            # Validate and fix the data
            data = validate_and_fix_data(data)
            return data
            
        except Exception as e:
            error_msg = f"Strategy {i} failed: {str(e)}"
            print(error_msg)
            debug_info["errors"].append(error_msg)
            
            # If it's a demjson3 error about omitted elements, try to fix the structure
            if "Can not omit elements" in str(e) or "Unknown behavior" in str(e):
                try:
                    # Try to fix the structure by adding missing elements
                    fixed_text = repaired_text.replace('""', 'null')
                    fixed_text = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":null', fixed_text)
                    # Try with the most lenient settings
                    data = demjson3.decode(fixed_text, strict=False)
                    print("Fixed omitted elements and retried successfully!")
                    print(json.dumps(data, indent=2))
                    data = validate_and_fix_data(data)
                    return data
                except Exception as fix_error:
                    error_msg = f"Failed to fix omitted elements: {str(fix_error)}"
                    print(error_msg)
                    debug_info["errors"].append(error_msg)
            continue
    
    print("\n=== All Parsing Strategies Failed ===")
    print("Debug Information:")
    print(json.dumps(debug_info, indent=2))
    
    # Try to extract any valid JSON objects from the text
    try:
        # Find all potential JSON objects in the text
        json_objects = re.findall(r'\{[^{}]*\}', response_text)
        if json_objects:
            print("\nFound potential JSON objects in the text:")
            for i, obj in enumerate(json_objects, 1):
                print(f"\nObject {i}:")
                print(obj)
    except Exception as e:
        print(f"Error extracting potential JSON objects: {str(e)}")
    
    # Return a valid default structure as last resort
    return {
        "sections": [],
        "company_info": {"name": "", "address": "", "contact": ""},
        "client_info": {"name": "", "project_name": ""},
        "terms": {"payment_terms": "Net 30", "validity": "30 days"},
        "total_amount": 0,
        "_debug_info": debug_info  # Include debug information in the response
    }