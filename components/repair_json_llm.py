import re


def repair_json_with_llm(text, model):
    """Use LLM to detect and fix malformed JSON"""
    try:
        # Pre-process the text to handle unterminated strings
        def fix_unterminated_strings(text):
            lines = text.split('\n')
            fixed_lines = []
            in_string = False
            escape_next = False
            
            for line in lines:
                fixed_line = ""
                for char in line:
                    if escape_next:
                        fixed_line += char
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        fixed_line += char
                        continue
                    
                    if char == '"':
                        in_string = not in_string
                        fixed_line += char
                    else:
                        fixed_line += char
                
                # If we're still in a string at the end of the line, add a closing quote
                if in_string:
                    fixed_line += '"'
                    in_string = False
                
                fixed_lines.append(fixed_line)
            
            return '\n'.join(fixed_lines)
        
        # Fix unterminated strings first
        text = fix_unterminated_strings(text)
        
        # Create a prompt for the LLM to analyze and fix the JSON
        prompt = f"""You are a JSON repair expert. Analyze the following malformed JSON and fix it to be valid JSON.
        Return ONLY the fixed JSON without any explanation or additional text.
        
        Rules for fixing:
        1. Ensure all strings are properly terminated with double quotes
        2. Fix any unterminated strings by adding closing quotes
        3. Ensure all objects have proper structure
        4. Add missing commas between properties
        5. Convert empty values to null
        6. Quote any unquoted property names
        7. Fix any unbalanced brackets
        8. Remove any trailing commas
        9. Ensure all properties have values
        10. Handle any escape sequences properly
        
        Malformed JSON:
        {text}
        
        Fixed JSON:"""
        
        # Get response from LLM
        response = model.generate_content(prompt)
        if not response or not response.text:
            return None
            
        # Extract the fixed JSON from the response
        fixed_json = response.text.strip()
        
        # Remove any markdown code block markers if present
        fixed_json = re.sub(r'```json\s*', '', fixed_json)
        fixed_json = re.sub(r'```\s*$', '', fixed_json)
        
        # Ensure the JSON starts with { and ends with }
        if not fixed_json.startswith('{'):
            fixed_json = '{' + fixed_json
        if not fixed_json.endswith('}'):
            fixed_json = fixed_json + '}'
        
        # Verify the fixed JSON has balanced quotes
        quote_count = fixed_json.count('"')
        if quote_count % 2 != 0:
            # If we have an odd number of quotes, try to fix it
            fixed_json = fixed_json.replace('""', '"')  # Remove any double quotes
            if fixed_json.count('"') % 2 != 0:
                fixed_json += '"'  # Add a closing quote if still unbalanced
            
        return fixed_json
        
    except Exception as e:
        print(f"Error in LLM JSON repair: {str(e)}")
        return None