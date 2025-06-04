import os
import pandas as pd
import tempfile



def generate_quotation_prompt(examples, input_docs_text, similar_examples=None, template=None):
    # Format examples
    examples_text = ""
    for i, example in enumerate(examples, 1):
        examples_text += f"\nExample {i}:\n"
        examples_text += f"Input: {example['input_text']}\n"
        examples_text += f"Output: {example['output_text']}\n"
    
    # Format similar examples if available
    similar_examples_text = ""
    if similar_examples:
        similar_examples_text = "\nSimilar Examples from Previous Cases:\n"
        for i, example in enumerate(similar_examples, 1):
            similar_examples_text += f"\nSimilar Example {i}:\n{example.page_content}\n"
    
    # Extract template placeholders if template is provided
    template_analysis = ""
    if template and hasattr(template, 'name') and template.name.endswith('.xlsx'):
        try:
            # Handle UploadedFile object
            if hasattr(template, 'getvalue'):
                # Create a temporary file to read the Excel
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                    tmp_file.write(template.getvalue())
                    tmp_file.flush()
                    df = pd.read_excel(tmp_file.name)
                    os.unlink(tmp_file.name)
            else:
                df = pd.read_excel(template)
            
            # Extract column names as placeholders
            placeholders = list(df.columns)
            template_analysis = f"\nTemplate Analysis:\nThe provided template contains the following columns: {', '.join(placeholders)}\n"
            template_analysis += "Ensure all these columns are properly filled with relevant information from the input documents.\n"
        except Exception as e:
            print(f"Error analyzing template: {str(e)}")
            template_analysis = "\nTemplate Analysis:\nCould not analyze template structure. Proceeding with default format.\n"
    
    # Chain of thought reasoning prompt
    chain_of_thought = f"""
Follow these steps carefully:

1. Template Analysis:
   - Identify all columns in the template
   - Understand the structure and formatting requirements
   - Note any specific sections or categories needed
   {template_analysis}

2. Content Extraction:
   - Extract relevant information from input documents
   - Match extracted information to template columns
   - Identify any missing information that needs to be inferred

3. Granular Decomposition:
   - Break down each major item into its smallest possible components
   - For AV equipment:
     * Main equipment (cameras, projectors, etc.)
     * Each camera needs: body, lens, battery, memory card, case
     * Each projector needs: unit, mount, cables, remote, spare lamp
     * Each screen needs: screen, frame, mounting hardware
     * Each audio system needs: mixer, speakers, microphones, stands, cables
     * Each lighting system needs: fixtures, controllers, cables, stands, gels
   - For labor:
     * Break down by role, time block, and specific task
     * Include setup, operation, and teardown for each item
     * Add prep time for each piece of equipment
     * Include travel time between locations
     * Add buffer time for contingencies
   - For rooms/areas:
     * Break down each room's requirements separately
     * Include all equipment needed for each room
     * Add setup and teardown labor for each room
     * Include room-specific consumables
   - For supporting items:
     * Include all necessary cables and connectors
     * Add mounting hardware and brackets
     * Include cases and protective equipment
     * Add spare parts and consumables
     * Include testing and calibration equipment

4. Validation:
   - Ensure all required sections are present
   - Verify that all template columns are filled
   - Check for consistency in formatting and style
   - Validate numerical calculations and units
   - Confirm all components are properly categorized
   - Verify minimum line item count (150-200 lines)
   - Ensure no major components are missing

5. Final Review:
   - Cross-reference with example quotations
   - Ensure all pricing follows established patterns
   - Verify source references are accurate
   - Check for any missing or incomplete information
   - Confirm proper grouping and sub-categorization
   - Verify template synchronization
   - Count total line items to ensure sufficient detail
"""

    json_structure = '''{
    "company_info": {
        "name": "string",
        "address": "string",
        "contact": "string"
    },
    "client_info": {
        "name": "string",
        "project_name": "string"
    },
    "sections": [
        {
            "name": "AV Equipment",
            "subsections": [
                {
                    "name": "Main Equipment",
                    "items": [
                        {
                            "description": "string",
                            "quantity": number,
                            "unit_price": number,
                            "total_price": number,
                            "source": {
                                "document": "string",
                                "line_reference": "string"
                            }
                        }
                    ]
                }
            ]
        }
    ],
    "terms": {
        "payment_terms": "string",
        "validity": "string"
    },
    "total_amount": number
}'''
    
    prompt = f"""You are a smart proposal quotation document creator. I will provide you with examples and new input documents.
    
Previous Examples:
{examples_text}

{similar_examples_text}

New Input Documents:
{input_docs_text}

{chain_of_thought}

Please analyze both the examples and new input documents to create a detailed quotation that follows the same style and structure as the examples, but with content specific to the new input.

The quotation should be highly detailed and granular, breaking down each major item into its smallest possible components. For example:

1. AV Equipment should be broken down into:
   - Main equipment (cameras, projectors, etc.)
   - Each camera needs: body, lens, battery, memory card, case
   - Each projector needs: unit, mount, cables, remote, spare lamp
   - Each screen needs: screen, frame, mounting hardware
   - Each audio system needs: mixer, speakers, microphones, stands, cables
   - Each lighting system needs: fixtures, controllers, cables, stands, gels
   - All necessary cables and connectors
   - All mounting hardware and brackets
   - All cases and protective equipment
   - All spare parts and consumables
   - All testing and calibration equipment

2. Labor should be detailed by:
   - Role (Camera Op, A1, Stagehand, etc.)
   - Time block (Load-in, Day 1, Day 2, etc.)
   - Specific task
   - Hours and rates
   - Setup time for each piece of equipment
   - Operation time for each piece of equipment
   - Teardown time for each piece of equipment
   - Prep time for each piece of equipment
   - Travel time between locations
   - Buffer time for contingencies
   - Per diem and meal buyouts

3. Each room or area should have its own detailed breakdown:
   - All required equipment
   - Setup and teardown labor
   - Supporting items
   - Consumables
   - Room-specific testing and calibration

4. Include all supporting services:
   - Freight and shipping
   - Equipment prep and testing
   - Warehouse pulls
   - Local union charges
   - Miscellaneous consumables
   - Insurance and permits
   - Quality control and testing

Format the response as a JSON object with the following structure:
{json_structure}

Ensure all prices are in USD and include appropriate units for quantities. Follow the pricing patterns and item categorization from the examples while adapting to the specific requirements in the new input documents. For each item, include a reference to the source document and line number where the information was derived from.

IMPORTANT: The quotation MUST contain at least 150-200 detailed line items. Break down every component into its smallest possible parts to achieve this level of detail. Do not combine items that should be separate line items.

{template_analysis}"""
    
    return prompt