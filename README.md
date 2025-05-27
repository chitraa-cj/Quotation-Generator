# Few-Shot Learning Chat App

This is a Streamlit-based chat application that uses few-shot learning to generate responses based on example input-output pairs and document context. The app supports document uploads in various formats (PDF, DOCX, XLSX, TXT) and uses Google's Gemini API for generating responses.

## Features

- Upload and process multiple document types (PDF, DOCX, XLSX, TXT)
- Add example input-output pairs for few-shot learning
- Chat interface with context-aware responses
- Document context integration
- Chat history tracking

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
   You can get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Running the App

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Usage

1. Upload documents using the sidebar
2. Add example input-output pairs in the left column
3. Enter your input in the chat interface
4. Click "Generate Response" to get a response based on your examples and document context

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI Python SDK
- Other dependencies listed in requirements.txt

## Note

Make sure to keep your Google API key secure and never commit it to version control.
