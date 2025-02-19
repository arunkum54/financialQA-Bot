# Live : https://financialapp-bot.streamlit.app/
""Part 1""

Overview
This Google Colab notebook is designed to process financial statements from PDF documents, extract and structure the text, and generate accurate answers to financial queries using advanced natural language processing (NLP) techniques. The pipeline combines Optical Character Recognition (OCR), text structuring, chunking, vector embeddings, and OpenAI's GPT-3.5-turbo model to provide a robust solution for financial document analysis.

Features

1.Text Extraction: Uses OCR (Tesseract) and PDF text extraction (pdfplumber) for reliable text retrieval from PDFs, including scanned documents.
2.Text Structuring: Detects and organizes text into key-value pairs, tables, and unstructured text for easier analysis.
3.Text Chunking: Splits text into smaller, context-rich chunks for efficient processing by language models.
4.Vector Embeddings: Converts text chunks into numerical representations using a pre-trained sentence transformer model (all-MiniLM-L6-v2).
5.Vector Storage: Stores embeddings in Pinecone, a vector database, for fast and accurate retrieval.
6.Query Handling: Generates precise answers to financial queries using OpenAI's GPT-3.5-turbo model.
7.Output Storage: Saves questions and answers in a JSON file for future reference.

Workflow

Text Extraction: Extracts text from PDFs using OCR and PDF text extraction.
Text Structuring: Organizes extracted text into key-value pairs, tables, and unstructured text.
Text Chunking: Splits structured text into smaller chunks for better processing by language models.
Vector Embeddings: Converts text chunks into embeddings and stores them in Pinecone.
Query Processing: Retrieves relevant text chunks from Pinecone and generates answers using OpenAI's GPT-3.5-turbo.
Output Storage: Saves the generated questions and answers in a JSON file.

Installation

Open the Google Colab notebook.

Run the first cell to install all required dependencies:
!pip install openai
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers sentence-transformers pinecone-client pdfplumber pytesseract pdf2image pillow python-dateutil
!pip install pytesseract pdfplumber pdf2image pillow
!apt-get install poppler-utils -qq
!pip install pdf2image pytesseract -q

Usage
Upload PDF: Upload your financial statement PDF to the Colab environment.
Update PDF Path: Update the pdf_path variable in the notebook with the path to your PDF file.
Run the Notebook: Execute all cells in the notebook sequentially.

View Output:
Extracted text and structured data will be printed in the notebook.
Generated questions and answers will be saved in a JSON file (Tested_question_answers.json).

Configuration

Pinecone API Key: Update the pinecone_api_key variable with your Pinecone API key.
OpenAI API Key: Update the openai_api_key variable with your OpenAI API key.
Chunk Size and Overlap: Adjust the chunk_size and overlap parameters in the chunk_structured_data function as needed.

Output:
Structured Data: Text is structured into key-value pairs, tables, and unstructured text.
Chunks: Text is split into smaller chunks for efficient processing by language models.
QA Pairs: Questions and answers are saved in a JSON file (Tested_question_answers.json).

Dependencies
Python 3.x
Libraries: 
pdfplumber, pytesseract, pdf2image, sentence-transformers, pinecone-client, openai, langchain



""Part 2 ""


Project Overview
This project involves a Streamlit app that can be run locally or deployed on an AWS EC2 instance using Docker. The app is containerized for easy deployment and scalability.

File Structure
Copy
part2/
├── app.py                # Main Streamlit application file
├── backend.py            # Backend logic for the app
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration file
└── README.md             # Project documentation



Run Locally
Clone the repository:
git clone https://github.com/arunkum54/financialQA-Bot.git
cd part2


Install dependencies:
pip install -r requirements.txt
Run the app:
streamlit run app.py

Open your browser and navigate to:
http://localhost:8501
