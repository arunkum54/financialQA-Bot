

# Run this cell first to install dependencies
# !pip install openai
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# !pip install transformers sentence-transformers pinecone-client pdfplumber pytesseract pdf2image pillow python-dateutil
# !pip install pytesseract pdfplumber pdf2image pillow
# !apt-get install poppler-utils -qq
# !pip install pdf2image pytesseract -q

import torchvision  
torchvision.disable_beta_transforms_warning()
# IMPORTING LIBRARIES
import pdfplumber
import pinecone
import torch
import re
import pytesseract
import os
import pandas as pd
from PIL import Image
from datetime import datetime
from dateutil.parser import parse
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI


# PLease use your API for the following , sorry for this but you can use my API just to test my code 
PINECONE_API_KEY = "pcsk_utwYR_C7q4xZxvhh32bgHuBHYVGDA4XYZPCr9dgpb64mHx3gKRvLLfArTTZnqL6uKNa8" #use your API
OPENAI_API_KEY = "pcsk_utwYR_C7q4xZxvhh32bgHuBHYVGDA4XYZPCr9dgpb64mHx3gKRvLLfArTTZnqL6uKNea8" # use your API
# Text extracting with OCR fallback to pdfplumber
def extract_text_from_pdf(pdf_path):
    extracted_text = []

    try:
        images = convert_from_path(pdf_path) # To Convert PDF to images

        for i, image in enumerate(images): # first Trying OCR extraction
            page_text = pytesseract.image_to_string(image)

            # if and only OCR extraction failed
            if not page_text.strip() or len(page_text) < 50:
                # then Fallback to pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    page = pdf.pages[i]
                    page_text = page.extract_text() or ""

            extracted_text.append(page_text)
    except Exception as e:
        print(f"Error in OCR processing: {e}")
        with pdfplumber.open(pdf_path) as pdf:
            extracted_text = [page.extract_text() or "" for page in pdf.pages]

    return "\n".join(extracted_text)

# Structuring extracted text into key-value pairs and tables
def structure_extracted_text(text):
    structured_data = []
    current_table = []
    in_table = False

    for line in text.split("\n"):
        # To Table detection
        if re.match(r"(\s{2,}.+){2,}", line) or "|" in line:
            if not in_table:
                in_table = True
                current_table = []
            # Cleaning and split table row
            row = [cell.strip() for cell in re.split(r"\s{2,}|\|", line) if cell.strip()]
            current_table.append(row)
        else:
            if in_table:
                structured_data.append({"table": current_table})
                current_table = []
                in_table = False

            # Key-value detection
            kv_match = re.match(r"^(.+?):\s*(.+)$", line)
            if kv_match:
                key = kv_match.group(1).strip()
                value = kv_match.group(2).strip()
                structured_data.append({key: value})
            elif line.strip():
                structured_data.append({"text": line.strip()})

    if in_table and current_table:
        structured_data.append({"table": current_table})

    return structured_data

# Defining multiple chunking strategies
def chunk_structured_data(structured_data, chunk_size=1000, overlap=296):
    # Converting structured data to text for chunking (unchanged)
    text_representation = []
    for item in structured_data:
        if "table" in item:
            table_text = "\n".join([", ".join(row) for row in item["table"]])
            text_representation.append(f"Table:\n{table_text}")
        elif "text" in item:
            text_representation.append(item["text"])
        else:
            for key, value in item.items():
                text_representation.append(f"{key}: {value}")
    full_text = "\n".join(text_representation)

    # Initializing text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "|"],
        length_function=len
    )

    # Split and add metadata
    chunks = text_splitter.split_text(full_text)
    return [{
        "text": chunk,
        "chunk_size": chunk_size,
        "overlap": overlap
    } for chunk in chunks]

from pinecone import Pinecone, ServerlessSpec


def init_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "testin" # Here i have testin index if you want to use then create testin or change the name as your preferens
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            )
        )
    return pc.Index(index_name)

def generate_response(query, index, embedder):
    client = OpenAI(api_key=OPENAI_API_KEY)
    query_embedding = embedder.encode(query).tolist()
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    
    context = [m.metadata['text'] for m in results.matches]
    prompt = f"""Answer the financial question using only this context:
    {chr(10).join(context)}

    Question: {query}
    Answer:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides accurate financial information based on the given context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].message.content
