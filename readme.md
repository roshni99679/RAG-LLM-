# Gemini RAG App — README

This Streamlit app implements an **Explainable Retrieval-Augmented Generation (RAG)** system using **Google Gemini API**. It allows users to query documents, retrieves relevant chunks, generates answers, and records user feedback to improve future responses.  

---

## Table of Contents

1. [Setup and Installation](#setup-and-installation)  
2. [Code Overview](#code-overview)  
3. [Main Sections Explained](#main-sections-explained)  
4. [Feedback System](#feedback-system)  
5. [Customizing the App](#customizing-the-app)

---

## Setup and Installation

Install all required packages:

```bash
pip install -r requirements.txt
Make sure you have your Google Gemini API key in an env.py file:


GEMINI_API_KEY = "your_api_key_here"
Create a docs/ folder and place .pdf or .docx documents there for querying.

Code Overview
The app consists of:

Streamlit frontend: For user input and displaying results.

Document loaders: To read .pdf or .docx files.

Chunking: Splits large documents into smaller text chunks.

Embeddings and vector store: Converts text into embeddings and stores them in Chroma.

Feedback system: Logs thumbs-up / thumbs-down to adjust future retrieval confidence.

LLM Integration: Uses Gemini for generating final answers.

Main Sections Explained
1. Importing Libraries

import streamlit as st
import os
import numpy as np
import pandas as pd
from datetime import datetime
from env import GEMINI_API_KEY
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from sklearn.metrics.pairwise import cosine_similarity
streamlit: Web app framework.

os: File system operations.

numpy, pandas: Numeric and tabular data handling.

datetime: Timestamp for feedback logs.

env: Stores API key securely.

google.generativeai: Gemini API configuration.

langchain.document_loaders: Load .pdf and .docx documents.

langchain.text_splitter: Split text into manageable chunks.

langchain_google_genai: Embeddings and LLM integration for Google Gemini.

langchain.vectorstores: Chroma database for storing embeddings.

langchain.chains.RetrievalQA: Query-answer chain.

cosine_similarity: Compute similarity between embeddings.

2. Gemini API Configuration

genai.configure(api_key=GEMINI_API_KEY)
Configures Gemini API with your key.

3. Streamlit App Setup

st.set_page_config(page_title="Gemini RAG App", page_icon="🤖", layout="wide")
st.title("📘 Adaptive RAG with Gemini — Explainable Q&A + Feedback Learning")
Sets page title, icon, and layout.

Displays the main app title.

4. User Inputs

query = st.text_input("🔎 Enter your question:")
models = [m.name for m in list(genai.list_models()) if "generateContent" in m.supported_generation_methods]
model_name = st.selectbox("Select Gemini Model", [m for m in models if "gemini" in m.lower()])
docs_folder = "docs"
os.makedirs(docs_folder, exist_ok=True)
doc_files = [f for f in os.listdir(docs_folder) if f.endswith((".pdf", ".docx"))]
selected_doc = st.selectbox("Select a document from docs folder", doc_files)
answer_btn = st.button("⚡ Get Answer")
Text input: User enters a question.

Model selection: Shows all Gemini models capable of content generation.

Document selection: Reads files from docs/.

Answer button: Triggers document retrieval and answer generation.

5. Feedback File Setup

feedback_file = "feedback_log.csv"
if not os.path.exists(feedback_file):
    pd.DataFrame(columns=["timestamp", "query", "answer", "confidence", "feedback"]).to_csv(feedback_file, index=False)
else:
    feedback_data = pd.read_csv(feedback_file)
    if feedback_data.empty:
        feedback_data = pd.DataFrame(columns=["timestamp", "query", "answer", "confidence", "feedback"])
Creates feedback_log.csv if it doesn’t exist.

Loads existing feedback data if present.

6. Feedback Bias Function

def feedback_bias(query_embedding, feedback_df, embedding_model):
    ...
Computes bias adjustment for retrieval based on past feedback.

Positive feedback increases confidence; negative decreases it.

Uses cosine similarity between current query embedding and past queries.

7. Main Logic: Document Retrieval, Embeddings, and LLM

if answer_btn and selected_doc and query:
    ...
Loads selected document using appropriate loader.

Splits document into chunks for embeddings.

Embeds chunks using Google Gemini embeddings.

Stores embeddings in Chroma database for fast similarity search.

Computes feedback-adjusted confidence for retrieved chunks.

Uses ChatGoogleGenerativeAI for LLM answer generation.

8. Displaying Results

if "last_answer" in st.session_state and st.session_state["last_answer"]:
    ...
Shows the main answer in a green-highlighted box.

Shows retrieved chunks and debug info in a light-gray box.

Displays source documents with confidence scores.

9. Feedback Buttons

st.subheader("🗳️ Was this answer helpful?")
col1, col2 = st.columns(2)

def save_feedback(feedback_type):
    ...
Allows users to submit positive or negative feedback.

Feedback is logged in feedback_log.csv with timestamp, query, answer, and confidence.

Stored feedback is later used for bias adjustment in retrieval.

Feedback System
Feedback improves future retrieval by adjusting similarity scores:

Thumbs up → boosts similar queries.

Thumbs down → penalizes similar queries.

Stored in CSV for persistent learning.

Customizing the App
Documents: Add .pdf or .docx files to docs/.

Models: Select different Gemini models in the dropdown.

Chunk size: Adjust RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) for larger or smaller text chunks.

Styling: Modify green/highlight boxes in st.markdown for different UI appearance.