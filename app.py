import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# Set your OpenAI API key
openai.api_key = st.secrets["openai"]["api_key"]

# Load SentenceTransformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    text = ""
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        text += page.extract_text()
    return text

# Function to create chunks with overlap
def create_chunks(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# Function to get the top N most similar chunks
def get_similar_chunks(query, k=5):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

@st.cache_data
def process_pdf(file):
    # Load PDF and extract text
    pdf_text = extract_text_from_pdf(file)
    
    # Create chunks
    chunks = create_chunks(pdf_text)
    
    # Generate embeddings
    embeddings = embedder.encode(chunks)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    
    return chunks, index

# Streamlit app
st.title("RAG Chatbot with OpenAI GPT-3.5 Turbo and FAISS")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Process PDF and cache results
    chunks, index = process_pdf(uploaded_file)
    
    user_input = st.text_input("Ask a question:")

    if user_input:
        # Get similar chunks from the PDF
        similar_chunks = get_similar_chunks(user_input)
        
        # Create context for the LLM
        context = " ".join(similar_chunks)
        
        # Generate response using the OpenAI API
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt=context + "\n\nQ: " + user_input + "\nA:",
            max_tokens=200
        )
        
        st.write(response.choices[0].text.strip())
