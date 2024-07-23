import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    text = ""
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        text += page.extract_text()
    return text

# Function to get the top N most similar chunks
def get_similar_chunks(query, k=5):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

# Streamlit app
st.title("RAG Chatbot with GPT-Neo-1.3B")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Load PDF and extract text
    pdf_text = extract_text_from_pdf(uploaded_file)

    # Split text into chunks (optional, depending on the size of your PDF text)
    chunks = pdf_text.split('\n\n')

    # Load sentence transformer model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    user_input = st.text_input("Ask a question:")

    if user_input:
        # Get similar chunks from the PDF
        similar_chunks = get_similar_chunks(user_input)
        
        # Create context for the LLM
        context = " ".join(similar_chunks)
        
        # Generate response using the LLM
        input_ids = tokenizer.encode(context + user_input, return_tensors='pt')
        response_ids = model.generate(input_ids)
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        st.write(response)
