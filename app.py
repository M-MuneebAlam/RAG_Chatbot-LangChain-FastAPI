import os
import shutil  # Added import
import tempfile  # Added import
from fastapi import FastAPI, File, UploadFile
from typing import List
from uuid import uuid4
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

# Initialize FastAPI app
app = FastAPI()

# Initialize Pinecone and LangChain Embeddings
pinecone_api_key = os.getenv("PINECONE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
index_name = "basic-rag-system"

# Set up Pinecone
pc = Pinecone(api_key=pinecone_api_key)
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
index = pc.Index(index_name)

# Set up Google Generative AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Set up Pinecone Vector Store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Set up LangChain Model (Google Gemini Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Utility functions for file processing
def load_documents_from_file(file: UploadFile) -> List[Document]:
    """
    Function to load a PDF or DOCX file using langchain loaders.
    """
    filename = file.filename
    
    # Create a temporary file path using tempfile for compatibility
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name  # Get the temp file path

        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        documents = []
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)  # Use the saved file for PDF loader
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(temp_file_path)  # Use the saved file for DOCX loader
        else:
            raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")
        
        documents.extend(loader.load())

    # Remove the temporary file after processing
    os.remove(temp_file_path)
    
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=40,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    return text_splitter.split_documents(documents)

def answer_to_user(query: str):
    # Vector Search
    vector_results = vector_store.similarity_search(query, k=2)
    final_answer = llm.invoke(f"ANSWER THIS USER QUERY: {query} USING THIS CONTEXT: {vector_results}")
    return final_answer.content

# API Endpoints

@app.post("/upload_documents/")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Endpoint to upload multiple documents (PDF/DOCX).
    Accepts a list of files to load and index them.
    """
    documents = []
    
    for file in files:
        documents.extend(load_documents_from_file(file))
    
    splits = split_documents(documents)
    
    # Generate UUIDs for each document chunk
    uuids = [str(uuid4()) for _ in range(len(splits))]
    
    # Store documents in Pinecone
    vector_store.add_documents(documents=splits, ids=uuids)
    
    return {"message": f"Uploaded and indexed {len(splits)} document chunks."}

@app.get("/query/")
async def query_system(query: str):
    """
    Endpoint to query the RAG system.
    Returns the generated response for the user's query.
    """
    answer = answer_to_user(query)
    return {"query": query, "answer": answer}
