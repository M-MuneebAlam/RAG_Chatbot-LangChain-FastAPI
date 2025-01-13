 # LangChain Basic RAG System

This repository contains a Jupyter Notebook for building a basic **Retrieval-Augmented Generation (RAG)** system using the LangChain framework. The notebook demonstrates how to integrate large language models with external data retrieval for enhanced query responses.

## Features

- **Pinecone Initialization**: Set up Pinecone as the vector database and configure API keys.
- **RAG Workflow with LangChain**:
  - Configure an embedding model for document representation.
  - Load and split documents into manageable chunks.
  - Embed and store documents in a vector store using Pinecone.
  - Set up a retrieval mechanism to fetch relevant context for queries.
- **Google Gemini Flash Model**: Integrate Google Gemini Flash as the LLM.
- **Retriever and LLM Combination**: Combine document retrieval with LLM capabilities for seamless integration.
- **Query the RAG System**: Execute queries to test and evaluate the RAG system's performance.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/langchain-basic-rag.git
   cd langchain-basic-rag
   ```

2. Open and run the notebook:
   ```bash
   jupyter notebook Project2_LangChain_Basic_RAG_System.ipynb
   ```
 