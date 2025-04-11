# Project Structure:
#
# rag-project/
# ├── app/
# │   ├── __init__.py
# │   ├── main.py
# │   ├── services/
# │   │   ├── __init__.py
# │   │   ├── document_store.py
# │   │   └── llm.py
# ├── data/
# │   ├── raw/            # Place your documents here
# │   └── processed/      # FAISS index will be stored here
# ├── scripts/
# │   └── ingest.py
# ├── Dockerfile
# ├── docker-compose.yml
# ├── requirements.txt
# └── README.md

#=============================================================================
# app/__init__.py
#=============================================================================
# Empty file to make the directory a Python package

#=============================================================================
# app/main.py
#=============================================================================
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time

from app.services.document_store import DocumentStore
from app.services.llm import LLMService

# Initialize FastAPI app
app = FastAPI(
    title="RAG Question Answering API",
    description="A Retrieval-Augmented Generation system for domain-specific question answering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    include_sources: Optional[bool] = False

class DocumentResponse(BaseModel):
    id: int
    content: str
    score: float

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[List[DocumentResponse]] = None
    processing_time: float

# Initialize services
document_store = None
llm_service = None

# Dependency for initializing services
def get_services():
    global document_store, llm_service
    
    if document_store is None:
        # Load FAISS index from saved file or create new
        index_path = os.getenv("FAISS_INDEX_PATH", "data/processed/faiss_index.bin")
        document_store = DocumentStore(index_path=index_path)
        
    if llm_service is None:
        # Initialize LLM service
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        llm_service = LLMService(model_name=model_name)
        
    return {"document_store": document_store, "llm_service": llm_service}

# API endpoints
@app.post("/api/answer", response_model=AnswerResponse)
async def answer_question(
    request: QuestionRequest,
    services: Dict = Depends(get_services)
):
    start_time = time.time()
    
    # Retrieve relevant documents
    document_store = services["document_store"]
    retrieved_docs = document_store.search(request.question, top_k=request.top_k)
    
    if not retrieved_docs:
        raise HTTPException(status_code=404, detail="No relevant context found")
    
    # Generate answer using LLM
    llm_service = services["llm_service"]
    if request.include_sources:
        result = llm_service.generate_answer_with_sources(request.question, retrieved_docs)
        answer = result["answer"]
        sources = result["sources"]
    else:
        answer = llm_service.generate_answer(request.question, retrieved_docs)
        sources = None
    
    processing_time = time.time() - start_time
    
    # Prepare response
    response = {
        "question": request.question,
        "answer": answer,
        "processing_time": processing_time
    }
    
    if request.include_sources:
        response["sources"] = [
            DocumentResponse(id=doc["id"], content=doc["content"][:200], score=doc["score"])
            for doc in retrieved_docs
        ]
    
    return response

# Document ingestion endpoint
class IngestRequest(BaseModel):
    documents: List[str]

class IngestResponse(BaseModel):
    document_ids: List[int]
    message: str

@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_documents(
    request: IngestRequest,
    services: Dict = Depends(get_services)
):
    document_store = services["document_store"]
    
    # Add documents to the index
    document_ids = document_store.add_documents(request.documents)
    
    return {
        "document_ids": document_ids,
        "message": f"Successfully ingested {len(document_ids)} documents"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Startup event
@app.on_event("startup")
async def startup_event():
    # Initialize services on startup
    get_services()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

#=============================================================================
# app/services/__init__.py
#=============================================================================
# Empty file to make the directory a Python package

#=============================================================================
# app/services/document_store.py
#=============================================================================
import os
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class DocumentStore:
    def __init__(self, index_path: str = None, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the document store with FAISS index.
        
        Args:
            index_path: Path to load existing FAISS index from
            embedding_model: Name of the sentence-transformers model to use
        """
        self.embedding_dim = 384  # Default dimension for the specified model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.document_store = {}  # Maps IDs to original document chunks
        
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
        else:
            # Create a new index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.current_id = 0
            
    def add_documents(self, documents: List[str]) -> List[int]:
        """
        Add documents to the index.
        
        Args:
            documents: List of document chunks to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
            
        # Generate embeddings
        embeddings = self._get_embeddings(documents)
        
        # Add to FAISS index
        document_ids = []
        for i, doc in enumerate(documents):
            doc_id = self.current_id
            self.document_store[doc_id] = doc
            document_ids.append(doc_id)
            self.current_id += 1
            
        if embeddings.shape[0] > 0:
            self.index.add(embeddings)
            
        return document_ids
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the index.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        # Generate query embedding
        query_embedding = self._get_embeddings([query])
        
        # Search in FAISS index
        scores, ids = self.index.search(query_embedding, top_k)
        
        results = []
        for i, doc_id in enumerate(ids[0]):
            if doc_id >= 0:  # FAISS returns -1 for padding when there are fewer results
                results.append({
                    "id": int(doc_id),
                    "score": float(scores[0][i]),
                    "content": self.document_store[int(doc_id)]
                })
                
        return results
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        return self.embedding_model.encode(texts, convert_to_numpy=True)
    
    def save_index(self, path: str, metadata_path: str = None):
        """
        Save the FAISS index and document store to disk.
        
        Args:
            path: Path to save the FAISS index
            metadata_path: Path to save the document store metadata
        """
        if metadata_path is None:
            metadata_path = f"{os.path.splitext(path)[0]}_metadata.pkl"
            
        # Save FAISS index
        faiss.write_index(self.index, path)
        
        # Save document store and current ID
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'document_store': self.document_store,
                'current_id': self.current_id
            }, f)
            
    def load_index(self, path: str, metadata_path: str = None):
        """
        Load the FAISS index and document store from disk.
        
        Args:
            path: Path to load the FAISS index from
            metadata_path: Path to load the document store metadata from
        """
        if metadata_path is None:
            metadata_path = f"{os.path.splitext(path)[0]}_metadata.pkl"
            
        # Load FAISS index
        self.index = faiss.read_index(path)
        
        # Load document store and current ID
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.document_store = metadata['document_store']
                self.current_id = metadata['current_id']
                
    def get_index_size(self) -> int:
        """Return the number of vectors in the index."""
        return self.index.ntotal

#=============================================================================
# app/services/llm.py
#=============================================================================
import os
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

class LLMService:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM service.
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.model_name = model_name
        
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]], 
                        max_tokens: int = 500) -> str:
        """
        Generate an answer based on the query and retrieved context documents.
        
        Args:
            query: User question
            context_docs: List of retrieved context documents
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated answer
        """
        # Format context for the prompt
        context_text = "\n\n".join([doc["content"] for doc in context_docs])
        
        # Create the system prompt
        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "If the answer cannot be found in the context, acknowledge that you don't have "
            "enough information instead of making up an answer. "
            "Always cite specific parts of the context to support your answer."
        )
        
        # Create the user prompt with context and query
        user_prompt = f"""
Context information is below:
--------------------------
{context_text}
--------------------------

Based on the above context, please answer the following question:
{query}
        """
        
        try:
            # Generate completion using OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3,  # Lower temperature for more focused answers
                top_p=0.9
            )
            
            return response.choices[0].message["content"].strip()
            
        except Exception as e:
            # Handle API errors
            return f"Error generating answer: {str(e)}"
            
    def generate_answer_with_sources(self, query: str, context_docs: List[Dict[str, Any]], 
                                 max_tokens: int = 500) -> Dict[str, Any]:
        """
        Generate an answer with source citations.
        
        Args:
            query: User question
            context_docs: List of retrieved context documents
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with answer and sources
        """
        # Prepare document sources for citation
        sources = []
        for i, doc in enumerate(context_docs):
            sources.append({
                "id": doc.get("id", i),
                "content": doc["content"][:100] + "...",  # Preview of content
                "score": doc.get("score", 0.0)
            })
        
        # Generate the answer
        answer = self.generate_answer(query, context_docs, max_tokens)
        
        return {
            "answer": answer,
            "sources": sources
        }

#=============================================================================
# scripts/ingest.py
#=============================================================================
#!/usr/bin/env python

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.document_store import DocumentStore

def read_files(directory: str, extensions: List[str] = ['.txt', '.md', '.json']) -> List[str]:
    """
    Read files from a directory with specified extensions.
    
    Args:
        directory: Directory path to read files from
        extensions: List of file extensions to include
        
    Returns:
        List of document strings
    """
    documents = []
    path = Path(directory)
    
    for ext in extensions:
        for file_path in path.glob(f"**/*{ext}"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if ext == '.json':
                        # For JSON files, try to parse and extract text
                        content = json.load(f)
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    documents.append(item["text"])
                                elif isinstance(item, str):
                                    documents.append(item)
                        elif isinstance(content, dict) and "text" in content:
                            documents.append(content["text"])
                    else:
                        # For text files, read directly
                        documents.append(f.read())
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                
    return documents

def chunk_documents(documents: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split documents into chunks with specified size and overlap.
    
    Args:
        documents: List of documents to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of document chunks
    """
    chunks = []
    
    for doc in documents:
        # Simple chunking by characters
        doc_length = len(doc)
        if doc_length <= chunk_size:
            chunks.append(doc)
        else:
            for i in range(0, doc_length, chunk_size - chunk_overlap):
                chunk = doc[i:i + chunk_size]
                if len(chunk) >= chunk_size // 2:  # Only add chunks that are at least half the target size
                    chunks.append(chunk)
                    
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system")
    parser.add_argument(
        "--input", "-i", 
        required=True, 
        help="Directory containing documents to ingest"
    )
    parser.add_argument(
        "--output", "-o", 
        default="data/processed/faiss_index.bin", 
        help="Path to save the FAISS index"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=1000, 
        help="Size of document chunks in characters"
    )
    parser.add_argument(
        "--chunk_overlap", 
        type=int, 
        default=200, 
        help="Overlap between consecutive chunks"
    )
    parser.add_argument(
        "--extensions", 
        nargs="+", 
        default=['.txt', '.md', '.json'], 
        help="File extensions to process"
    )
    
    args = parser.parse_args()
    
    print(f"Reading documents from {args.input}")
    documents = read_files(args.input, args.extensions)
    print(f"Found {len(documents)} documents")
    
    print(f"Chunking documents with size {args.chunk_size} and overlap {args.chunk_overlap}")
    chunks = chunk_documents(documents, args.chunk_size, args.chunk_overlap)
    print(f"Created {len(chunks)} chunks")
    
    print("Creating document store and adding documents")
    document_store = DocumentStore()
    document_ids = document_store.add_documents(chunks)
    
    print(f"Saving index to {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    document_store.save_index(args.output)
    
    print(f"Successfully ingested {len(document_ids)} document chunks")
    print(f"FAISS index contains {document_store.get_index_size()} vectors")

if __name__ == "__main__":
    main()

#=============================================================================
# Dockerfile
#=============================================================================
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed

# Make the ingest script executable
RUN chmod +x scripts/ingest.py

# Set environment variables
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

#=============================================================================
# docker-compose.yml
#=============================================================================
version: '3'

services:
  rag-app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    env_file:
      - .env
    restart: unless-stopped

#=============================================================================
# requirements.txt
#=============================================================================
fastapi==0.104.1
uvicorn==0.23.2
faiss-cpu==1.7.4
openai==0.28.1
numpy==1.26.0
sentence-transformers==2.2.2
python-dotenv==1.0.0
pydantic==2.4.2
pytest==7.4.3
python-multipart==0.0.6

#=============================================================================
# README.md
#=============================================================================
# RAG Question Answering System

This project implements a Retrieval-Augmented Generation (RAG) system that answers questions based on a private dataset. It combines efficient document retrieval using FAISS with OpenAI's GPT-3.5 to generate accurate and contextual answers.

## Features

- Document retrieval using FAISS for vector similarity search
- Answer generation with OpenAI's GPT-3.5
- FastAPI backend for serving the system
- Docker support for easy deployment
- Document ingestion script for processing various file formats

## Project Structure

```
rag-project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_store.py # Document storage and retrieval with FAISS
│   │   └── llm.py            # GPT-3.5 integration
├── data/
│   ├── raw/                  # Place your documents here
│   └── processed/            # FAISS index will be stored here
├── scripts/
│   └── ingest.py             # Script to ingest documents
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Setup and Installation

### Local Development

1. Clone the repository:
   ```
   git clone https://your-repository-url.git
   cd rag-project
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Place your documents in the `data/raw` directory

5. Run the ingestion script to process documents and build the index:
   ```
   python scripts/ingest.py -i data/raw
   ```

6. Start the FastAPI server:
   ```
   uvicorn app.main:app --reload
   ```

7. Access the API documentation at `http://localhost:8000/docs`

### Docker Deployment

1. Make sure Docker and Docker Compose are installed

2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Place your documents in the `data/raw` directory

4. Build and start the Docker containers:
   ```
   docker-compose up -d
   ```

5. Run the ingestion script inside the container:
   ```
   docker-compose exec rag-app python scripts/ingest.py -i data/raw
   ```

6. Access the API documentation at `http://localhost:8000/docs`

## API Endpoints

### Ask a Question

```
POST /api/answer
```

Request body:
```json
{
  "question": "What is the capital of France?",
  "top_k": 5,
  "include_sources": true
}
```

Response:
```json
{
  "question": "What is the capital of France?",
  "answer": "The capital of France is Paris.",
  "sources": [
    {
      "id": 42,
      "content": "Paris is the capital and most populous city of France...",
      "score": 0.92
    }
  ],
  "processing_time": 0.5
}
```

### Ingest Documents

```
POST /api/ingest
```

Request body:
```json
{
  "documents": [
    "Document content 1",
    "Document content 2"
  ]
}
```

Response:
```json
{
  "document_ids": [0, 1],
  "message": "Successfully ingested 2 documents"
}
```

## Deploying to AWS EC2

1. Launch an EC2 instance with Docker installed
2. Clone the repository on the instance
3. Follow the Docker deployment steps above
4. Configure security groups to expose port 8000

## Customization

- Modify the embedding model in `document_store.py` to use different sentence transformers
- Adjust chunking parameters in `ingest.py` for different document types
- Update the OpenAI model in `.env` or through environment variables
