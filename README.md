# RAG_Project
 Question Answering System
This project implements a Retrieval-Augmented Generation (RAG) system that answers questions based on a private dataset. It combines efficient document retrieval using FAISS with OpenAI's GPT-3.5 to generate accurate and contextual answers.
Features

Document retrieval using FAISS for vector similarity search
Answer generation with OpenAI's GPT-3.5
FastAPI backend for serving the system
Docker support for easy deployment
Document ingestion script for processing various file formats

Project Structure
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
Setup and Installation
Local Development

Clone the repository:
git clone https://your-repository-url.git
cd rag-project

Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Create a .env file with your OpenAI API key:
OPENAI_API_KEY=your_api_key_here

Place your documents in the data/raw directory
Run the ingestion script to process documents and build the index:
python scripts/ingest.py -i data/raw

Start the FastAPI server:
uvicorn app.main:app --reload

Access the API documentation at http://localhost:8000/docs

Docker Deployment

Make sure Docker and Docker Compose are installed
Create a .env file with your OpenAI API key:
OPENAI_API_KEY=your_api_key_here

Place your documents in the data/raw directory
Build and start the Docker containers:
docker-compose up -d

Run the ingestion script inside the container:
docker-compose exec rag-app python scripts/ingest.py -i data/raw

Access the API documentation at http://localhost:8000/docs

API Endpoints
Ask a Question
POST /api/answer
Request body:
json{
  "question": "What is the capital of France?",
  "top_k": 5,
  "include_sources": true
}
Response:
json{
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
Ingest Documents
POST /api/ingest
Request body:
json{
  "documents": [
    "Document content 1",
    "Document content 2"
  ]
}
Response:
json{
  "document_ids": [0, 1],
  "message": "Successfully ingested 2 documents"
}
Deploying to AWS EC2

Launch an EC2 instance with Docker installed
Clone the repository on the instance
Follow the Docker deployment steps above
Configure security groups to expose port 8000

Customization

Modify the embedding model in document_store.py to use different sentence transformers
Adjust chunking parameters in ingest.py for different document types
Update the OpenAI model in .env or through environment variables
