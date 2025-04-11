#!/usr/bin/env python
# scripts/ingest.py

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
