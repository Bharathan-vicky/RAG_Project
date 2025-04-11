# app/services/document_store.py

import os
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Tuple
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
