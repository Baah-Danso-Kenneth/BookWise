import logging
import numpy as np
import faiss
from typing import List, Dict, Any
import cohere

from app.memory.base import MemoryBase
from app.config import config


class SemanticMemory(MemoryBase):
    """
    Semantic Memory using Cohere embeddings + FAISS.
    Stores book knowledge for similarity search.
    """
    
    def __init__(self, books_data: List[Dict] = None):
        self.co = cohere.Client(config.COHERE_API_KEY)
        self.index = None
        self.documents = []
        self.metadata = []
        
        if books_data:
            self._build_index(books_data)
    
    def _build_index(self, books_data: List[Dict]):
        """Build FAISS index from book data"""
        logging.info("Building semantic memory index...")
        
        self.documents = []
        self.metadata = []
        
        for book in books_data:
            # Create rich text for embedding
            text = f"{book['title']}\n{book['description']}\nCategories: {', '.join(book.get('categories', []))}"
            self.documents.append(text)
            self.metadata.append({
                "title": book["title"],
                "author": book["author"],
                "categories": book.get("categories", []),
                "description": book.get("description", "")
            })
        
        # Get embeddings
        response = self.co.embed(
            texts=self.documents,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        
        embeddings = np.array(response.embeddings).astype("float32")
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        logging.info(f"Semantic memory built with {len(self.documents)} books")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar books"""
        if not self.index:
            return []
        
        # Get query embedding
        response = self.co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        
        query_embedding = np.array(response.embeddings).astype("float32")
        
        # Search
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            similarity = 1 / (1 + distances[0][i])
            results.append({
                **self.metadata[idx],
                "similarity_score": float(similarity),
                "content": self.documents[idx][:200]
            })
        
        return results
    
    def store(self, key: str, value: Any) -> bool:
        """Add new book to memory"""
        # Not implemented for MVP
        logging.warning("SemanticMemory.store not implemented")
        return False
    
    def retrieve(self, key: str) -> Any:
        """Retrieve by key"""
        # Not implemented for MVP
        return None
    
    def update(self, key: str, value: Any) -> bool:
        """Update existing memory"""
        return False
    
    def delete(self, key: str) -> bool:
        """Delete from memory"""
        return False