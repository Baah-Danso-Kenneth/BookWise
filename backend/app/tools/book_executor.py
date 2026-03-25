import cohere
import numpy as np
import faiss
from typing import List, Dict, Any



class BookExecutor:
    def __init__(self, books_data: List[Dict], cohere_api_key: str):
        self.books = books_data
        self.co = cohere.Client(cohere_api_key)
        self._build_faiss_index()


    def _build_faiss_index(self):
        """
        Create FAISS indexing using Cohere embeddings
        """
        texts = []
        for book in self.books:
            text = f"{book["title"]}\n{book["description"]}\nCategories: {', '.join(book["categories"])}"
            texts.append(text)
        
        response = self.co.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )

        embeddings = np.array(response.embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)


    
    def recommend_by_book(self, title: str) -> List[Dict]:
        """Find books similar to given title"""
        book_idx = self._find_book_index(title)
        if book_idx is None:
            return []
        book = self.books[book_idx]
        query_text = f"{book['title']}\n{book['description']}\nCategories: {', '.join(book['categories'])}"

        return self._search(query_text, title)
    

    def recommend_by_query(self, query: str) -> List[Dict]:
        """Find books matching user's query"""
        return self._search(query, query)
    
    
    def _search(self, query: str, exclude_title: str = None) -> List[Dict]:
        """Search FAISS index with query"""
        response = self.co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_document"
        )
        query_embedding = np.array(response.embeddings).astype("float32")
        k = min(5, len(self.books))
        distances, indices = self.index.search(query_embedding, k)

        recommendations = []

        for i, idx in enumerate(indices[0]):
            book = self.books[idx]

            if exclude_title and book['title'].lower() == exclude_title.lower():
                continue

            similarity = 1 / (1 + distances[0][i])

            recommendations.append({
                "title": book['title'],
                "author": book['author'],
                "description": book['description'],
                "score": float(similarity),
                "reason": self._generate_reason(query, book)
            })

            return recommendations[:5]
        
    
    def _generate_reason(self, book: Dict, similarity: float) -> str:
        """Generate human-readable reason for recommendation"""
        if similarity > 0.7:
            return f"Highly relevant to your interest"
        
        elif similarity > 0.5:
            return f"Related to your interest"
        else:
            return f"May be of interest based on categories and description"


    def _find_book_index(self, title: str) -> int:
        for i, book in enumerate(self.books):
            if book['title'].lower() == title.lower():
                return i
        return None