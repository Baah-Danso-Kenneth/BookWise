import cohere
import numpy as np
import faiss
from typing import List, Dict, Any
from langchain_community.tools.tavily_search import TavilySearchResults


class BookExecutor:
    def __init__(self, books_data: List[Dict], cohere_api_key: str, tavily_api_key: str):
        self.books = books_data
        self.co = cohere.Client(cohere_api_key)
        self.tavily = TavilySearchResults(api_key=tavily_api_key, max_results=3)
        self._build_faiss_index()


    def _build_faiss_index(self):
        """
        Create FAISS indexing using Cohere embeddings
        """
        texts = []
        for book in self.books:
            text = f"{book['title']}\n{book['description']}\nCategories: {', '.join(book['categories'])}"
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

        if book_idx is not None:
            book = self.books[book_idx]
            query_text = f"{book['title']}\n{book['description']}\nCategories: {', '.join(book['categories'])}"

            return self._search(query_text, title)
        else:
            return self._search_web(title)
    

    def recommend_by_query(self, query: str) -> List[Dict]:
        """Find books matching user's query"""
        local_results = self._search(query)

        if not local_results or local_results[0]["score"] < 0.5:
            web_results = self._search_web(query)
            return web_results 
        
        return local_results
    
    
    def _search(self, query: str, exclude_title: str = None) -> List[Dict]:
        """Search FAISS index with query"""
        response = self.co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
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
                "reason": self._generate_reason(book, similarity)
            })

        return recommendations[:5]
        

    def _search_web(self, query: str) -> List[Dict]:
        """Search Tavily for books similar to query"""
        search_query = f"books similar to {query} recommendations"
    
        try:
            results = self.tavily.invoke(search_query)
            recommendations = []

            for result in results[:3]:
                content = result.get("content", "")
            
                # Try to extract book info from content
                # Often looks like: "Book Title by Author Name - description"
                book_title = result.get("title", "Unknown")
                author = "Unknown"
            
                # Simple extraction: look for "by [Author]" pattern
                if " by " in content[:100]:
                    parts = content[:100].split(" by ")
                    if len(parts) > 1:
                        book_title = parts[0].strip()
                        author = parts[1].split(" -")[0].strip()
            
                recommendations.append({
                "title": book_title,
                "author": author,
                "description": content[:200],
                "score": result.get("score", 0.5),
                "reason": f"Found via web search"
                })
        
            return recommendations
        
        except Exception as e:
            print(f"Tavily search failed: {e}")
            return []
        
    
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