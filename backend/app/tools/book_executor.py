import re

import cohere
import numpy as np
import faiss
from typing import List, Dict, Any
from langchain_tavily import TavilySearch 

class BookExecutor:
    def __init__(self, books_data: List[Dict], cohere_api_key: str, tavily_api_key: str):
        self.books = books_data
        self.co = cohere.Client(cohere_api_key)
        self.tavily_api_key = tavily_api_key
        self.tavily = TavilySearch(api_key=tavily_api_key, max_results=3)
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
            response = self.tavily.invoke(search_query)
        
            # Tavily returns a dict with 'results' key containing the list
            if not isinstance(response, dict):
                print(f"Unexpected response type: {type(response)}")
                return []
        
            # Get the actual results list
            results_list = response.get("results", [])
        
            if not results_list:
                print("No results found")
                return []
        
            print(f"Found {len(results_list)} results from Tavily")
        
            recommendations = []

            for result in results_list[:3]:  # Take top 3
                content = result.get("content", "")
                book_title = result.get("title", "Unknown")
                author = self._extract_author(content, book_title)
            
                recommendations.append({
                "title": book_title,
                "author": author,
                "description": content[:200],
                "score": result.get("score", 0.5),
                "reason": "Found via web search"
                })
        
            return recommendations
        
        except Exception as e:
            print(f"Tavily search failed: {e}")
            import traceback
            traceback.print_exc()
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
    

    def _extract_author(self, content: str, title: str) -> str:
        """Extract author from Tavily content"""
        
        # Pattern 1: "by AuthorbaseName"
        match = re.search(r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', content)
        if match:
            return match.group(1)
        
        # Pattern 2: "Author Name (author)"
        match = re.search(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+\(author\)', content)
        if match:
            return match.group(1)
        
        # Pattern 3: ", Author Name"
        match = re.search(r',\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', content)
        if match:
            return match.group(1)
        
        # Pattern 4: "Author Name's" (like "Thomas Nagel's")
        match = re.search(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\'s', content)
        if match:
            return match.group(1)
        
        # Pattern 5: Use website name as fallback
        if "reddit.com" in content.lower():
            return "Reddit Recommendation"
        elif "goodreads.com" in content.lower():
            return "Goodreads"
        
        return "Unknown"