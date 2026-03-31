import logging
from typing import List, Dict, Any

from app.memory.semantic import SemanticMemory
from app.config import config

LOCAL_THRESHOLD = 0.55


class BookExecutor:
    """
    Executes book searches using SemanticMemory (Cohere + FAISS) for local
    catalogue searches and BookSearchServer (Tavily + LLM extraction) for web fallback.
    """

    def __init__(self, books_data: List[Dict]):
        self.books           = books_data
        self.semantic_memory = SemanticMemory(books_data=books_data)
        # Lazy-loaded so we don't import at module level
        self._book_search    = None
        logging.info(f"BookExecutor ready — {len(books_data)} books indexed")

    def recommend_by_book(self, title: str) -> List[Dict]:
        """Find books similar to a given title"""
        book = self._find_book(title)
        query = (
            f"{book['title']} {book['description']} {' '.join(book.get('categories', []))}"
            if book else title
        )

        results   = self.semantic_memory.search(query, k=6)
        filtered  = [r for r in results if r["title"].lower() != title.lower()]
        top_score = filtered[0].get("similarity_score", 0) if filtered else 0

        logging.info(f"BookExecutor: local top_score={top_score:.3f} threshold={LOCAL_THRESHOLD}")

        if not filtered or top_score < LOCAL_THRESHOLD:
            logging.info("BookExecutor: falling back to BookSearchServer")
            return self._search_web(title, search_type="book_based")

        return self._format_results(filtered[:5])

    def recommend_by_query(self, query: str) -> List[Dict]:
        """Find books matching a topic or complex query"""
        results   = self.semantic_memory.search(query, k=5)
        top_score = results[0].get("similarity_score", 0) if results else 0

        logging.info(f"BookExecutor: local top_score={top_score:.3f} threshold={LOCAL_THRESHOLD}")

        if not results or top_score < LOCAL_THRESHOLD:
            logging.info("BookExecutor: falling back to BookSearchServer")
            return self._search_web(query, search_type="topic_based")

        return self._format_results(results)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _search_web(self, query: str, search_type: str = "topic_based") -> List[Dict]:
        """
        Delegate to BookSearchServer which uses Tavily + LLM extraction.
        This ensures web results are parsed into clean book entries,
        not raw webpage titles/snippets.
        """
        book_search = self._get_book_search()
        result      = book_search.execute(query=query, search_type=search_type, max_results=5)
        return result.content.get("books", [])

    def _get_book_search(self):
        """Lazy-init BookSearchServer to avoid circular imports"""
        if self._book_search is None:
            from app.tools.book_search import BookSearchServer
            self._book_search = BookSearchServer()
        return self._book_search

    def _find_book(self, title: str) -> Dict:
        for book in self.books:
            if book["title"].lower() == title.lower():
                return book
        return None

    def _format_results(self, results: List[Dict]) -> List[Dict]:
        formatted = []
        for r in results:
            score = r.get("similarity_score", 0.5)
            formatted.append({
                "title":       r.get("title", ""),
                "author":      r.get("author", "Unknown"),
                "description": r.get("description", ""),
                "score":       round(score, 4),
                "reason":      self._reason(score)
            })
        return formatted

    def _reason(self, score: float) -> str:
        if score > 0.7:
            return "Highly relevant to your interest"
        elif score > 0.5:
            return "Related to your interest"
        return "May be of interest based on themes"