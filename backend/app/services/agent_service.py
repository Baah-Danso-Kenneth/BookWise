import json
from pathlib import Path
from app.config import config
from app.tools.book_executor import BookExecutor
from app.graph.state import AgentState



class AgentService:
    def __init__(self):
        with open(Path(__file__).parent.parent / "data" / "books.json") as f:
            books_data  = json.load(f)["books"]

        self.executor = BookExecutor(books_data, config.COHERE_API_KEY, config.TAVILY_API_KEY)
    
    def recommend(self, query: str) -> dict:
        source_book = None
        for book in self.executor.books:
            if book["title"].lower() in query.lower():
                source_book = book['title']
                break
        
        if source_book:
            books = self.executor.recommend_by_book(source_book)
        else:
            books = self.executor.recommend_by_query(query)

        return {"books": books, "status": "success"}