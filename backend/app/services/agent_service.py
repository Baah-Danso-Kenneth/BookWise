import json
from pathlib import Path
from app.config import config
from app.tools.book_executor import BookExecutor
from app.graph.state import AgentState
from app.agents.planner import PlannerAgent



class AgentService:
    def __init__(self):
        with open(Path(__file__).parent.parent / "data" / "books.json") as f:
            books_data  = json.load(f)["books"]

        self.executor = BookExecutor(books_data, config.COHERE_API_KEY, config.TAVILY_API_KEY)
        self.planner = PlannerAgent()

        
    
    def recommend(self, query: str) -> dict:
        plan_result = self.planner.process({"query": query})
        plan = plan_result["plan"]

        if plan["type"] == "book_based":
            books = self.executor.recommend_by_book(plan["value"])
        elif plan["type"] == "topic_based":
            books = self.executor.recommend_by_query(plan["value"])
        else:
            books = self.executor.recommend_by_query(query)

        return {
            "books": books,
            "plan": self.planner.get_plan_summary(plan),
            "status": "success"
            }
    
