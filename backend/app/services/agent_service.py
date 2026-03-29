import json
from pathlib import Path
from app.config import config
from app.tools.book_executor import BookExecutor
from app.graph.state import AgentState
from app.agents.planner import PlannerAgent
from app.agents.executor import ExecutorAgent




class AgentService:
    def __init__(self):
        with open(Path(__file__).parent.parent / "data" / "books.json") as f:
            books_data  = json.load(f)["books"]

        self.book_executor = BookExecutor(
            books_data,
            config.COHERE_API_KEY,
            config.TAVILY_API_KEY
        )
        
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent(self.book_executor)

        
    
    def recommend(self, query: str) -> dict:
        plan_result = self.planner.process({"query": query})
        plan = plan_result["plan"]

        execution_result = self.executor.process({"plan": plan})


        return {
            "books": execution_result.get("recommendations", []),
            "plan": self.planner.get_plan_summary(plan),
            "execution": self.executor.get_execution_summary(execution_result),
            "status": execution_result.get("execution_status", "unknown")
            }
    
