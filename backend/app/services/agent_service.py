import json
from pathlib import Path

from app.agents.planner import PlannerAgent
from app.agents.executor import ExecutorAgent
from app.agents.critic import CriticAgent
from app.tools.guardrail import GuardrailServer
from app.graph.graph import run_graph
from app.graph.nodes import init_container
from app.memory.episodic import EpisodicMemory


class AgentService:
    def __init__(self):
        # Load book catalogue
        data_path = Path(__file__).parent.parent / "data" / "books.json"
        with open(data_path) as f:
            self.books_data = json.load(f)["books"]

        # Instantiate agents
        self.planner   = PlannerAgent()
        self.executor  = ExecutorAgent()
        self.critic    = CriticAgent()
        self.guardrail = GuardrailServer()

        # Wire all agents into the graph via container (no globals)
        init_container(self.planner, self.executor, self.critic, self.guardrail)

        # Initialise BookExecutor (builds SemanticMemory FAISS index)
        self.executor.set_books_data(self.books_data)

    # ── Public API ───────────────────────────────────────────────────────────

    async def recommend(self, query: str, user_id: str = "default_user") -> dict:
        """Run query through the full async LangGraph pipeline"""
        final_state = await run_graph(query, user_id)

        memory = EpisodicMemory(user_id)
        memory.add_to_history(query, final_state.get("final_recommendations", []))

        return {
            "books":            final_state.get("final_recommendations", []),
            "score":            final_state.get("critic_score", 0),
            "feedback":         final_state.get("critic_feedback", ""),
            "attempts":         final_state.get("attempt_number", 1),
            "disclaimer_added": final_state.get("disclaimer_added", False),
            "conversation_id":  final_state.get("conversation_id"),
            "status":           "success"
        }

    async def rate_book(self, user_id: str, title: str, rating: int) -> dict:
        memory = EpisodicMemory(user_id)
        memory.add_read_book(title, rating)
        return {"status": "success", "message": f"Rated '{title}' {rating}/5"}

    async def get_history(self, user_id: str) -> dict:
        memory = EpisodicMemory(user_id)
        return {
            "read_books":      memory.get_read_books(),
            "preferences":     memory.get_preferences(),
            "session_history": memory.data.get("session_history", [])
        }