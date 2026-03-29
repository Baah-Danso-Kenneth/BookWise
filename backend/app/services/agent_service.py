import json
from pathlib import Path

from app.agents.planner import PlannerAgent
from app.agents.executor import ExecutorAgent
from app.agents.critic import CriticAgent
from app.tools.guardrail import GuardrailServer
from app.graph.graph import run_graph
from app.graph.nodes import set_agents, set_guardrail
from app.memory.episodic import EpisodicMemory


class AgentService:
    def __init__(self):
        # Load book catalogue
        data_path = Path(__file__).parent.parent / "data" / "books.json"
        with open(data_path) as f:
            self.books_data = json.load(f)["books"]

        # Instantiate agents
        self.planner   = PlannerAgent()
        self.executor  = ExecutorAgent()   # no args — books injected below
        self.critic    = CriticAgent()
        self.guardrail = GuardrailServer()

        # Wire agents into LangGraph nodes
        set_agents(self.planner, self.executor, self.critic)
        set_guardrail(self.guardrail)

        # Give executor the local book catalogue + lazy-init its MCP tools
        self.executor.set_books_data(self.books_data)


    def recommend(self, query: str, user_id: str = "default_user") -> dict:
        """Run query through the full LangGraph pipeline"""
        final_state = run_graph(query, user_id)

        # Persist session to episodic memory
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


    def rate_book(self, user_id: str, title: str, rating: int) -> dict:
        """Record a rating and update episodic memory"""
        memory = EpisodicMemory(user_id)
        memory.add_read_book(title, rating)
        return {"status": "success", "message": f"Rated '{title}' {rating}/5"}


    def get_history(self, user_id: str) -> dict:
        """Return user's reading history and preferences"""
        memory = EpisodicMemory(user_id)
        return {
            "read_books":      memory.get_read_books(),
            "preferences":     memory.get_preferences(),
            "session_history": memory.data.get("session_history", [])
        }