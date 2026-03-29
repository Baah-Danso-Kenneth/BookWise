import logging
from typing import Any, Dict, List

from app.agents.base import A2AAgent, A2AAgentCard, A2ATask, A2ATaskStatus


class ExecutorAgent(A2AAgent):
    """
    Executor Agent - Executes plans and returns book recommendations.
    Books data is injected after construction via set_books_data().
    """

    def __init__(self):
        super().__init__()
        self.books_data: List[Dict] = []
        self.book_titles: List[str] = []
        self.book_search = None
        self.taste_analyzer = None

    # ── Required by A2AAgent ────────────────────────────────────────────────

    def _register_card(self) -> A2AAgentCard:
        return A2AAgentCard(
            name="executor-agent",
            version="1.0.0",
            description="Executes book search plans and returns ranked recommendations",
            capabilities=["book_search", "semantic_search", "web_search"],
            input_modes=["plan"],
            output_modes=["recommendations"],
            endpoint="in-process://executor-agent/tasks",
            skills=[{
                "name": "execute_plan",
                "description": "Search for books matching a plan",
                "input": {"plan": "dict", "read_books": "list"},
                "output": {"recommendations": "list"}
            }]
        )

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the plan from Planner.

        Accepts a plain dict (as called by executor_node in nodes.py).
        Returns a dict with recommendations and execution status.
        """
        plan = task.get("plan", {})
        plan_type = plan.get("type", "unknown")
        plan_value = plan.get("value", "")
        read_books = task.get("read_books", [])

        logging.info(f"Executor: executing plan type '{plan_type}' with value '{plan_value}'")

        try:
            # Lazy-init MCP tools (only after set_books_data has been called)
            self._ensure_tools()

            search_result = self.book_search.execute(
                query=plan_value,
                search_type=plan_type,
                max_results=5
            )

            books = search_result.content.get("books", [])

            # Filter out books already read by the user
            read_lower = [rb.lower() for rb in read_books]
            filtered = [b for b in books if b.get("title", "").lower() not in read_lower]

            # Apply taste analysis
            if filtered:
                taste_result = self.taste_analyzer.execute(
                    books=filtered,
                    user_preferences=task.get("preferences", {})
                )
                recommendations = taste_result.content.get("analyzed_books", filtered)
            else:
                recommendations = filtered

            logging.info(f"Executor: found {len(recommendations)} recommendations")

            return {
                "recommendations": recommendations,
                "search_results": search_result.content,
                "execution_status": "success",
                "plan_executed": plan,
                "count": len(recommendations)
            }

        except Exception as e:
            logging.error(f"Executor: execution failed: {e}")
            return {
                "recommendations": [],
                "execution_status": "failed",
                "error": str(e),
                "plan_executed": plan
            }

    # ── Helpers ─────────────────────────────────────────────────────────────

    def set_books_data(self, books_data: List[Dict]):
        """Inject book data after construction (called by AgentService)"""
        self.books_data = books_data
        self.book_titles = [b["title"].lower() for b in books_data]
        self._ensure_tools()

    def _ensure_tools(self):
        """Lazy-initialise MCP tool servers"""
        if self.book_search is None:
            from app.tools.book_search import BookSearchServer
            self.book_search = BookSearchServer()
        if self.taste_analyzer is None:
            from app.tools.taste_analyzer import TasteAnalyzerServer
            self.taste_analyzer = TasteAnalyzerServer()

    def get_execution_summary(self, result: Dict[str, Any]) -> str:
        status = result.get("execution_status", "unknown")
        count = result.get("count", 0)
        if status == "success":
            return f"Found {count} book recommendations"
        return f"Execution failed: {result.get('error', 'unknown error')}"