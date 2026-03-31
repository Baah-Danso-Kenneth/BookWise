import logging
from typing import Any, Dict, List

from app.agents.base import A2AAgent, A2AAgentCard, A2ATask, A2ATaskStatus


class ExecutorAgent(A2AAgent):
    """
    Executor Agent - Executes plans using BookExecutor (SemanticMemory + Tavily).
    Books data is injected after construction via set_books_data().
    """

    def __init__(self):
        super().__init__()
        self.book_executor = None
        self.book_search   = None
        self.taste_analyzer = None

    def _register_card(self) -> A2AAgentCard:
        return A2AAgentCard(
            name="executor-agent",
            version="1.0.0",
            description="Executes book search plans and returns ranked recommendations",
            capabilities=["semantic_search", "web_search", "taste_analysis"],
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
        """Execute the plan — called by executor_node in nodes.py"""
        plan       = task.get("plan", {})
        plan_type  = plan.get("type", "unknown")
        plan_value = plan.get("value", "")
        read_books = task.get("read_books", [])

        logging.info(f"Executor: plan type='{plan_type}' value='{plan_value}'")

        try:
            # ── Local semantic search via BookExecutor ───────────────────────
            if self.book_executor:
                if plan_type == "book_based":
                    books = self.book_executor.recommend_by_book(plan_value)
                else:
                    books = self.book_executor.recommend_by_query(plan_value)
            else:
                # Fallback: MCP BookSearchServer if BookExecutor not ready
                books = self._fallback_search(plan_value, plan_type)

            # Filter already-read books
            read_lower = [rb.lower() for rb in read_books]
            filtered   = [b for b in books if b.get("title", "").lower() not in read_lower]

            # Taste analysis
            if filtered and self.taste_analyzer:
                taste_result  = self.taste_analyzer.execute(
                    books=filtered,
                    user_preferences=task.get("preferences", {})
                )
                recommendations = taste_result.content.get("analyzed_books", filtered)
            else:
                recommendations = filtered

            logging.info(f"Executor: {len(recommendations)} recommendations")

            return {
                "recommendations":  recommendations,
                "execution_status": "success",
                "plan_executed":    plan,
                "count":            len(recommendations)
            }

        except Exception as e:
            logging.error(f"Executor failed: {e}")
            return {
                "recommendations":  [],
                "execution_status": "failed",
                "error":            str(e),
                "plan_executed":    plan
            }

    def set_books_data(self, books_data: List[Dict]):
        """
        Inject book catalogue and initialise BookExecutor (SemanticMemory + FAISS).
        Called once by AgentService at startup.
        """
        from app.tools.book_executor import BookExecutor
        from app.tools.taste_analyzer import TasteAnalyzerServer
        from app.tools.book_search import BookSearchServer

        self.book_executor  = BookExecutor(books_data=books_data)
        self.taste_analyzer = TasteAnalyzerServer()
        self.book_search    = BookSearchServer()

        logging.info("ExecutorAgent: BookExecutor and tools ready")

    def _fallback_search(self, query: str, plan_type: str) -> List[Dict]:
        """MCP BookSearchServer fallback if BookExecutor unavailable"""
        if self.book_search:
            result = self.book_search.execute(query=query, search_type=plan_type)
            return result.content.get("books", [])
        return []