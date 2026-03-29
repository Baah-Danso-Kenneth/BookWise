import logging
from typing import Any, Dict, List
from app.agents.base import A2AAgent
from app.tools.book_executor import BookExecutor


class ExecutorAgent(A2AAgent):
    """
    Executor Agent - Executes plans using BookExecutor tool.
    Takes a plan from the Planner, executes the appropriate search,
    and return recommendations.
    """

    def __init__(self, book_executor: BookExecutor):
        super().__init__()
        self.executor = book_executor

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the plan from Planner.

        Args:
            task: Dictionary with "Plan" key containing execution plan.

        Returns:
            Dictionary with recommendations and execution status
        """

        plan = task.get("plan", {})
        plan_type = plan.get("type", "unknown")
        plan_value = plan.get("value", "")

        logging.info(f"Executor: executing plan type '{plan_type}' with value '{plan_value}'")

        try:
            if plan_type == "book_based":
                recommendations = self.executor.recommend_by_book(plan_value)
            elif plan_type == "topic_based":
                recommendations = self.executor.recommend_by_query(plan_value)
            elif plan_type == "complex":
                # Complex queries use the original query as search
                recommendations = self.executor.recommend_by_query(plan_value)
            else:
                recommendations = self.executor.recommend_by_query(plan_value)

            execution_result = {
                "recommendations": recommendations,
                "execution_status": "success",
                "plan_executed": plan,
                "count": len(recommendations)
            }

            logging.info(f"Executor: found {len(recommendations)} recommendations")

            return execution_result
        
        except Exception as e:
            logging.error(f"Executor: execution failed: {e}")

            return {
                "recommendations": [],
                "execution_status": "failed",
                "error": str(e),
                "plan_executed": plan
            }
        

    def get_execution_summary(self, result: Dict[str, Any]) -> str:

        """
        Get human-readable summary of execution 
        """

        status = result.get("execution_status", "unknown")
        count = result.get("count", 0)

        if status == "success":
            return f"Found {count} book recommendations"
        
        else:
            return f"Execution failed: {result.get('error', 'unknown error')}"