import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from app.prompts import PLANNER_SYSTEM_PROMPT
from app.agents.base import A2AAgent, A2AAgentCard, A2ATask, A2ATaskStatus


class PlannerAgent(A2AAgent):
    """
    Planner Agent - Understands user queries and creates execution plans.

    Types of plans:
    - book_based: User mentioned a specific book title
    - topic_based: User asked about a category/topic
    - complex: User has multi-criteria request (mood, tone, etc...)
    """

    def __init__(self):
        super().__init__()
        self._load_books()

    # ── Required by A2AAgent ────────────────────────────────────────────────

    def _register_card(self) -> A2AAgentCard:
        return A2AAgentCard(
            name="planner-agent",
            version="1.0.0",
            description="Analyses user queries and creates structured execution plans",
            capabilities=["query_understanding", "plan_generation", "intent_classification"],
            input_modes=["text"],
            output_modes=["plan"],
            endpoint="in-process://planner-agent/tasks",
            skills=[{
                "name": "create_plan",
                "description": "Turn a user query into a typed execution plan",
                "input": {"query": "string"},
                "output": {"type": "string", "value": "string", "intent": "string"}
            }]
        )

    def process_task(self, task: A2ATask) -> A2ATask:
        """A2A-compliant entry point — delegates to process()"""
        task.update_status(A2ATaskStatus.WORKING)
        try:
            query = task.context.get("query", "")
            result = self.process({"query": query})
            task.result = result
            task.update_status(A2ATaskStatus.COMPLETED)
        except Exception as e:
            logging.error(f"PlannerAgent failed: {e}")
            task.errors.append(str(e))
            task.update_status(A2ATaskStatus.FAILED)
        self._record_task(task)
        return task

    # ── Core logic ──────────────────────────────────────────────────────────

    def _load_books(self):
        """Load local book titles for title detection"""
        try:
            books_path = Path(__file__).parent.parent / "data" / "books.json"
            with open(books_path) as f:
                books_data = json.load(f)
                self.local_titles = [b["title"].lower() for b in books_data.get("books", [])]
        except Exception as e:
            logging.warning(f"Could not load books for Planner: {e}")
            self.local_titles = []

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user query and create execution plan.

        Args:
            task: Dictionary with "query" key containing user input.

        Returns:
            Dictionary with "plan", "original_query", "status"
        """
        query = task.get("query", "")
        logging.info(f"Planner: processing query: {query[:100]}...")

        plan = self._rule_based_plan(query)

        if plan["type"] == "fallback":
            plan = self._llm_plan(query)

        logging.info(f"Planner: created plan type: {plan['type']}")

        return {
            "plan": plan,
            "original_query": query,
            "status": "success"
        }

    def _rule_based_plan(self, query: str) -> Dict[str, Any]:
        """Simple rule-based planning for common patterns"""
        query_lower = query.lower()

        for title in self.local_titles:
            if title in query_lower:
                original = title.title()
                return {
                    "type": "book_based",
                    "value": original,
                    "intent": "find books similar to specified title",
                    "confidence": "high"
                }

        topics = {
            "philosophy": ["philosophy", "stoic", "existential", "ethics"],
            "finance": ["finance", "money", "wealth", "investing", "rich"],
            "history": ["history", "historical", "ancient", "war"],
            "science": ["science", "physics", "biology", "space", "universe"],
            "self-help": ["self-help", "self help", "personal development", "habits"],
            "psychology": ["psychology", "mind", "behavior", "brain"]
        }

        for topic, keywords in topics.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return {
                        "type": "topic_based",
                        "value": topic,
                        "intent": f"find books about {topic}",
                        "confidence": "medium"
                    }

        moods = {
            "light_funny": ["light", "funny", "humor", "comedy", "entertaining"],
            "serious_deep": ["serious", "deep", "profound", "thoughtful"],
            "educational": ["educational", "learn", "knowledge", "informative"],
            "inspiring": ["inspiring", "motivational", "encouraging"]
        }

        detected_moods = []
        for mood, keywords in moods.items():
            for keyword in keywords:
                if keyword in query_lower:
                    detected_moods.append(mood)

        if detected_moods:
            return {
                "type": "complex",
                "value": query,
                "intent": f"Find books that are {', '.join(detected_moods)}",
                "confidence": "medium"
            }

        return {
            "type": "fallback",
            "value": query,
            "intent": "Could not determine specific plan, fallback to LLM",
            "confidence": "low"
        }

    def _llm_plan(self, query: str) -> Dict[str, Any]:
        """Use LLM for complex or unclear queries"""
        try:
            response = self._invoke_llm(PLANNER_SYSTEM_PROMPT, f"Query: {query}")

            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()
            plan = json.loads(clean)
            plan["confidence"] = "high" if plan.get("type") != "complex" else "medium"
            return plan

        except json.JSONDecodeError as e:
            logging.error(f"Planner: JSON parse failed: {e}")
        except Exception as e:
            logging.error(f"Planner: LLM Planning failed: {e}")

        return {
            "type": "topic_based",
            "value": query,
            "intent": "fallback to keyword search",
            "confidence": "low"
        }

    def get_plan_summary(self, plan: Dict[str, Any]) -> str:
        """Get human-readable summary of the plan"""
        plan_type = plan.get("type", "unknown")
        if plan_type == "book_based":
            return f"Searching for books similar to {plan.get('value', 'unknown')}"
        elif plan_type == "topic_based":
            return f"Finding books about {plan.get('value', 'this topic')}"
        elif plan_type == "complex":
            return f"Searching for: {plan.get('intent', 'your request')}"
        return "Understanding your request..."