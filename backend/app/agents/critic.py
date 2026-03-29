import json
import logging
from typing import Dict, Any, List, Union

from app.agents.base import A2AAgent, A2AAgentCard, A2ATask, A2ATaskStatus
from app.prompts import CRITIC_SYSTEM_PROMPT


class CriticAgent(A2AAgent):
    """
    Critic Agent - Evaluates recommendations and triggers self-correction.

    Scores on:
    - Relevance  (40 pts) — do descriptions/titles relate to the query?
    - Diversity  (30 pts) — are titles unique?
    - Quality    (30 pts) — are descriptions meaningful (>50 chars)?
    """

    def __init__(self):
        super().__init__()

    def _register_card(self) -> A2AAgentCard:
        return A2AAgentCard(
            name="critic-agent",
            version="1.0.0",
            description="Evaluates book recommendations and triggers self-correction",
            capabilities=["recommendation_scoring", "quality_evaluation", "self_correction"],
            input_modes=["recommendations"],
            output_modes=["score", "verdict"],
            endpoint="in-process://critic-agent/tasks",
            skills=[{
                "name": "evaluate_recommendations",
                "description": "Score and evaluate recommendations",
                "input": {"recommendations": "list", "query": "string"},
                "output": {"score": "float", "verdict": "string", "feedback": "string"}
            }]
        )

    def process_task(self, task: Union[Dict[str, Any], A2ATask]) -> Union[Dict[str, Any], A2ATask]:
        """Accepts plain dict (LangGraph nodes) or A2ATask (A2A protocol)"""
        if isinstance(task, dict):
            return self._evaluate_dict(task)

        # A2ATask path
        task.update_status(A2ATaskStatus.WORKING)
        try:
            result = self._evaluate_dict(task.context)
            task.result = result
            task.update_status(A2ATaskStatus.COMPLETED)
            logging.info(f"CriticAgent: verdict={result['verdict']}, score={result['score']}/100")
        except Exception as e:
            logging.error(f"CriticAgent failed: {e}")
            task.errors.append(str(e))
            task.update_status(A2ATaskStatus.FAILED)
        self._record_task(task)
        return task

    # ── Core evaluation ──────────────────────────────────────────────────────

    def _evaluate_dict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        recommendations = context.get("recommendations", [])
        query           = context.get("query", "")
        attempt         = context.get("attempt", 1)

        evaluation = self._score_recommendations(recommendations, query)
        verdict    = self._determine_verdict(evaluation, attempt)

        return {
            "score":    evaluation["total_score"],
            "verdict":  verdict,
            "feedback": evaluation["feedback"],
            "scores":   evaluation["scores"],
            "issues":   evaluation.get("issues", []),
            "attempt":  attempt
        }

    def _score_recommendations(self, recommendations: List[Dict], query: str) -> Dict[str, Any]:
        if not recommendations:
            return {
                "total_score": 0,
                "feedback": "No recommendations found",
                "scores": {"relevance": 0, "diversity": 0, "quality": 0},
                "issues": ["empty_results"]
            }

        query_words = set(query.lower().split())

        # ── Relevance (40 pts) ───────────────────────────────────────────────
        # Check how many books have titles/descriptions that overlap with query words
        # OR have a non-trivial description (meaning the search found real content)
        relevance_hits = 0
        for r in recommendations:
            title       = r.get("title", "").lower()
            description = r.get("description", "").lower()
            author      = r.get("author", "").lower()
            combined    = f"{title} {description} {author}"

            # Award relevance if description is substantial and not generic
            if len(description) > 80:
                relevance_hits += 1
            elif any(w in combined for w in query_words if len(w) > 3):
                relevance_hits += 1

        relevance_score = min(40, int((relevance_hits / len(recommendations)) * 40))

        # ── Diversity (30 pts) ───────────────────────────────────────────────
        titles         = [r.get("title", "") for r in recommendations]
        unique_ratio   = len(set(titles)) / len(recommendations)
        diversity_score = min(30, int(unique_ratio * 30))

        # ── Quality (30 pts) ─────────────────────────────────────────────────
        # Award 10 pts per book with a real description (>50 chars) and known author
        quality_score = 0
        for r in recommendations:
            desc   = r.get("description", "")
            author = r.get("author", "Unknown")
            if len(desc) > 50 and author != "Unknown":
                quality_score += 10
            elif len(desc) > 50:
                quality_score += 5
        quality_score = min(30, quality_score)

        total_score = relevance_score + diversity_score + quality_score
        feedback    = self._generate_feedback(relevance_score, diversity_score, quality_score)

        issues = []
        if relevance_score < 25:
            issues.append("low_relevance")
        if diversity_score < 20:
            issues.append("low_diversity")
        if quality_score < 20:
            issues.append("low_quality")

        return {
            "total_score": total_score,
            "feedback":    feedback,
            "scores":      {"relevance": relevance_score, "diversity": diversity_score, "quality": quality_score},
            "issues":      issues
        }

    def _determine_verdict(self, evaluation: Dict[str, Any], attempt: int) -> str:
        score = evaluation["total_score"]
        if attempt >= 3:
            logging.info(f"Max attempts ({attempt}) reached, forcing PASS")
            return "PASS"
        if score >= 70:
            return "PASS"
        elif score >= 50:
            return "REVISE"
        return "FAIL"

    def _generate_feedback(self, relevance: int, diversity: int, quality: int) -> str:
        parts = []
        if relevance < 25:
            parts.append(f"Relevance is low ({relevance}/40). Results don't match query well.")
        elif relevance > 35:
            parts.append(f"Good relevance ({relevance}/40).")
        if diversity < 20:
            parts.append(f"Lack of diversity ({diversity}/30).")
        elif diversity > 25:
            parts.append(f"Good diversity ({diversity}/30).")
        if quality < 20:
            parts.append(f"Quality issues ({quality}/30). Some descriptions are incomplete.")
        elif quality > 25:
            parts.append(f"Good quality ({quality}/30).")
        if not parts:
            return f"Score: {relevance + diversity + quality}/100. Acceptable recommendations."
        return " ".join(parts)

    def get_critique_summary(self, result: Dict[str, Any]) -> str:
        return f"[{result.get('verdict', 'unknown')}] Score: {result.get('score', 0)}/100 - {result.get('feedback', '')}"