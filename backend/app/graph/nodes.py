import logging
from typing import Dict, Any

from app.graph.state import AgentState
from app.memory.episodic import EpisodicMemory


class AgentContainer:
    """
    Holds all agent and tool instances.
    Passed into the graph at build time — no mutable module-level globals.
    """
    def __init__(self, planner, executor, critic, guardrail):
        self.planner   = planner
        self.executor  = executor
        self.critic    = critic
        self.guardrail = guardrail


# Single container instance — set once by AgentService at startup
_container: AgentContainer = None


def init_container(planner, executor, critic, guardrail):
    """Called once by AgentService to wire everything up"""
    global _container
    _container = AgentContainer(planner, executor, critic, guardrail)


def get_container() -> AgentContainer:
    if _container is None:
        raise RuntimeError("AgentContainer not initialised — call init_container() first")
    return _container


# ── Node 1: Planner ──────────────────────────────────────────────────────────

async def planner_node(state: AgentState) -> Dict[str, Any]:
    logging.info(f"PlannerNode: processing query: {state['query']}")

    query   = state["query"]
    user_id = state.get("user_id", "default_user")

    memory      = EpisodicMemory(user_id)
    preferences = memory.get_preferences()
    read_books  = memory.get_read_books()

    result = get_container().planner.process({
        "query":       query,
        "preferences": preferences,
        "read_books":  read_books
    })
    plan = result.get("plan", {})

    # If book_based, user already read the source — exclude it from results
    if plan.get("type") == "book_based":
        source = plan.get("value", "")
        if source and source not in read_books:
            read_books = read_books + [source]

    logging.info(f"PlannerNode: created {plan.get('type')} plan")

    return {
        "plan":             plan,
        "plan_type":        plan.get("type", "unknown"),
        "plan_value":       plan.get("value", query),
        "user_preferences": preferences,
        "read_books":       read_books
    }


# ── Node 2: Executor ─────────────────────────────────────────────────────────

async def executor_node(state: AgentState) -> Dict[str, Any]:
    logging.info(f"ExecutorNode: executing {state['plan_type']} plan")

    task = {
        "plan":        state["plan"],
        "query":       state["query"],
        "read_books":  state.get("read_books", []),
        "preferences": state.get("user_preferences", {})
    }

    result          = get_container().executor.process_task(task)
    recommendations = result.get("recommendations", [])

    # Secondary filter — belt-and-braces read_books check
    read_lower = [rb.lower() for rb in state.get("read_books", [])]
    filtered   = [r for r in recommendations if r.get("title", "").lower() not in read_lower]

    logging.info(
        f"ExecutorNode: {len(filtered)} recommendations "
        f"(filtered {len(recommendations) - len(filtered)} already-read)"
    )

    return {
        "search_results":  result.get("search_results", []),
        "recommendations": filtered
    }


# ── Node 3: Critic ───────────────────────────────────────────────────────────

async def critic_node(state: AgentState) -> Dict[str, Any]:
    logging.info(f"CriticNode: evaluating {len(state.get('recommendations', []))} recommendations")

    task = {
        "recommendations": state.get("recommendations", []),
        "query":           state["query"],
        "attempt":         state.get("attempt_number", 1)
    }

    result   = get_container().critic.process_task(task)
    verdict  = result.get("verdict", "FAIL")
    score    = result.get("score", 0)
    feedback = result.get("feedback", "")

    logging.info(f"CriticNode: verdict={verdict}, score={score}")

    return {
        "critic_verdict":  verdict,
        "critic_score":    score,
        "critic_feedback": feedback,
        "attempt_number":  state.get("attempt_number", 1) + 1
    }


# ── Node 4: Guardrail ────────────────────────────────────────────────────────

async def guardrail_node(state: AgentState) -> Dict[str, Any]:
    logging.info("GuardrailNode: scanning content")

    recommendations = state.get("recommendations", [])
    content = "\n".join([
        f"{r.get('title')} by {r.get('author')}: {r.get('description', '')}"
        for r in recommendations[:5]
    ])

    result            = get_container().guardrail.execute(content)
    passed            = result.content.get("passed", False)
    sanitized_content = result.content.get("sanitized_content", content)
    disclaimer_added  = result.content.get("disclaimer_added", False)
    violations        = result.content.get("violations", [])

    final_recommendations = recommendations
    if not passed and violations:
        logging.warning(f"GuardrailNode: blocked — {len(violations)} violations")
        final_recommendations = []

    if disclaimer_added:
        for rec in final_recommendations:
            rec["disclaimer"] = "Taste is personal - preview before buying"

    logging.info(f"GuardrailNode: passed={passed}, violations={len(violations)}")

    return {
        "final_recommendations": final_recommendations,
        "sanitized_content":     sanitized_content,
        "disclaimer_added":      disclaimer_added
    }


# ── Node 5: Output ───────────────────────────────────────────────────────────

async def output_node(state: AgentState) -> Dict[str, Any]:
    logging.info("OutputNode: preparing final response")
    return {
        "final_recommendations": state.get("final_recommendations", []),
        "critic_score":          state.get("critic_score", 0),
        "critic_feedback":       state.get("critic_feedback", ""),
        "attempts_used":         state.get("attempt_number", 1)
    }