import logging
from typing import Dict, Any
from app.graph.state import AgentState
from app.agents.planner import PlannerAgent
from app.agents.executor import ExecutorAgent
from app.agents.critic import CriticAgent
from app.tools.guardrail import GuardrailServer
from app.memory.episodic import EpisodicMemory


# Agent instances — injected by AgentService via set_agents()
planner   = None
executor  = None
critic    = None
guardrail = None


def set_agents(planner_agent, executor_agent, critic_agent):
    global planner, executor, critic
    planner  = planner_agent
    executor = executor_agent
    critic   = critic_agent


def set_guardrail(guardrail_server):
    global guardrail
    guardrail = guardrail_server


# ── Node 1: Planner ──────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> Dict[str, Any]:
    logging.info(f"PlannerNode: processing query: {state['query']}")

    query   = state["query"]
    user_id = state.get("user_id", "default_user")

    memory      = EpisodicMemory(user_id)
    preferences = memory.get_preferences()
    read_books  = memory.get_read_books()

    result = planner.process({"query": query, "preferences": preferences, "read_books": read_books})
    plan   = result.get("plan", {})

    # If book_based, the user already read that book — add it so executor filters it out
    if plan.get("type") == "book_based":
        source_title = plan.get("value", "")
        if source_title and source_title not in read_books:
            read_books = read_books + [source_title]

    logging.info(f"PlannerNode: created {plan.get('type')} plan")

    return {
        "plan":             plan,
        "plan_type":        plan.get("type", "unknown"),
        "plan_value":       plan.get("value", query),
        "user_preferences": preferences,
        "read_books":       read_books
    }


# ── Node 2: Executor ─────────────────────────────────────────────────────────

def executor_node(state: AgentState) -> Dict[str, Any]:
    logging.info(f"ExecutorNode: executing {state['plan_type']} plan")

    task = {
        "plan":        state["plan"],
        "query":       state["query"],
        "read_books":  state.get("read_books", []),
        "preferences": state.get("user_preferences", {})
    }

    result          = executor.process_task(task)
    recommendations = result.get("recommendations", [])

    # Secondary filter — catch anything still matching read_books
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

def critic_node(state: AgentState) -> Dict[str, Any]:
    logging.info(f"CriticNode: evaluating {len(state.get('recommendations', []))} recommendations")

    task = {
        "recommendations": state.get("recommendations", []),
        "query":           state["query"],
        "attempt":         state.get("attempt_number", 1)
    }

    result   = critic.process_task(task)
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

def guardrail_node(state: AgentState) -> Dict[str, Any]:
    logging.info("GuardrailNode: scanning content")

    recommendations = state.get("recommendations", [])

    content = "\n".join([
        f"{r.get('title')} by {r.get('author')}: {r.get('description', '')}"
        for r in recommendations[:5]
    ])

    result            = guardrail.execute(content)
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

def output_node(state: AgentState) -> Dict[str, Any]:
    logging.info("OutputNode: preparing final response")

    return {
        "final_recommendations": state.get("final_recommendations", []),
        "critic_score":          state.get("critic_score", 0),
        "critic_feedback":       state.get("critic_feedback", ""),
        "attempts_used":         state.get("attempt_number", 1)
    }