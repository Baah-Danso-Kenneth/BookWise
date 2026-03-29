import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END, START

from app.graph.state import AgentState
from app.graph.nodes import (
    planner_node, executor_node, critic_node, 
    guardrail_node, output_node
)


def route_after_critic(state: AgentState) -> str:
    """Conditional edge: decide to loop back or proceed to guardrail"""
    
    verdict = state.get("critic_verdict", "FAIL")
    attempt = state.get("attempt_number", 1)
    max_attempts = 3
    
    logging.info(f"RouteAfterCritic: verdict={verdict}, attempt={attempt}")
    
    if verdict == "FAIL":
        if attempt < max_attempts:
            logging.info(f"Routing back to executor (attempt {attempt}/{max_attempts})")
            return "executor"
        else:
            logging.info(f"Max attempts reached, proceeding to guardrail")
            return "guardrail"
    elif verdict == "REVISE":
        if attempt < max_attempts:
            logging.info(f"Routing back to executor for revision (attempt {attempt}/{max_attempts})")
            return "executor"
        else:
            logging.info(f"Max attempts reached, proceeding to guardrail")
            return "guardrail"
    else:  # PASS
        logging.info("Verdict PASS, proceeding to guardrail")
        return "guardrail"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph"""
    
    # Create graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("output", output_node)
    
    # Add edges
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "critic")
    
    # Conditional edge after critic (creates the loop!)
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "executor": "executor",      # Loop back
            "guardrail": "guardrail"      # Proceed to output
        }
    )
    
    # Fixed edges from guardrail to output to end
    graph.add_edge("guardrail", "output")
    graph.add_edge("output", END)
    
    return graph.compile()


def run_graph(
    query: str, 
    user_id: str = "default_user",
    conversation_id: str = None
) -> Dict[str, Any]:
    """Run the graph with initial state"""
    
    import uuid
    
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    initial_state: AgentState = {
        "query": query,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "plan": {},
        "plan_type": "",
        "plan_value": "",
        "search_results": [],
        "recommendations": [],
        "critic_score": 0.0,
        "critic_feedback": "",
        "critic_verdict": "",
        "attempt_number": 1,
        "user_preferences": {},
        "read_books": [],
        "final_recommendations": [],
        "sanitized_content": "",
        "disclaimer_added": False,
        "errors": [],
        "acp_messages": []
    }
    
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    
    return final_state