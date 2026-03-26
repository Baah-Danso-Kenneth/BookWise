from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    user_input: str
    plan: Dict[str, Any]
    search_results: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    critic_score: float
    critic_feedback: str
    attempt: int
    final_output: Dict[str, Any]
    errors: List[str]


