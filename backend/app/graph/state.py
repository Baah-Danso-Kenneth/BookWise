from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    """State that flows through the LangGraph"""
    
    # Input
    query: str
    user_id: str
    conversation_id: str
    
    # Planning
    plan: Dict[str, Any]
    plan_type: str
    plan_value: str
    
    # Execution
    search_results: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    
    # Review
    critic_score: float
    critic_feedback: str
    critic_verdict: str
    attempt_number: int
    
    # Memory
    user_preferences: Dict[str, Any]
    read_books: List[str]
    
    # Output
    final_recommendations: List[Dict[str, Any]]
    sanitized_content: str
    disclaimer_added: bool
    
    # Observability
    errors: List[str]
    acp_messages: List[Dict]