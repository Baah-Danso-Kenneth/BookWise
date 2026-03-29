import logging
from typing import Dict, Any

from app.memory.base import MemoryBase


class ProceduralMemory(MemoryBase):
    """
    Procedural Memory - Stores best practices, rules, and guidelines.
    
    This memory is injected into prompts to guide agent behavior.
    """
    
    def __init__(self):
        self.rules = self._load_rules()
    
    def _load_rules(self) -> Dict[str, str]:
        """Load procedural rules"""
        return {
            "recommendation": """
                When recommending books:
                1. Always explain why the book matches the user's interest
                2. Include author and brief description
                3. If user has read similar books, mention that
                4. Never recommend books the user has already read
            """,
            
            "critique": """
                When evaluating recommendations:
                1. Check if books match the query intent
                2. Ensure diversity in recommendations
                3. Verify descriptions are meaningful
                4. Flag any books the user has already read
            """,
            
            "search": """
                When searching for books:
                1. Use specific search queries
                2. Prefer known authors and publishers
                3. Filter out duplicates
                4. Prioritize recent publications when relevant
            """,
            
            "safety": """
                Safety rules:
                1. Never recommend inappropriate content
                2. No spoilers in descriptions
                3. Include disclaimer for personal taste
                4. Flag controversial books with context
            """
        }
    
    def get_rule(self, rule_name: str) -> str:
        """Get a specific rule"""
        return self.rules.get(rule_name, "")
    
    def get_all_rules(self) -> str:
        """Get all rules as formatted string"""
        return "\n\n".join([f"### {key.upper()} RULES:\n{value}" for key, value in self.rules.items()])
    
    def get_prompt_injection(self, context: str = "recommendation") -> str:
        """Get formatted rules for prompt injection"""
        return f"""
PROCEDURAL MEMORY - BEST PRACTICES:

{self.get_rule(context)}

FOLLOW THESE RULES CAREFULLY.
"""
    
    def store(self, key: str, value: Any) -> bool:
        """Add a new rule"""
        self.rules[key] = value
        return True
    
    def retrieve(self, key: str) -> Any:
        return self.get_rule(key)
    
    def update(self, key: str, value: Any) -> bool:
        return self.store(key, value)
    
    def delete(self, key: str) -> bool:
        if key in self.rules:
            del self.rules[key]
            return True
        return False