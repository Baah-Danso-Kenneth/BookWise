import logging
from typing import List, Dict, Any
import numpy as np

from app.tools.base import MCPToolServer, MCPToolDescriptor, MCPToolResult, MCPToolStatus


class TasteAnalyzerServer(MCPToolServer):
    """
    MCP-compliant taste analysis tool.
    Compares books against user preferences.
    """
    
    def _register(self) -> MCPToolDescriptor:
        return MCPToolDescriptor(
            name="TasteAnalyzer",
            version="1.0.0",
            description="Analyzes how well books match user preferences",
            input_schema={
                "type": "object",
                "properties": {
                    "books": {"type": "array", "description": "List of books to analyze"},
                    "user_preferences": {"type": "object", "description": "User taste profile"}
                },
                "required": ["books"]
            }
        )
    
    def execute(self, books: List[Dict], user_preferences: Dict = None) -> MCPToolResult:
        """Analyze books against user preferences"""
        logging.info(f"TasteAnalyzer: analyzing {len(books)} books")
        
        try:
            analyzed_books = []
            for book in books:
                score = self._calculate_match_score(book, user_preferences or {})
                analyzed_books.append({
                    **book,
                    "match_score": score,
                    "reason": self._generate_reason(book, score)
                })
            
            # Sort by match score
            analyzed_books.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            
            return MCPToolResult(
                tool_name="TasteAnalyzer",
                status=MCPToolStatus.SUCCESS,
                content={
                    "analyzed_books": analyzed_books,
                    "total_analyzed": len(analyzed_books)
                },
                metadata={
                    "has_preferences": user_preferences is not None
                }
            )
            
        except Exception as e:
            logging.error(f"TasteAnalyzer failed: {e}")
            return MCPToolResult(
                tool_name="TasteAnalyzer",
                status=MCPToolStatus.ERROR,
                content={"error": str(e)},
                metadata={}
            )
    
    def _calculate_match_score(self, book: Dict, preferences: Dict) -> float:
        """Calculate match score based on preferences"""
        # Simple scoring for now
        # Will be enhanced with embeddings later
        base_score = book.get("score", 0.5)
        
        # If preferences exist, adjust score
        if preferences:
            liked_topics = preferences.get("liked_topics", [])
            book_title = book.get("title", "").lower()
            
            for topic in liked_topics:
                if topic.lower() in book_title:
                    base_score = min(1.0, base_score + 0.1)
        
        return base_score
    
    def _generate_reason(self, book: Dict, score: float) -> str:
        """Generate human-readable reason"""
        if score > 0.8:
            return "Excellent match for your taste"
        elif score > 0.6:
            return "Good match based on your interests"
        elif score > 0.4:
            return "May be of interest"
        else:
            return "Consider if you're exploring new genres"