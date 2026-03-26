from pydantic import BaseModel
from typing import List, Optional

class BookRecommendation(BaseModel):
    title: str
    author: str
    reason: str
    match_score: float
    description: Optional[str] = None



class RecommendationResponse(BaseModel):
    type: str #"category" or "book-base"
    source: Optional[str] 
    recommendations: List[BookRecommendation]
    disclaimer: str = "Taste is personal. Your experience may vary"

