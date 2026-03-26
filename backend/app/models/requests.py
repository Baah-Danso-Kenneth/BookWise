from pydantic import BaseModel
from typing import Optional


class RecommendationRequest(BaseModel):
    input: str # "I want a finance book" or "I read The Richest Man in Babylon"
    user_id: Optional[str] = None

