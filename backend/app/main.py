from fastapi import FastAPI, HTTPException
from app.services.agent_service import AgentService
from app.models.requests import RecommendationRequest

app = FastAPI(title="BookWise API", version="1.0.0")
agent_service = AgentService()


@app.get("/")
async def root():
    return {"message": "BookWise API is running"}


@app.post("/recommend")
async def recommend(request: dict):
    query   = request.get("query", "")
    user_id = request.get("user_id", "default_user")

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    return await agent_service.recommend(query, user_id)


@app.post("/rate")
async def rate_book(request: dict):
    user_id = request.get("user_id", "default_user")
    title   = request.get("title", "")
    rating  = request.get("rating", 0)

    if not title or not (1 <= rating <= 5):
        raise HTTPException(status_code=400, detail="title and rating (1-5) are required")

    return await agent_service.rate_book(user_id, title, rating)


@app.get("/history/{user_id}")
async def get_history(user_id: str):
    return await agent_service.get_history(user_id)