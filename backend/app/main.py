from fastapi import FastAPI
from app.services.agent_service import AgentService


app = FastAPI()
agent_service = AgentService()


@app.get("/")
def root():
    return {"message": "BookWise API is up and running!!!"}


@app.post("/recommend")
def recommend(request: dict):
    query = request.get("query", "")
    result = agent_service.recommend(query)

    return result