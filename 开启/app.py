from fastapi import FastAPI
from pydantic import BaseModel
from agent.agent_core import ULMMarAgent
from agent.model_loader import ModelLoader
from agent.config import MODEL_ID

app = FastAPI(title="ULM-MARv2 Agent")
agent: ULMMarAgent | None = None

class QueryRequest(BaseModel):
    input_text: str
    context: str | None = None

class QueryResponse(BaseModel):
    output_text: str
    memory_summary: str | None = None

@app.on_event("startup")
def startup_event() -> None:
    global agent
    loader = ModelLoader(model_name_or_path=MODEL_ID)
    loader.load_model()
    agent = ULMMarAgent(model_loader=loader)

@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "agent_loaded": agent is not None}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if agent is None:
        raise RuntimeError("Agent has not been initialized.")
    output, summary = agent.process_query(request.input_text, request.context)
    return QueryResponse(output_text=output, memory_summary=summary)
