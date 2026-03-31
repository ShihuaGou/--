import logging
from fastapi import FastAPI
from pydantic import BaseModel
from agent.agent_core import ULMMarAgent
from agent.model_loader import ModelLoader
from agent.config import MODEL_ID, LOGS_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    try:
        logger.info("正在初始化ULM-MAR代理...")
        loader = ModelLoader(model_name_or_path=MODEL_ID)
        loader.load_model()
        agent = ULMMarAgent(model_loader=loader)
        logger.info("ULM-MAR代理初始化完成")
    except Exception as e:
        logger.error(f"代理初始化失败: {e}")
        raise

@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "agent_loaded": agent is not None}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if agent is None:
        raise RuntimeError("Agent has not been initialized.")
    try:
        output, summary = agent.process_query(request.input_text, request.context)
        return QueryResponse(output_text=output, memory_summary=summary)
    except Exception as e:
        logger.error(f"查询处理失败: {e}")
        raise
