import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_pipeline import RAGPipeline


app = FastAPI(title="Legal RAG Assistant API")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Lifespan — loads pipeline once at startup, cleans up on shutdown
pipeline: RAGPipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Starting up — loading RAGPipeline...")
    try:
        pipeline = RAGPipeline()
        logger.info("RAGPipeline loaded successfully.")
    except Exception as e:
        logger.error("Failed to initialize RAGPipeline: %s", e)
        pipeline = None
    yield
    logger.info("Shutting down.")


# App
app = FastAPI(
    title="Legal RAG Assistant API",
    description="Bilingual (Arabic/English) RAG pipeline for Lebanese labor law.",
    version="1.0.0",
    lifespan=lifespan
)

# Request / response models
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The legal question to answer (Arabic or English)."
    )

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

# Routes
@app.get("/health")
async def health_check():
    """Returns API status and whether the pipeline is loaded."""
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    """
    Submit a legal question in Arabic or English.
    Returns a grounded answer and source citations.
    """
    if pipeline is None:
        logger.error("Request received but pipeline is not initialized.")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable — pipeline failed to load at startup."
        )

    logger.info("Received question: %s", request.question[:80])

    try:
        result = pipeline.generate_answer(request.question)
    except Exception as e:
        logger.error("Pipeline error on question '%s': %s", request.question[:80], e)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your request."
        )

    logger.info("Response ready — answer length: %d chars", len(result["answer"]))
    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"]
    )        