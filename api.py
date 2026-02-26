from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import traceback
from rag_pipeline import RAGPipeline

app = FastAPI(title="Legal RAG Assistant API")

# Initialize the RAG pipeline 
try:
    pipeline = RAGPipeline()
except Exception as e:
    print("Failed to initialize RAGPipeline:", e)
    pipeline = None


@app.post("/ask")
async def ask_endpoint(request: Request):
    if pipeline is None:
        return JSONResponse(
            status_code=500,
            content={"error": "RAG pipeline not initialized. Check logs."}
        )

    try:
        data = await request.json()
        question = data.get("question", "").strip()

        if not question:
            return JSONResponse(
                content={"answer": "No question provided.", "sources": []}
            )

        # Generate answer using the pipeline
        result = pipeline.generate_answer(question, k=3)

        # Ensure result has 'answer' and 'sources'
        answer = result.get("answer", "No answer returned.")
        sources = result.get("sources", [])

        return {"answer": answer, "sources": sources}

    except Exception as e:
        # Catch any exception and return traceback for debugging
        tb = traceback.format_exc()
        print(tb)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": tb}
        )