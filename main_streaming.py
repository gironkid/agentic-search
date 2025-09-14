from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import asyncio
from dotenv import load_dotenv

from services.truly_agentic_search import TrulyAgenticSearch
from services.llm import LLMService

load_dotenv()

app = FastAPI(title="Agentic Medical Search API with Streaming")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
llm_service = LLMService()

class SearchRequest(BaseModel):
    query: str
    use_tool_calling: bool = True
    stream: bool = False

class SearchResponse(BaseModel):
    answer: str
    sources: List[str]
    quality_assessment: Dict[str, str]
    execution_time: float
    iterations: int

async def event_generator(query: str, use_tool_calling: bool):
    """Generate SSE events for streaming progress"""
    search = TrulyAgenticSearch(
        llm_service=llm_service,
        use_llm_tool_calling=use_tool_calling
    )

    # Progress callback that yields events
    async def progress_callback(message: str, data: dict = None):
        event = {
            "type": "progress",
            "message": message,
            "data": data or {}
        }
        return f"data: {json.dumps(event)}\n\n"

    try:
        # Send initial event
        yield f"data: {json.dumps({'type': 'start', 'message': 'Starting search...'})}\n\n"

        # Track tool calls
        tool_calls = []

        # Modified search execution with progress tracking
        # This would need modification in TrulyAgenticSearch to support callbacks
        # For now, simulating progress events

        yield f"data: {json.dumps({'type': 'tool_call', 'tool': 'analyze_query', 'message': 'Analyzing query...'})}\n\n"
        await asyncio.sleep(0.5)

        yield f"data: {json.dumps({'type': 'tool_call', 'tool': 'search_pubmed', 'message': 'Searching PubMed...'})}\n\n"
        await asyncio.sleep(1)

        yield f"data: {json.dumps({'type': 'tool_call', 'tool': 'web_search', 'message': 'Searching web...'})}\n\n"
        await asyncio.sleep(1)

        # Execute actual search
        result = await search.execute_agentic(query)

        # Send final result
        yield f"data: {json.dumps({'type': 'complete', 'result': result})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

@app.post("/search")
async def agentic_search(request: SearchRequest):
    """Execute agentic medical search with optional streaming"""

    if request.stream:
        # Return streaming response
        return StreamingResponse(
            event_generator(request.query, request.use_tool_calling),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
    else:
        # Regular non-streaming response
        try:
            search = TrulyAgenticSearch(
                llm_service=llm_service,
                use_llm_tool_calling=request.use_tool_calling
            )

            result = await search.execute_agentic(request.query)

            return SearchResponse(
                answer=result["answer"],
                sources=result.get("tools_used", []),
                quality_assessment=result.get("quality_assessment", {}),
                execution_time=result.get("execution_time", 0),
                iterations=result.get("iterations", 0)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/stream")
async def stream_search(query: str):
    """GET endpoint for easy SSE testing"""
    return StreamingResponse(
        event_generator(query, use_tool_calling=True),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agentic-medical-search"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)