from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv

from services.truly_agentic_search import TrulyAgenticSearch
from services.llm import LLMService

load_dotenv()

app = FastAPI(title="Agentic Medical Search API")

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

class SearchResponse(BaseModel):
    answer: str
    sources: List[str]
    quality_assessment: Dict[str, str]
    execution_time: float
    iterations: int

@app.post("/search", response_model=SearchResponse)
async def agentic_search(request: SearchRequest):
    """Execute agentic medical search"""
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agentic-medical-search"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
