"""
Comprehensive Agentic Medical Search API
Includes all CLI functionality: streaming, all tools, multiple modes
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal
import os
import json
import asyncio
import time
from datetime import datetime
from dotenv import load_dotenv

from services.truly_agentic_search import TrulyAgenticSearch
from services.optimized_search_fixed import OptimizedMedicalSearchFixed
from services.llm import LLMService

load_dotenv()

app = FastAPI(
    title="Agentic Medical Search API - Complete Edition",
    description="Comprehensive medical search with all tools and streaming support",
    version="2.0.0"
)

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
llm_service = LLMService()

# Request/Response Models
class SearchRequest(BaseModel):
    query: str
    mode: Literal["agentic", "optimized", "simple"] = "agentic"
    use_tool_calling: bool = True
    stream: bool = False
    tools: Optional[List[str]] = None  # Specific tools to use
    max_iterations: int = 5
    verbose: bool = False

class ToolCallRequest(BaseModel):
    tool: str
    params: Dict[str, Any]

class SearchResponse(BaseModel):
    answer: str
    sources: List[str]
    quality_assessment: Dict[str, str]
    execution_time: float
    iterations: int
    tools_used: List[str]
    was_refined: bool = False
    raw_results: Optional[Dict] = None

class ToolsResponse(BaseModel):
    tools: List[Dict[str, str]]
    categories: Dict[str, List[str]]

# Available tools documentation
AVAILABLE_TOOLS = {
    "search_pubmed": {
        "name": "PubMed Search",
        "description": "Search medical research articles from PubMed/NCBI",
        "category": "research"
    },
    "search_fda": {
        "name": "FDA Database",
        "description": "Search FDA drug information and approvals",
        "category": "regulatory"
    },
    "search_clinical_trials": {
        "name": "Clinical Trials",
        "description": "Search ongoing and completed clinical trials",
        "category": "research"
    },
    "web_search": {
        "name": "Web Search",
        "description": "General medical web search via Tavily",
        "category": "general"
    },
    "calculate_pediatric_dose": {
        "name": "Pediatric Dosing",
        "description": "Calculate weight-based pediatric medication dosing",
        "category": "calculator"
    },
    "check_drug_interactions": {
        "name": "Drug Interactions",
        "description": "Check for drug-drug interactions",
        "category": "safety"
    },
    "check_pregnancy_safety": {
        "name": "Pregnancy Safety",
        "description": "Check medication safety during pregnancy",
        "category": "safety"
    },
    "check_renal_dosing": {
        "name": "Renal Dosing",
        "description": "Adjust medication dosing for kidney function",
        "category": "calculator"
    },
    "analyze_lab_values": {
        "name": "Lab Interpretation",
        "description": "Interpret laboratory test results",
        "category": "diagnostic"
    },
    "get_disease_statistics": {
        "name": "Disease Statistics",
        "description": "Get prevalence and mortality statistics",
        "category": "epidemiology"
    },
    "calculate_medical_score": {
        "name": "Medical Scores",
        "description": "Calculate clinical scores (CHADS2, GCS, etc)",
        "category": "calculator"
    },
    "search_clinical_guidelines": {
        "name": "Clinical Guidelines",
        "description": "Search clinical practice guidelines",
        "category": "guidelines"
    },
    "get_drug_information": {
        "name": "Drug Information",
        "description": "Detailed drug monograph information",
        "category": "reference"
    }
}

async def event_generator(request: SearchRequest):
    """Generate SSE events for streaming search progress"""
    try:
        # Initialize appropriate search engine
        if request.mode == "agentic":
            search = TrulyAgenticSearch(
                llm_service=llm_service,
                use_llm_tool_calling=request.use_tool_calling
            )
        else:
            search = OptimizedMedicalSearchFixed(llm_service=llm_service)

        # Send initial event
        yield f"data: {json.dumps({'type': 'start', 'message': f'Starting {request.mode} search...', 'timestamp': datetime.now().isoformat()})}\n\n"

        # Track progress
        events_sent = []
        start_time = time.time()

        # Progress callback for agentic mode
        async def progress_callback(message: str):
            event = {
                "type": "progress",
                "message": message,
                "timestamp": datetime.now().isoformat()
            }

            # Detect tool calls
            if "Executing" in message or "Searching" in message:
                event["type"] = "tool_call"
                # Extract tool name if possible
                if ":" in message:
                    tool = message.split(":")[0].replace("Executing", "").strip()
                    event["tool"] = tool

            # Detect iterations
            elif "Iteration" in message:
                event["type"] = "iteration"

            # Detect quality assessment
            elif "quality" in message.lower():
                event["type"] = "quality"

            events_sent.append(event)
            return f"data: {json.dumps(event)}\n\n"

        # Initialize result to avoid UnboundLocalError
        result = {}

        # Execute search based on mode
        if request.mode == "agentic":
            # Stream progress during execution
            result_queue = asyncio.Queue()

            async def execute_with_streaming():
                result = await search.execute_agentic(
                    request.query,
                    streaming_callback=lambda msg: result_queue.put_nowait(msg)
                )
                await result_queue.put({"type": "result", "data": result})

            # Start execution
            task = asyncio.create_task(execute_with_streaming())

            # Stream events
            while True:
                try:
                    item = await asyncio.wait_for(result_queue.get(), timeout=0.5)
                    if isinstance(item, str):
                        event = await progress_callback(item)
                        yield event
                    elif isinstance(item, dict) and item.get("type") == "result":
                        result = item["data"]
                        break
                except asyncio.TimeoutError:
                    if task.done():
                        break
                    continue

        elif request.mode == "optimized":
            # Optimized mode with tool tracking
            yield f"data: {json.dumps({'type': 'tool_selection', 'message': 'Analyzing query and selecting tools...'})}\n\n"

            tools = request.tools if request.tools else await search._determine_tools(request.query)
            yield f"data: {json.dumps({'type': 'tools_selected', 'tools': tools})}\n\n"

            # Execute tools in parallel
            for tool in tools:
                yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool, 'message': f'Executing {tool}...'})}\n\n"

            results = await search.search_parallel(request.query, tools)

            yield f"data: {json.dumps({'type': 'synthesis', 'message': 'Synthesizing results...'})}\n\n"
            answer = await search._synthesize_results(request.query, results)

            result = {
                'answer': answer,
                'tools_used': tools,
                'raw_results': results if request.verbose else None,
                'execution_time': time.time() - start_time,
                'iterations': 1
            }

        else:  # simple mode
            # Just use basic search
            yield f"data: {json.dumps({'type': 'search', 'message': 'Performing simple search...'})}\n\n"
            search = OptimizedMedicalSearchFixed(llm_service=llm_service)
            result = await search.execute_with_reasoning(request.query)

        # Send final result
        final_event = {
            "type": "complete",
            "result": {
                "answer": result.get("answer", ""),
                "sources": result.get("tools_used", []),
                "quality_assessment": result.get("quality_assessment", {}),
                "execution_time": result.get("execution_time", time.time() - start_time),
                "iterations": result.get("iterations", 1),
                "tools_used": result.get("tools_used", []),
                "was_refined": result.get("was_refined", False)
            },
            "timestamp": datetime.now().isoformat()
        }

        if request.verbose:
            final_event["result"]["raw_results"] = result.get("raw_results", {})

        yield f"data: {json.dumps(final_event)}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'timestamp': datetime.now().isoformat()})}\n\n"

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Main search endpoint with multiple modes and streaming support

    Modes:
    - agentic: Truly agentic search with iterative refinement and quality assessment
    - optimized: Optimized parallel search with smart tool selection
    - simple: Basic search with minimal processing
    """

    if request.stream:
        # Return SSE streaming response
        return StreamingResponse(
            event_generator(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    # Non-streaming response
    try:
        start_time = time.time()

        if request.mode == "agentic":
            search = TrulyAgenticSearch(
                llm_service=llm_service,
                use_llm_tool_calling=request.use_tool_calling
            )
            result = await search.execute_agentic(request.query)

        elif request.mode == "optimized":
            search = OptimizedMedicalSearchFixed(llm_service=llm_service)

            if request.tools:
                # Use specific tools if provided
                results = await search.search_parallel(request.query, request.tools)
                answer = await search._synthesize_results(request.query, results)
                result = {
                    'answer': answer,
                    'tools_used': request.tools,
                    'raw_results': results if request.verbose else None,
                    'execution_time': time.time() - start_time,
                    'iterations': 1
                }
            else:
                result = await search.execute_with_reasoning(request.query)

        else:  # simple mode
            search = OptimizedMedicalSearchFixed(llm_service=llm_service)
            tools = ['search_pubmed', 'web_search']  # Basic tools only
            results = await search.search_parallel(request.query, tools)
            answer = await search._synthesize_results(request.query, results)
            result = {
                'answer': answer,
                'tools_used': tools,
                'execution_time': time.time() - start_time,
                'iterations': 1
            }

        return SearchResponse(
            answer=result.get("answer", ""),
            sources=result.get("tools_used", []),
            quality_assessment=result.get("quality_assessment", {}),
            execution_time=result.get("execution_time", 0),
            iterations=result.get("iterations", 0),
            tools_used=result.get("tools_used", []),
            was_refined=result.get("was_refined", False),
            raw_results=result.get("raw_results") if request.verbose else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/stream")
async def stream_search(
    query: str = Query(..., description="Medical question to search"),
    mode: str = Query("agentic", description="Search mode: agentic, optimized, or simple"),
    use_tool_calling: bool = Query(True, description="Use LLM tool calling for agentic mode"),
    verbose: bool = Query(False, description="Include raw results in response")
):
    """
    GET endpoint for SSE streaming - easier to test in browser

    Example: /search/stream?query=treatment for migraine
    """
    request = SearchRequest(
        query=query,
        mode=mode,
        use_tool_calling=use_tool_calling,
        stream=True,
        verbose=verbose
    )

    return StreamingResponse(
        event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@app.post("/tool/execute")
async def execute_tool(request: ToolCallRequest):
    """
    Execute a specific tool directly

    Useful for testing individual tools or building custom workflows
    """
    try:
        search = OptimizedMedicalSearchFixed(llm_service=llm_service)

        # Execute the requested tool
        result = await search._execute_tool_safe(request.tool, request.params.get("query", ""))

        return {
            "tool": request.tool,
            "params": request.params,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools", response_model=ToolsResponse)
async def list_tools():
    """
    List all available tools and their descriptions
    """
    tools_list = [
        {"name": key, **value}
        for key, value in AVAILABLE_TOOLS.items()
    ]

    # Group by category
    categories = {}
    for tool_key, tool_info in AVAILABLE_TOOLS.items():
        category = tool_info["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(tool_key)

    return ToolsResponse(
        tools=tools_list,
        categories=categories
    )

@app.get("/tools/{tool_name}")
async def get_tool_info(tool_name: str):
    """
    Get detailed information about a specific tool
    """
    if tool_name not in AVAILABLE_TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    return {
        "name": tool_name,
        **AVAILABLE_TOOLS[tool_name],
        "examples": get_tool_examples(tool_name)
    }

def get_tool_examples(tool_name: str) -> List[str]:
    """Get example queries for a specific tool"""
    examples = {
        "search_pubmed": [
            "Latest research on CRISPR gene therapy",
            "Metformin mechanism of action studies"
        ],
        "search_fda": [
            "FDA approval status of Ozempic",
            "Black box warnings for antidepressants"
        ],
        "search_clinical_trials": [
            "Phase 3 trials for Alzheimer's disease",
            "CAR-T therapy trials recruiting patients"
        ],
        "calculate_pediatric_dose": [
            "Amoxicillin dose for 25kg child",
            "Ibuprofen dosing for 8 year old"
        ],
        "check_drug_interactions": [
            "Interactions between warfarin and aspirin",
            "Can I take ibuprofen with lisinopril?"
        ],
        "check_pregnancy_safety": [
            "Is metformin safe during pregnancy?",
            "Pregnancy category for sertraline"
        ],
        "analyze_lab_values": [
            "Interpret troponin 0.5 ng/mL",
            "What does elevated WBC 15000 mean?"
        ]
    }
    return examples.get(tool_name, [])

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "agentic-medical-search",
        "version": "2.0.0",
        "features": {
            "streaming": True,
            "tool_calling": True,
            "modes": ["agentic", "optimized", "simple"],
            "tools_count": len(AVAILABLE_TOOLS)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats")
async def get_stats():
    """Get service statistics and performance metrics"""
    # This would connect to a monitoring system in production
    return {
        "uptime": "calculating...",
        "total_requests": 0,
        "active_searches": 0,
        "average_response_time": 0,
        "cache_hit_rate": 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/docs/examples")
async def get_examples():
    """Get example API calls for documentation"""
    return {
        "simple_search": {
            "method": "POST",
            "endpoint": "/search",
            "body": {
                "query": "What is the treatment for migraine?",
                "mode": "simple"
            }
        },
        "agentic_search_streaming": {
            "method": "POST",
            "endpoint": "/search",
            "body": {
                "query": "Compare ACE inhibitors vs ARBs for hypertension",
                "mode": "agentic",
                "stream": True,
                "use_tool_calling": True
            }
        },
        "specific_tools": {
            "method": "POST",
            "endpoint": "/search",
            "body": {
                "query": "Latest clinical trials for diabetes",
                "mode": "optimized",
                "tools": ["search_clinical_trials", "search_pubmed"]
            }
        },
        "direct_tool_execution": {
            "method": "POST",
            "endpoint": "/tool/execute",
            "body": {
                "tool": "calculate_pediatric_dose",
                "params": {
                    "query": "Amoxicillin dose for 25kg child"
                }
            }
        },
        "sse_streaming": {
            "method": "GET",
            "endpoint": "/search/stream?query=drug interactions warfarin aspirin&mode=agentic",
            "description": "Open in browser or use EventSource in JavaScript"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)