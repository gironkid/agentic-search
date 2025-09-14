# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic Medical Search System - An LLM-driven search system that intelligently queries multiple medical databases (PubMed, FDA, ClinicalTrials.gov) using tool calling to provide comprehensive medical information with citations.

## Development Commands

### Running the Application
```bash
# API Server (runs on port 8000)
python main.py

# CLI with Rich UI
python cli.py "your medical query"

# Simple test script
python test_simple.py "your medical query"
```

### Testing
```bash
# No formal test suite found - use test_simple.py for basic testing
python test_simple.py
```

### Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt
```

## Architecture

### Core Components

1. **services/truly_agentic_search.py** - Main agentic search orchestrator that:
   - Uses LLM tool calling (GPT-4o-mini) to decide which tools to use
   - Evaluates result quality and automatically retries
   - Inherits from OptimizedMedicalSearchFixed for base functionality
   - Implements iterative search with quality assessment

2. **services/llm.py** - LLM service wrapper supporting multiple providers via OpenRouter

3. **services/optimized_search_fixed.py** - Base search implementation with memory management and parallel tool execution

4. **Tool Clients** in services/:
   - pubmed_client.py - PubMed article search
   - clinical_trials_client.py - ClinicalTrials.gov search
   - tavily_search_client.py - Web search via Tavily
   - fda_api_service.py (optional) - FDA drug information

### API Structure

FastAPI server (main.py) with single endpoint:
- POST /search - Executes agentic medical search with optional tool calling

### Key Features

- **LLM Tool Calling**: Uses OpenAI's function calling to let the AI decide which tools to use
- **Quality Assessment**: Evaluates each tool's results and decides if more information is needed
- **Iterative Search**: Continues searching until quality threshold met or max iterations reached
- **Parallel Execution**: Tools run concurrently for better performance

## Environment Variables

Required in .env:
- OPENROUTER_API_KEY - For medical synthesis LLM
- OPENAI_API_KEY - For GPT-4o-mini tool calling
- PUBMED_API_KEY (optional) - For better PubMed rate limits
- TAVILY_API_KEY - For web search

## Configuration

config.yaml contains system prompts for different modes (medical, technical, creative, etc.) and default model settings.

## Key Implementation Details

- SearchState dataclass tracks query evolution and quality scores
- ResultQuality enum (EXCELLENT, GOOD, POOR, EMPTY) drives search iteration
- Tool selection minimum: Always uses PubMed + Web search, adds others as needed
- 30-second timeout for thorough search with 5 max iterations