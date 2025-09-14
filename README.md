# Agentic Medical Search System

A truly agentic medical search system that uses LLM tool calling to intelligently search multiple medical databases and synthesize comprehensive answers.

## Features

- 🤖 **LLM-driven tool selection** - AI decides which tools to use
- 🔍 **Multiple data sources** - PubMed, FDA, ClinicalTrials.gov, Web
- 💊 **Drug interaction checking** - Comprehensive drug safety database
- 👶 **Pediatric dosing calculator** - Weight-based dosing calculations
- 🤰 **Pregnancy safety checker** - Trimester-specific guidance
- 🔄 **Dynamic iteration** - AI decides when enough information is gathered
- 📊 **Quality assessment** - Evaluates result quality from each source

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Test with CLI:
```bash
python cli.py "What is the treatment for migraine?"
```

4. Run API server:
```bash
python main.py
# API will be available at http://localhost:8000
```

## API Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={"query": "treatment for ankle sprain"}
)

print(response.json()["answer"])
```

## Required API Keys

- **OPENROUTER_API_KEY**: For medical synthesis (supports many models)
- **OPENAI_API_KEY**: For GPT-4o-mini tool calling
- **PUBMED_API_KEY** (optional): For better PubMed rate limits

## How It Works

1. Query analysis by LLM
2. Tool selection (minimum: PubMed + Web search)
3. Parallel tool execution
4. Quality assessment of results
5. LLM decides if more information needed
6. Synthesis of comprehensive answer with citations

## License

MIT
