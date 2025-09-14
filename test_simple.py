#!/usr/bin/env python
"""
Simple test script for standalone agentic medical search
"""

import asyncio
import sys
from dotenv import load_dotenv
from services.truly_agentic_search import TrulyAgenticSearch
from services.llm import LLMService

load_dotenv()

async def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the treatment for ankle sprain?"

    print(f"\n🔍 Testing query: {query}\n")

    # Initialize services
    llm = LLMService()
    search = TrulyAgenticSearch(llm, use_llm_tool_calling=True)

    # Progress callback
    async def show_progress(msg):
        print(f"  {msg}")

    # Execute search
    result = await search.execute_agentic(query, show_progress)

    # Show results
    print(f"\n📋 Answer:\n{result['answer'][:500]}...")
    print(f"\n✅ Tools used: {', '.join(result.get('tools_used', []))}")
    print(f"⏱️  Time: {result.get('execution_time', 0):.1f}s")
    print(f"🔄 Iterations: {result.get('iterations', 0)}")

if __name__ == "__main__":
    asyncio.run(main())