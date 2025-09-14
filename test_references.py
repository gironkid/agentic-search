"""Test script to verify clickable references are included in responses"""

import asyncio
import os
from dotenv import load_dotenv
from services.truly_agentic_search import TrulyAgenticSearch
from services.llm import LLMService

# Load environment variables
load_dotenv()

async def test_search_with_references():
    # Initialize services
    llm_service = LLMService()
    search = TrulyAgenticSearch(llm_service, use_llm_tool_calling=True)

    # Test query
    query = "What are the latest treatments for type 2 diabetes?"

    print(f"🔍 Testing query: {query}")
    print("-" * 80)

    # Execute search
    results = await search.execute_agentic(query)

    # Display answer
    answer = results.get('answer', '')
    print("\n📋 Answer:")
    print(answer)
    print("-" * 80)

    # Check for references
    print("\n🔗 Checking for references...")

    # Look for URLs in the answer
    import re
    urls = re.findall(r'https?://[^\s\)]+', answer)

    if urls:
        print(f"✅ Found {len(urls)} URLs in the answer:")
        for i, url in enumerate(urls, 1):
            print(f"   {i}. {url}")
    else:
        print("❌ No URLs found in the answer")

    # Check for reference section
    if "## References" in answer or "References:" in answer or "[1]" in answer:
        print("✅ References section found")
    else:
        print("⚠️  No explicit references section found")

    # Check for citations in text
    citations = re.findall(r'\[\d+\]', answer)
    if citations:
        print(f"✅ Found {len(set(citations))} unique citations in text")
    else:
        print("⚠️  No numbered citations found in text")

    print("\n✨ Test complete!")

    # Cleanup
    await llm_service.cleanup()
    await search.cleanup()

if __name__ == "__main__":
    asyncio.run(test_search_with_references())