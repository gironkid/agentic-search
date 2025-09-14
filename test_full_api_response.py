"""Show the full API response with clickable references"""

import asyncio
import aiohttp
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def show_full_api_response():
    query = "What is the latest research on COVID-19 vaccines?"

    async with aiohttp.ClientSession() as session:
        url = "http://localhost:8000/search"
        payload = {
            "query": query,
            "use_tool_calling": True
        }

        print(f"🔍 Query: {query}")
        print("=" * 80)

        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()

                    # Show the full answer with references
                    answer = result.get('answer', '')

                    print("\n📋 FULL API RESPONSE (answer field):")
                    print("-" * 80)
                    print(answer)
                    print("-" * 80)

                    print(f"\n📊 API Metadata:")
                    print(f"- Tools used: {result.get('tools_used', [])}")
                    print(f"- Iterations: {result.get('iterations', 0)}")
                    print(f"- Execution time: {result.get('execution_time', 0):.2f}s")
                    print(f"- Method: {result.get('method', 'unknown')}")

                else:
                    print(f"❌ API returned status {response.status}")

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(show_full_api_response())