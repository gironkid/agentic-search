"""Test the API endpoint to verify references are included in the backend response"""

import asyncio
import aiohttp
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_api_references():
    query = "What are the latest treatments for type 2 diabetes?"

    async with aiohttp.ClientSession() as session:
        # Call the API endpoint
        url = "http://localhost:8000/search"
        payload = {
            "query": query,
            "use_tool_calling": True
        }

        print(f"🔍 Testing API with query: {query}")
        print("-" * 80)

        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()

                    # Check the answer field
                    answer = result.get('answer', '')

                    print("📋 API Response Answer Field:")
                    print(answer[:500] + "..." if len(answer) > 500 else answer)
                    print("-" * 80)

                    # Check for URLs in the answer
                    urls = re.findall(r'https?://[^\s\)]+', answer)

                    print("\n🔗 Checking for references in API response...")
                    if urls:
                        print(f"✅ Found {len(urls)} URLs in the API response:")
                        for i, url in enumerate(urls, 1):
                            print(f"   {i}. {url}")
                    else:
                        print("❌ No URLs found in the API response")

                    # Check for reference section
                    if "References" in answer and ("[1]" in answer or "1." in answer):
                        print("✅ References section found in API response")
                    else:
                        print("⚠️  No explicit references section found in API response")

                    # Check for citations in text
                    citations = re.findall(r'\[\d+\]', answer)
                    if citations:
                        print(f"✅ Found {len(set(citations))} unique citations in API response text")
                    else:
                        print("⚠️  No numbered citations found in API response text")

                    print("\n📊 Full API Response Structure:")
                    print(f"- Tools used: {result.get('tools_used', [])}")
                    print(f"- Iterations: {result.get('iterations', 0)}")
                    print(f"- Execution time: {result.get('execution_time', 0):.2f}s")

                else:
                    print(f"❌ API returned status {response.status}")
                    error_text = await response.text()
                    print(f"Error: {error_text}")

        except aiohttp.ClientConnectorError:
            print("❌ Could not connect to API. Make sure the server is running with:")
            print("   python main.py")
        except Exception as e:
            print(f"❌ Error calling API: {e}")

if __name__ == "__main__":
    asyncio.run(test_api_references())