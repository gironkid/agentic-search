"""
Tavily AI-optimized web search client for medical information.
"""

import os
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class TavilySearchClient:
    """
    Client for Tavily AI-optimized web search.
    Provides high-quality, AI-curated search results.
    """
    
    def __init__(self):
        self.api_key = os.getenv('TAVILY_API_KEY')
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        self.base_url = "https://api.tavily.com"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search(self, query: str, max_results: int = 10, search_depth: str = "advanced") -> Dict[str, Any]:
        """
        Search using Tavily's AI-optimized search.
        
        Args:
            query: Search query
            max_results: Maximum number of results (default 10)
            search_depth: "basic" or "advanced" (advanced provides better quality)
            
        Returns:
            Dictionary with search results
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            url = f"{self.base_url}/search"
            
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_answer": True,  # Get AI-generated answer
                "include_raw_content": False,  # Don't need full HTML
                "include_domains": [  # Prioritize medical sources
                    "pubmed.ncbi.nlm.nih.gov",
                    "nih.gov",
                    "who.int",
                    "cdc.gov",
                    "mayoclinic.org",
                    "nejm.org",
                    "thelancet.com",
                    "bmj.com",
                    "jamanetwork.com",
                    "nature.com",
                    "sciencedirect.com"
                ]
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Format results
                    results = []
                    for item in data.get('results', []):
                        results.append({
                            'title': item.get('title', ''),
                            'snippet': item.get('content', ''),
                            'url': item.get('url', ''),
                            'score': item.get('score', 0),
                            'published_date': item.get('published_date', '')
                        })
                    
                    return {
                        'query': query,
                        'answer': data.get('answer', ''),  # AI-generated summary
                        'total_results': len(results),
                        'results': results,
                        'query_id': data.get('query_id', ''),
                        'status': 'success'
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Tavily API error: {response.status} - {error_text}")
                    return {
                        'query': query,
                        'error': f"API error: {response.status}",
                        'results': [],
                        'status': 'error'
                    }
                    
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return {
                'query': query,
                'error': str(e),
                'results': [],
                'status': 'error'
            }
    
    async def search_medical(self, query: str) -> Dict[str, Any]:
        """
        Medical-focused search with enhanced medical context.
        """
        # Add medical context to query
        medical_query = f"medical research evidence {query} latest studies guidelines"
        return await self.search(medical_query, search_depth="advanced")
    
    async def search_with_context(self, query: str, context: str) -> Dict[str, Any]:
        """
        Search with additional context for better results.
        """
        enhanced_query = f"{context} {query}"
        return await self.search(enhanced_query, search_depth="advanced")


async def test_tavily_search():
    """
    Test the Tavily search client.
    """
    print("\n" + "="*60)
    print("TESTING TAVILY AI SEARCH")
    print("="*60)
    
    async with TavilySearchClient() as client:
        # Test 1: HCC in biliary atresia
        print("\n[TEST 1] HCC risk in biliary atresia")
        results = await client.search_medical(
            "hepatocellular carcinoma HCC risk biliary atresia latest research 2023 2024"
        )
        
        print(f"\nStatus: {results.get('status')}")
        print(f"Results found: {results.get('total_results', 0)}")
        
        if results.get('answer'):
            print(f"\nAI Summary:\n{results['answer'][:500]}...")
        
        for i, result in enumerate(results.get('results', [])[:3], 1):
            print(f"\n{i}. {result.get('title', 'No title')}")
            print(f"   Score: {result.get('score', 0):.2f}")
            print(f"   {result.get('snippet', '')[:150]}...")
            print(f"   URL: {result.get('url', 'N/A')}")
        
        # Test 2: Specific medical query
        print("\n[TEST 2] Kasai procedure outcomes")
        results2 = await client.search(
            "Kasai portoenterostomy long-term outcomes survival rates",
            search_depth="advanced"
        )
        
        print(f"\nResults found: {results2.get('total_results', 0)}")
        if results2.get('answer'):
            print(f"AI Summary: {results2['answer'][:300]}...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_tavily_search())