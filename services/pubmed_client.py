"""
Real PubMed API integration for medical literature search.
Uses NCBI E-utilities API for searching and fetching articles.
"""

import os
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PubMedClient:
    """
    Client for interacting with PubMed E-utilities API.
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize PubMed client.
        
        Args:
            api_key: NCBI API key for increased rate limits (optional but recommended)
        """
        self.api_key = api_key or os.getenv("PUBMED_API_KEY")
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "relevance",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        publication_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Search PubMed for articles.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            sort: Sort order (relevance, pub_date, Author, etc.)
            min_date: Minimum publication date (YYYY/MM/DD)
            max_date: Maximum publication date (YYYY/MM/DD)
            publication_types: Filter by publication types
            
        Returns:
            Dictionary with search results
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            # Step 1: Search for IDs
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": sort
            }
            
            if self.api_key:
                search_params["api_key"] = self.api_key
            
            if min_date:
                search_params["mindate"] = min_date
            if max_date:
                search_params["maxdate"] = max_date
                
            # Add publication type filters
            if publication_types:
                type_query = " OR ".join([f'"{pt}"[Publication Type]' for pt in publication_types])
                search_params["term"] = f"({query}) AND ({type_query})"
            
            # Search for article IDs
            search_url = f"{self.BASE_URL}/esearch.fcgi"
            async with self.session.get(search_url, params=search_params) as response:
                search_data = await response.json()
            
            if "esearchresult" not in search_data:
                return {"error": "Invalid search response", "articles": []}
            
            id_list = search_data["esearchresult"].get("idlist", [])
            total_count = int(search_data["esearchresult"].get("count", 0))
            
            if not id_list:
                return {
                    "query": query,
                    "total_results": 0,
                    "returned": 0,
                    "articles": []
                }
            
            # Step 2: Fetch article details
            articles = await self.fetch_articles(id_list)
            
            return {
                "query": query,
                "total_results": total_count,
                "returned": len(articles),
                "articles": articles
            }
            
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return {
                "query": query,
                "error": str(e),
                "articles": []
            }
    
    async def fetch_articles(self, pmid_list: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch detailed article information for a list of PMIDs.
        
        Args:
            pmid_list: List of PubMed IDs
            
        Returns:
            List of article dictionaries
        """
        if not pmid_list:
            return []
        
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmid_list),
            "retmode": "xml"
        }
        
        if self.api_key:
            fetch_params["api_key"] = self.api_key
        
        fetch_url = f"{self.BASE_URL}/efetch.fcgi"
        
        async with self.session.get(fetch_url, params=fetch_params) as response:
            xml_data = await response.text()
        
        # Parse XML response
        articles = []
        root = ET.fromstring(xml_data)
        
        for article_elem in root.findall(".//PubmedArticle"):
            article = self._parse_article_xml(article_elem)
            if article:
                articles.append(article)
        
        return articles
    
    def _parse_article_xml(self, article_elem: ET.Element) -> Optional[Dict[str, Any]]:
        """
        Parse a single PubMed article XML element.
        
        Args:
            article_elem: XML element for a PubMed article
            
        Returns:
            Dictionary with article information
        """
        try:
            # Extract PMID
            pmid_elem = article_elem.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None
            
            # Extract title
            title_elem = article_elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "No title"
            
            # Extract abstract
            abstract_texts = []
            abstract_elems = article_elem.findall(".//AbstractText")
            for elem in abstract_elems:
                label = elem.get("Label", "")
                text = elem.text or ""
                if label:
                    abstract_texts.append(f"{label}: {text}")
                else:
                    abstract_texts.append(text)
            abstract = " ".join(abstract_texts) if abstract_texts else "No abstract available"
            
            # Extract authors
            authors = []
            for author_elem in article_elem.findall(".//Author"):
                last_name = author_elem.find("LastName")
                fore_name = author_elem.find("ForeName")
                if last_name is not None:
                    name = last_name.text
                    if fore_name is not None:
                        name = f"{last_name.text} {fore_name.text}"
                    authors.append(name)
            
            # Extract journal info
            journal_elem = article_elem.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else "Unknown"
            
            # Extract publication year
            year_elem = article_elem.find(".//PubDate/Year")
            if year_elem is None:
                year_elem = article_elem.find(".//MedlineDate")
            year = year_elem.text[:4] if year_elem is not None else "Unknown"
            
            # Extract DOI
            doi_elem = article_elem.find(".//ArticleId[@IdType='doi']")
            doi = doi_elem.text if doi_elem is not None else None
            
            # Extract publication type
            pub_types = []
            for pt_elem in article_elem.findall(".//PublicationType"):
                if pt_elem.text:
                    pub_types.append(pt_elem.text)
            
            # Extract keywords
            keywords = []
            for kw_elem in article_elem.findall(".//Keyword"):
                if kw_elem.text:
                    keywords.append(kw_elem.text)
            
            # Construct PubMed URL
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
            
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors[:5],  # Limit to first 5 authors
                "journal": journal,
                "year": year,
                "doi": doi,
                "publication_types": pub_types,
                "keywords": keywords[:10],  # Limit to first 10 keywords
                "pubmed_url": pubmed_url,
                "full_text_url": f"https://doi.org/{doi}" if doi else None
            }
            
        except Exception as e:
            logger.error(f"Error parsing article: {e}")
            return None
    
    async def get_related_articles(self, pmid: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Get articles related to a given PMID.
        
        Args:
            pmid: PubMed ID
            max_results: Maximum number of related articles
            
        Returns:
            List of related articles
        """
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "json",
            "retmax": max_results,
            "cmd": "neighbor"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        url = f"{self.BASE_URL}/elink.fcgi"
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
        
        # Extract related PMIDs
        related_ids = []
        if "linksets" in data:
            for linkset in data["linksets"]:
                if "linksetdbs" in linkset:
                    for db in linkset["linksetdbs"]:
                        if db.get("dbto") == "pubmed":
                            related_ids.extend(db.get("links", [])[:max_results])
        
        if related_ids:
            return await self.fetch_articles(related_ids[:max_results])
        
        return []


async def test_pubmed_client():
    """Test the PubMed client with a real query"""
    
    print("\n" + "="*60)
    print("TESTING REAL PUBMED API")
    print("="*60)
    
    async with PubMedClient() as client:
        # Test 1: Search for biliary atresia
        print("\n[TEST 1] Searching for: biliary atresia Kasai")
        results = await client.search(
            query="biliary atresia Kasai portoenterostomy",
            max_results=3,
            sort="relevance"
        )
        
        print(f"Total results found: {results.get('total_results', 0)}")
        print(f"Articles returned: {results.get('returned', 0)}")
        
        for i, article in enumerate(results.get("articles", []), 1):
            print(f"\n{i}. {article.get('title', 'No title')}")
            print(f"   Authors: {', '.join(article.get('authors', [])[:3])}")
            print(f"   Journal: {article.get('journal', 'Unknown')} ({article.get('year', 'Unknown')})")
            print(f"   PMID: {article.get('pmid', 'N/A')}")
            print(f"   URL: {article.get('pubmed_url', 'N/A')}")
            if article.get('doi'):
                print(f"   DOI: {article.get('doi')}")
        
        # Test 2: Search with date filter
        print("\n[TEST 2] Searching for recent articles (2020-2024)")
        recent_results = await client.search(
            query="biliary atresia treatment",
            max_results=2,
            min_date="2020/01/01",
            max_date="2024/12/31",
            publication_types=["Review", "Meta-Analysis"]
        )
        
        print(f"Recent articles found: {recent_results.get('returned', 0)}")
        for article in recent_results.get("articles", []):
            print(f"- {article.get('title', 'No title')} ({article.get('year', 'Unknown')})")
            print(f"  Type: {', '.join(article.get('publication_types', [])[:2])}")


if __name__ == "__main__":
    asyncio.run(test_pubmed_client())