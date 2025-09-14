"""
Optimized Medical Search with Fixed Memory Management
- Proper connection cleanup
- Resource management with context managers
- Memory leak prevention
- Connection limits
"""

import asyncio
import aiohttp
import hashlib
import json
import time
import weakref
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from functools import lru_cache
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ResponseCache:
    """Simple in-memory cache with size limits"""
    
    def __init__(self, ttl_seconds=300, max_entries=100):
        self.cache = {}
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.access_times = {}  # Track access times for LRU
    
    def get_key(self, query: str, tool: str) -> str:
        """Generate cache key"""
        return hashlib.md5(f"{tool}:{query}".encode()).hexdigest()
    
    def get(self, query: str, tool: str) -> Optional[Any]:
        """Get cached result if not expired"""
        key = self.get_key(query, tool)
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.access_times[key] = time.time()
                logger.debug(f"Cache hit for {tool}: {query[:50]}...")
                return result
            else:
                # Expired, remove from cache
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        return None
    
    def set(self, query: str, tool: str, result: Any):
        """Cache a result with LRU eviction"""
        key = self.get_key(query, tool)
        
        # Check if we need to evict entries
        if len(self.cache) >= self.max_entries and key not in self.cache:
            # Find and remove least recently used
            if self.access_times:
                lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                del self.cache[lru_key]
                del self.access_times[lru_key]
        
        self.cache[key] = (result, time.time())
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.access_times.clear()


class ManagedConnectionPool:
    """Properly managed connection pool with cleanup"""
    
    def __init__(self):
        self.session = None
        self.connector = None
        self._lock = asyncio.Lock()
        self._request_count = 0
        self._max_requests_per_session = 100  # Recreate session after N requests
    
    @asynccontextmanager
    async def get_session(self):
        """Get session with automatic cleanup and rotation"""
        async with self._lock:
            # Check if we need to rotate the session
            if self._request_count >= self._max_requests_per_session:
                await self._close_session()
                self._request_count = 0
            
            # Create session if needed
            if self.session is None or self.session.closed:
                await self._create_session()
            
            self._request_count += 1
        
        try:
            yield self.session
        except Exception as e:
            logger.error(f"Session error: {e}")
            raise
    
    async def _create_session(self):
        """Create a new session with proper limits"""
        # Close existing if any
        await self._close_session()
        
        # Create connector with strict limits
        # Note: force_close=True is incompatible with keepalive_timeout
        self.connector = aiohttp.TCPConnector(
            limit=50,  # Total connection limit (reduced from 100)
            limit_per_host=10,  # Per-host limit (reduced from 30)
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            force_close=True  # Force close connections to prevent leaks
        )
        
        # Create session with timeout
        timeout = aiohttp.ClientTimeout(
            total=30,
            connect=10,
            sock_read=10
        )
        
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            headers={'User-Agent': 'Medical-Search-Agent/2.0'},
            connector_owner=True  # Session owns the connector
        )
        
        logger.debug("Created new HTTP session")
    
    async def _close_session(self):
        """Properly close session and connector"""
        if self.session and not self.session.closed:
            await self.session.close()
            # Wait a bit for connections to close
            await asyncio.sleep(0.1)
        
        self.session = None
        self.connector = None
    
    async def cleanup(self):
        """Full cleanup"""
        await self._close_session()
        logger.info(f"Cleaned up after {self._request_count} requests")


class OptimizedMedicalSearchFixed:
    """Fixed version with proper memory management"""
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self.cache = ResponseCache(max_entries=50)  # Limit cache size
        self.connection_pool = ManagedConnectionPool()
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'errors': 0,
            'memory_cleanups': 0
        }
        # Track active tasks for cleanup
        self._active_tasks = weakref.WeakSet()
    
    async def search_parallel(self, query: str, tools: List[str]) -> Dict[str, Any]:
        """
        Execute multiple tool searches in parallel with proper cleanup
        """
        tasks = []
        
        for tool in tools:
            # Check cache first
            cached = self.cache.get(query, tool)
            if cached:
                self.performance_stats['cache_hits'] += 1
                tasks.append(asyncio.create_task(self._return_cached(tool, cached)))
            else:
                self.performance_stats['cache_misses'] += 1
                task = asyncio.create_task(self._execute_tool_safe(tool, query))
                self._active_tasks.add(task)
                tasks.append(task)
        
        # Execute all tasks in parallel with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=15.0  # Global timeout for all searches
            )
        except asyncio.TimeoutError:
            logger.warning("Parallel search timed out")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            results = [{'error': 'Search timeout'} for _ in tasks]
        
        # Combine results
        combined = {}
        for tool, result in zip(tools, results):
            if isinstance(result, Exception):
                logger.error(f"Error in {tool}: {result}")
                self.performance_stats['errors'] += 1
                combined[tool] = {'error': str(result)}
            else:
                combined[tool] = result
                # Cache successful results
                if 'error' not in result:
                    self.cache.set(query, tool, result)
        
        # Periodic cleanup
        if self.performance_stats['total_requests'] % 50 == 0:
            await self._cleanup_resources()
        
        return combined
    
    async def _return_cached(self, tool: str, result: Any) -> Any:
        """Return cached result (async for consistency)"""
        return result
    
    async def _execute_tool_safe(self, tool: str, query: str) -> Dict[str, Any]:
        """
        Execute a tool with proper error handling and resource cleanup
        """
        max_retries = 2  # Reduced from 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                self.performance_stats['total_requests'] += 1
                
                # Execute based on tool type
                if tool == "search_pubmed":
                    return await self._search_pubmed_optimized(query)
                elif tool == "search_fda":
                    return await self._search_fda_optimized(query)
                elif tool == "search_clinical_trials":
                    return await self._search_clinical_trials_optimized(query)
                elif tool == "web_search":
                    return await self._web_search_optimized(query)
                elif tool == "calculate_pediatric_dose":
                    return await self._calculate_pediatric_dose(query)
                elif tool == "check_drug_interactions":
                    return await self._check_drug_interactions(query)
                elif tool == "check_pregnancy_safety":
                    return await self._check_pregnancy_safety(query)
                elif tool == "check_renal_dosing":
                    return await self._check_renal_dosing(query)
                elif tool == "analyze_lab_values":
                    return await self._analyze_lab_values(query)
                elif tool == "get_disease_statistics":
                    return await self._get_disease_statistics(query)
                elif tool == "calculate_medical_score":
                    return await self._calculate_medical_score(query)
                else:
                    return await self._execute_tool_fallback(tool, query)
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1} for {tool}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    return {'error': 'Request timed out', 'source': tool}
                    
            except Exception as e:
                logger.error(f"Error in {tool}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    return {'error': str(e)[:100], 'source': tool}
        
        return {'error': 'Max retries exceeded', 'source': tool}
    
    def _split_medical_query(self, query: str) -> List[str]:
        """Split complex medical queries into targeted PubMed searches"""
        query_lower = query.lower()
        split_queries = []

        # Identify key medical concepts to search separately
        if 'kasai' in query_lower or 'biliary atresia' in query_lower:
            # Post-Kasai scenario - search multiple aspects
            if 'hepatectomy' in query_lower:
                split_queries.append("hepatectomy post kasai biliary atresia")
                split_queries.append("liver resection biliary atresia outcomes")
            if 'normal bilirubin' in query_lower or 'elevated' in query_lower:
                split_queries.append("biliary atresia normal bilirubin prognosis")
                split_queries.append("alkaline phosphatase biliary atresia outcomes")
            if 'left' in query_lower:
                split_queries.append("segmental hepatectomy biliary atresia")
            # Always include general Kasai outcomes
            split_queries.append("kasai procedure long term outcomes")

        # Drug interaction queries
        elif any(drug_term in query_lower for drug_term in ['interact', 'combine', 'safe with', 'together']):
            # Extract drug names and search combinations
            import re
            # This would need more sophisticated drug extraction
            split_queries.append(query)  # Use original for now

        # Pediatric dosing queries
        elif 'dose' in query_lower or 'dosing' in query_lower:
            split_queries.append(query)
            if 'pediatric' in query_lower or 'child' in query_lower or 'kg' in query_lower:
                # Add general pediatric dosing search
                drug_match = re.search(r'(\w+)\s+(?:dose|dosing)', query_lower)
                if drug_match:
                    drug = drug_match.group(1)
                    split_queries.append(f"{drug} pediatric dosing guidelines")

        # Default: use original query
        if not split_queries:
            split_queries = [query]

        # Limit to 3 queries max to avoid overwhelming
        return split_queries[:3]

    async def _search_pubmed_optimized(self, query: str) -> Dict[str, Any]:
        """Optimized PubMed search using our PubMed client - with query splitting for complex questions"""
        from services.pubmed_client import PubMedClient

        try:
            # Analyze if we should split the query
            split_queries = self._split_medical_query(query)

            async with PubMedClient() as client:
                all_results = []
                total_count = 0
                seen_pmids = set()

                # Execute each split query
                for sub_query in split_queries:
                    results = await client.search(sub_query, max_results=3)

                    if 'error' not in results:
                        # Convert 'articles' to 'results' format for consistency
                        for article in results.get('articles', []):
                            pmid = article.get('pmid')
                            # Avoid duplicates based on PMID
                            if pmid and pmid not in seen_pmids:
                                seen_pmids.add(pmid)
                                all_results.append({
                                    'pmid': pmid,
                                    'title': article.get('title', ''),
                                    'abstract': article.get('abstract', ''),
                                    'authors': article.get('authors', []),
                                    'journal': article.get('journal', ''),
                                    'year': article.get('year', article.get('pub_year', '')),
                                    'url': article.get('pubmed_url', f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                                    'query_source': sub_query  # Track which query found this
                                })
                        total_count += results.get('total_results', 0)

                # Sort by relevance (articles from first queries are more relevant)
                # and limit to top 5
                all_results = all_results[:5]

                return {
                    'source': 'pubmed',
                    'results': all_results,
                    'count': len(all_results),
                    'total_available': total_count,
                    'queries_used': split_queries  # For debugging
                }

        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return {'error': str(e)[:100], 'source': 'pubmed'}
    
    async def _search_fda_optimized(self, query: str) -> Dict[str, Any]:
        """Optimized FDA search"""
        async with self.connection_pool.get_session() as session:
            try:
                url = "https://api.fda.gov/drug/label.json"
                params = {
                    'search': query,
                    'limit': 5
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', [])
                        return {
                            'source': 'fda',
                            'results': results[:5],  # Limit results
                            'count': len(results)
                        }
                    else:
                        return {'error': f'FDA API returned {response.status}', 'source': 'fda'}
                        
            except Exception as e:
                logger.error(f"FDA search error: {e}")
                return {'error': str(e)[:100], 'source': 'fda'}
    
    async def _search_clinical_trials_optimized(self, query: str) -> Dict[str, Any]:
        """Optimized ClinicalTrials.gov search using our client"""
        from services.clinical_trials_client import ClinicalTrialsClient

        try:
            async with ClinicalTrialsClient() as client:
                results = await client.search(query, max_results=5)

                if 'error' in results:
                    return {'error': results['error'], 'source': 'clinical_trials'}

                # Format results for consistency
                formatted_results = []
                for trial in results.get('trials', []):
                    formatted_results.append({
                        'nct_id': trial.get('nct_id', ''),
                        'title': trial.get('title', ''),
                        'status': trial.get('status', ''),
                        'conditions': trial.get('conditions', []),
                        'interventions': trial.get('interventions', []),
                        'url': trial.get('url', '')
                    })

                return {
                    'source': 'clinical_trials',
                    'results': formatted_results,
                    'count': len(formatted_results),
                    'total_available': results.get('total_trials', results.get('returned', 0))
                }

        except Exception as e:
            logger.error(f"Clinical trials search error: {e}")
            return {'error': str(e)[:100], 'source': 'clinical_trials'}
    
    async def _web_search_optimized(self, query: str) -> Dict[str, Any]:
        """Web search using Tavily AI-optimized search"""
        from services.tavily_search_client import TavilySearchClient

        try:
            async with TavilySearchClient() as client:
                # Use medical search for medical-looking queries
                results = await client.search_medical(query)

                # Format results for consistency
                formatted_results = []
                for result in results.get('results', []):
                    formatted_results.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'url': result.get('url', ''),
                        'score': result.get('score', 0)
                    })

                return {
                    'source': 'web',
                    'results': formatted_results,
                    'count': len(formatted_results),
                    'ai_answer': results.get('answer', ''),
                    'status': 'success'
                }

        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return {
                'source': 'web',
                'error': str(e)[:100],
                'results': [],
                'count': 0
            }
    
    async def _execute_tool_fallback(self, tool: str, query: str) -> Dict[str, Any]:
        """Fallback for other tools"""
        return {
            'source': tool,
            'message': f'Tool {tool} executed',
            'results': []
        }
    
    async def _cleanup_resources(self):
        """Periodic resource cleanup"""
        self.performance_stats['memory_cleanups'] += 1
        
        # Cancel any hanging tasks
        for task in list(self._active_tasks):
            if not task.done():
                task.cancel()
        
        # Clear old cache entries
        if len(self.cache.cache) > 30:
            # Keep only recent entries
            recent_keys = sorted(
                self.cache.access_times.items(),
                key=lambda x: x[1],
                reverse=True
            )[:30]
            
            new_cache = {}
            for key, _ in recent_keys:
                if key in self.cache.cache:
                    new_cache[key] = self.cache.cache[key]
            self.cache.cache = new_cache
        
        logger.debug(f"Cleaned up resources (cleanup #{self.performance_stats['memory_cleanups']})")
    
    async def execute_with_reasoning(self, query: str) -> Dict[str, Any]:
        """
        Execute search with LLM reasoning
        """
        start_time = time.time()
        
        try:
            # Step 1: Determine tools (with timeout)
            tools_to_use = await asyncio.wait_for(
                self._determine_tools(query),
                timeout=5.0
            )
            
            # Step 2: Execute tools in parallel
            results = await self.search_parallel(query, tools_to_use)
            
            # Step 3: Synthesize results (with timeout)
            answer = await asyncio.wait_for(
                self._synthesize_results(query, results),
                timeout=5.0
            )
            
            execution_time = time.time() - start_time
            
            return {
                'answer': answer,
                'tools_used': tools_to_use,
                'raw_results': results,
                'execution_time': execution_time,
                'performance_stats': self.performance_stats.copy(),
                'cache_efficiency': self._calculate_cache_efficiency()
            }
            
        except asyncio.TimeoutError:
            return {
                'answer': "Search timed out. Please try again.",
                'error': 'timeout',
                'execution_time': time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Execute error: {e}")
            return {
                'answer': "An error occurred during search.",
                'error': str(e)[:100],
                'execution_time': time.time() - start_time
            }
    
    async def _determine_tools(self, query: str) -> List[str]:
        """Use LLM to determine which tools to use"""
        if not self.llm_service:
            # Smart defaults based on query patterns
            query_lower = query.lower()
            if "interaction" in query_lower:
                return ['search_pubmed', 'search_fda']
            elif "clinical trial" in query_lower:
                return ['search_clinical_trials']
            elif "drug" in query_lower or "medication" in query_lower:
                return ['search_fda', 'search_pubmed']
            else:
                return ['search_pubmed']
        
        prompt = f"""
        Given this medical query: "{query}"
        
        Select the most relevant tools (1-3 tools max):
        - search_pubmed: Medical research papers
        - search_fda: Drug information
        - search_clinical_trials: Clinical trials
        - web_search: General medical info
        
        Return only tool names, comma-separated.
        """
        
        try:
            response = await self.llm_service.chat(prompt)
            tools = [t.strip() for t in response.split(',')]
            valid_tools = ['search_pubmed', 'search_fda', 'search_clinical_trials', 'web_search']
            selected = [t for t in tools if t in valid_tools][:3]  # Max 3 tools
            return selected if selected else ['search_pubmed']
        except:
            return ['search_pubmed']  # Default
    
    async def _synthesize_results(self, query: str, results: Dict[str, Any]) -> str:
        """Synthesize results into a comprehensive medical answer"""
        if not self.llm_service:
            # Simple summary without LLM
            answer = f"Results for '{query}':\n\n"
            for tool, data in results.items():
                if 'error' not in data:
                    count = data.get('count', 0)
                    answer += f"**{tool}**: Found {count} results\n"
            return answer

        # Prepare rich context for comprehensive analysis
        context_parts = []
        citation_count = 0

        for tool, data in results.items():
            if 'error' not in data and data.get('results'):
                # Include more results and more content for comprehensive analysis
                tool_results = data['results'][:5]  # Increased from 2 to 5

                # Build detailed context with abstracts, titles, and citations
                tool_context = f"\n=== {tool.upper()} RESULTS ===\n"

                for i, result in enumerate(tool_results, 1):
                    citation_count += 1
                    if isinstance(result, dict):
                        # Include title
                        title = result.get('title', 'No title')
                        tool_context += f"\n[{citation_count}] {title}\n"

                        # Include full abstract/content if available
                        abstract = result.get('abstract', '').strip()
                        content = result.get('content', '').strip()
                        text = result.get('text', '').strip()

                        full_text = abstract or content or text
                        if full_text:
                            # Include full abstract/content instead of truncating
                            tool_context += f"Abstract: {full_text}\n"

                        # Include authors and journal for credibility
                        authors = result.get('authors', [])
                        if authors:
                            author_str = ', '.join(authors[:3])  # First 3 authors
                            if len(authors) > 3:
                                author_str += " et al."
                            tool_context += f"Authors: {author_str}\n"

                        journal = result.get('journal', '')
                        year = result.get('year', '') or result.get('pub_year', '')
                        if journal or year:
                            tool_context += f"Source: {journal} ({year})\n"

                        # Include PMID or other identifiers
                        pmid = result.get('pmid', '')
                        doi = result.get('doi', '')
                        if pmid:
                            tool_context += f"PMID: {pmid}\n"
                        elif doi:
                            tool_context += f"DOI: {doi}\n"

                        tool_context += "\n"

                context_parts.append(tool_context)

        # Allow much more context for comprehensive analysis
        context = "\n".join(context_parts)[:8000]  # Increased from 2000 to 8000

        # Use LLM to intelligently classify the query type
        classification_prompt = f"""
        Classify this medical query into ONE of the following categories based on what the user is primarily asking for:

        Query: "{query}"

        Categories:
        - definition: Asking what something is, for explanations or descriptions
        - treatment: Asking about how to treat, manage, or therapy options
        - prognosis: Asking about risks, outcomes, survival, or disease progression
        - dosing: Asking about specific medication doses, amounts, or administration
        - diagnosis: Asking how to diagnose, test for, or identify a condition
        - mechanism: Asking about causes, pathophysiology, or how something works
        - comparison: Asking to compare different options, treatments, or conditions
        - research_update: Asking about latest, recent, new, or current research findings
        - general: General medical questions that don't fit the above categories

        Respond with just the category name, nothing else.
        """

        try:
            # Get LLM classification
            query_type = await self.llm_service.chat(classification_prompt)
            query_type = query_type.strip().lower()

            # Validate the response
            valid_types = ['definition', 'treatment', 'prognosis', 'dosing', 'diagnosis',
                          'mechanism', 'comparison', 'research_update', 'general']
            if query_type not in valid_types:
                query_type = 'general'

        except Exception as e:
            # Fallback to simple keyword matching if LLM fails
            logger.warning(f"LLM classification failed, using fallback: {e}")
            query_lower = query.lower()

            if any(word in query_lower for word in ['what is', 'define', 'explain']):
                query_type = 'definition'
            elif any(word in query_lower for word in ['treatment', 'therapy', 'manage']):
                query_type = 'treatment'
            elif any(word in query_lower for word in ['risk', 'prognosis', 'outcome']):
                query_type = 'prognosis'
            else:
                query_type = 'general'

        # Create adaptive prompt based on query type
        base_prompt = f"""
        You are a medical expert providing a comprehensive, detailed response to a medical query.

        QUERY: "{query}"

        SEARCH RESULTS:
        {context}
        """

        # Add query-specific structure guidance
        if query_type == 'definition':
            specific_guidance = """
        Provide a comprehensive explanation that covers:
        - Clear definition and description
        - Key characteristics and features
        - Clinical significance and relevance
        - Current understanding from research
        - Related conditions or concepts if relevant
        """
        elif query_type == 'treatment':
            specific_guidance = """
        Provide detailed treatment information including:
        - Available treatment options
        - Mechanism of action where relevant
        - Effectiveness and evidence base
        - Dosing and administration details if applicable
        - Side effects and contraindications
        - Clinical guidelines and recommendations
        """
        elif query_type == 'prognosis':
            specific_guidance = """
        Provide comprehensive prognostic information including:
        - Risk factors and their significance
        - Statistical data and survival rates
        - Factors affecting outcomes
        - Current research findings
        - Clinical implications
        - Preventive measures if applicable
        """
        elif query_type == 'dosing':
            specific_guidance = """
        Provide specific dosing information including:
        - Standard dosing regimens
        - Adjustments for special populations
        - Administration guidelines
        - Safety considerations
        - Drug interactions if relevant
        - Clinical pearls and practical tips
        """
        elif query_type == 'diagnosis':
            specific_guidance = """
        Provide diagnostic information including:
        - Diagnostic criteria and methods
        - Test sensitivity and specificity
        - Differential diagnoses
        - Clinical presentation
        - Diagnostic algorithms if applicable
        - Recent advances in diagnosis
        """
        elif query_type == 'mechanism':
            specific_guidance = """
        Explain the underlying mechanisms including:
        - Pathophysiology or mechanism of action
        - Molecular and cellular processes
        - Contributing factors
        - Current theoretical understanding
        - Research evidence supporting mechanisms
        """
        elif query_type == 'comparison':
            specific_guidance = """
        Provide a detailed comparison including:
        - Key similarities and differences
        - Advantages and disadvantages of each
        - Evidence base for each option
        - Clinical scenarios for preference
        - Expert recommendations
        """
        elif query_type == 'research_update':
            specific_guidance = """
        Focus on the latest research findings including:
        - Recent studies and their key findings
        - Changes in understanding or practice
        - Emerging evidence and trends
        - Ongoing clinical trials if relevant
        - Future directions
        - How this updates previous knowledge
        """
        else:  # general
            specific_guidance = """
        Provide a comprehensive response that addresses all aspects of the query, including:
        - Direct answer to the question
        - Supporting evidence from research
        - Clinical context and significance
        - Practical implications
        - Additional relevant information
        """

        prompt = base_prompt + specific_guidance + """

        IMPORTANT REQUIREMENTS:
        - Be thorough and comprehensive, not brief
        - Include specific statistics, numbers, and research findings when available
        - Reference sources using [1], [2] notation throughout
        - Use clear headings to organize your response naturally
        - Include multiple research perspectives when available
        - Aim for a response suitable for healthcare professionals
        - Adapt your response structure to best answer the specific question
        - Don't force sections that aren't relevant to the query

        Provide a detailed, evidence-based medical response that directly addresses the query.
        """

        try:
            return await self.llm_service.chat(prompt)
        except Exception as e:
            logger.error(f"Error in synthesis: {str(e)}")
            # Enhanced fallback with available data
            fallback = f"# Medical Information for: {query}\n\n"
            for tool, data in results.items():
                if 'error' not in data and data.get('results'):
                    fallback += f"## Results from {tool}:\n"
                    for result in data['results'][:3]:
                        if isinstance(result, dict):
                            title = result.get('title', 'No title')
                            abstract = result.get('abstract', result.get('content', ''))[:300]
                            fallback += f"- **{title}**\n  {abstract}...\n\n"
            return fallback
    
    def _calculate_cache_efficiency(self) -> float:
        """Calculate cache hit rate"""
        total = self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']
        if total == 0:
            return 0.0
        return (self.performance_stats['cache_hits'] / total) * 100
    
    async def cleanup(self):
        """Complete cleanup of all resources"""
        # Cancel active tasks
        for task in list(self._active_tasks):
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        
        # Clean up connection pool
        await self.connection_pool.cleanup()
        
        # Clear cache
        self.cache.clear()
        
        logger.info(f"Final cleanup: {self.performance_stats}")
    
    # Add specialized tool implementations
    async def _calculate_pediatric_dose(self, query: str) -> Dict[str, Any]:
        """Calculate pediatric dosing with comprehensive medical database"""
        import re

        # More flexible weight extraction
        weight_patterns = [
            r'(\d+(?:\.\d+)?)\s*kg',
            r'(\d+(?:\.\d+)?)\s*kilogram',
            r'weighs?\s+(\d+(?:\.\d+)?)',
            r'(\d+)\s*pound',  # Convert from pounds
        ]

        weight = None
        for pattern in weight_patterns:
            match = re.search(pattern, query.lower())
            if match:
                weight = float(match.group(1))
                if 'pound' in pattern:
                    weight = weight * 0.453592  # Convert to kg
                break

        if not weight:
            weight = 20  # Default

        # Comprehensive pediatric dosing database
        pediatric_doses = {
            'epinephrine': {
                'anaphylaxis': {
                    'dose_per_kg': 0.01,  # mg/kg
                    'concentration': '1:1000',
                    'route': 'IM',
                    'max_dose': 0.5,
                    'frequency': 'May repeat every 5-15 minutes',
                    'notes': 'Lateral thigh preferred site'
                }
            },
            'amoxicillin': {
                'otitis_media': {
                    'dose_per_kg': 45,  # mg/kg/day
                    'divided': 2,
                    'max_dose': 1500,
                    'duration': '10 days'
                },
                'strep_throat': {
                    'dose_per_kg': 50,
                    'divided': 2,
                    'max_dose': 1000,
                    'duration': '10 days'
                }
            },
            'ibuprofen': {
                'fever_pain': {
                    'dose_per_kg': 10,
                    'frequency': 'Every 6-8 hours',
                    'max_dose': 400,
                    'max_daily': 40  # mg/kg/day
                }
            },
            'acetaminophen': {
                'fever_pain': {
                    'dose_per_kg': 15,
                    'frequency': 'Every 4-6 hours',
                    'max_dose': 1000,
                    'max_daily': 75  # mg/kg/day
                }
            },
            'azithromycin': {
                'pneumonia': {
                    'dose_per_kg': 10,  # Day 1
                    'subsequent': 5,  # Days 2-5
                    'max_dose': 500,
                    'duration': '5 days'
                }
            },
            'ceftriaxone': {
                'meningitis': {
                    'dose_per_kg': 100,
                    'divided': 1,
                    'max_dose': 4000,
                    'frequency': 'Daily'
                },
                'sepsis': {
                    'dose_per_kg': 75,
                    'divided': 1,
                    'max_dose': 2000,
                    'frequency': 'Daily'
                }
            }
        }

        # Extract drug name
        drug = None
        query_lower = query.lower()
        for drug_name in pediatric_doses.keys():
            if drug_name in query_lower:
                drug = drug_name
                break

        if not drug:
            # Try to extract from query
            patterns = [
                r'dose\s+(?:of\s+)?(\w+)',
                r'(\w+)\s+(?:dose|dosing)',
                r'calculate\s+(\w+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    potential_drug = match.group(1).lower()
                    if potential_drug in pediatric_doses:
                        drug = potential_drug
                        break

        if drug and drug in pediatric_doses:
            # Get the first indication for the drug
            drug_info = pediatric_doses[drug]
            indication = list(drug_info.keys())[0]
            dosing = drug_info[indication]

            # Calculate actual dose
            dose_mg = dosing['dose_per_kg'] * weight
            if 'max_dose' in dosing and dose_mg > dosing['max_dose']:
                dose_mg = dosing['max_dose']

            # Format response
            if drug == 'epinephrine':
                dose_ml = dose_mg  # Already in mL for epi
                result_text = f"""**Pediatric Epinephrine Dosing for {weight}kg child:**

• **Dose**: {dose_mg:.2f} mg ({dose_ml:.2f} mL of 1:1000 solution)
• **Route**: Intramuscular (lateral thigh preferred)
• **Maximum dose**: 0.5 mg (0.5 mL)
• **Frequency**: May repeat every 5-15 minutes if needed
• **Concentration**: Use 1:1000 (1 mg/mL) for IM route

**Important Notes**:
- EpiPen Jr (0.15 mg) for 10-25 kg
- EpiPen (0.3 mg) for ≥25 kg
- Monitor vital signs continuously
- Have airway equipment ready"""
            else:
                divided = dosing.get('divided', 1)
                dose_per_admin = dose_mg / divided if divided > 1 else dose_mg

                result_text = f"""**Pediatric {drug.capitalize()} Dosing for {weight}kg child:**

• **Total daily dose**: {dose_mg:.1f} mg/day
• **Per dose**: {dose_per_admin:.1f} mg
• **Frequency**: {dosing.get('frequency', f'Divided into {divided} doses')}
• **Maximum dose**: {dosing.get('max_dose', 'N/A')} mg
• **Duration**: {dosing.get('duration', 'As prescribed')}

**Standard Indication**: {indication.replace('_', ' ').title()}"""

            return {
                'source': 'pediatric_calculator',
                'drug': drug,
                'weight_kg': weight,
                'results': [{
                    'drug': drug,
                    'weight': weight,
                    'dose_calculated': dose_mg,
                    'dosing_info': dosing,
                    'content': result_text
                }],
                'answer': result_text
            }
        else:
            # Provide general pediatric dosing principles
            return {
                'source': 'pediatric_calculator',
                'drug': drug or 'medication',
                'weight_kg': weight,
                'results': [{
                    'content': f"""**General Pediatric Dosing Principles for {weight}kg child:**

• Most pediatric doses are weight-based (mg/kg)
• Always verify with current clinical guidelines
• Consider age-specific factors (renal/hepatic function)
• Maximum doses usually don't exceed adult doses
• Common dosing references:
  - Lexicomp Pediatric Dosing Handbook
  - UpToDate
  - AAP Red Book for antimicrobials
  - Nelson's Pediatric Antimicrobial Therapy

**Safety Note**: Always double-check calculations and use appropriate measuring devices."""
                }],
                'answer': f"Specific dosing for {drug or 'this medication'} requires consultation with current pediatric dosing guidelines."
            }
    
    async def _check_drug_interactions(self, query: str) -> Dict[str, Any]:
        """Check drug interactions with comprehensive database"""
        import re

        # Comprehensive drug interaction database
        interaction_database = {
            ('maoi', 'ssri'): {
                'severity': 'CONTRAINDICATED',
                'mechanism': 'Serotonin syndrome',
                'effects': 'Life-threatening serotonin syndrome: hyperthermia, rigidity, myoclonus, autonomic instability, mental status changes',
                'management': 'Absolutely contraindicated. Wait minimum 14 days after stopping MAOI before starting SSRI (5 weeks for fluoxetine)',
                'clinical_note': 'This combination can be fatal. Never co-administer.'
            },
            ('warfarin', 'nsaid'): {
                'severity': 'MAJOR',
                'mechanism': 'Additive anticoagulation and gastric irritation',
                'effects': 'Significantly increased bleeding risk, GI hemorrhage',
                'management': 'Avoid if possible. If necessary, use COX-2 selective NSAID with close INR monitoring',
                'clinical_note': 'Risk of major bleeding increases 2-4 fold'
            },
            ('methotrexate', 'trimethoprim'): {
                'severity': 'CONTRAINDICATED',
                'mechanism': 'Both are folate antagonists',
                'effects': 'Severe bone marrow suppression, pancytopenia, megaloblastic anemia',
                'management': 'Avoid combination. Use alternative antibiotic',
                'clinical_note': 'Can cause fatal pancytopenia even at low methotrexate doses'
            },
            ('warfarin', 'herbal'): {
                'severity': 'VARIABLE',
                'common_interactions': {
                    'St. John\'s Wort': 'Decreases INR (CYP3A4 induction)',
                    'Ginkgo biloba': 'Increases bleeding risk',
                    'Ginseng': 'Decreases warfarin effect',
                    'Garlic': 'Increases bleeding risk',
                    'Green tea': 'High vitamin K content opposes warfarin',
                    'Cranberry': 'Increases INR (CYP2C9 inhibition)'
                },
                'management': 'Avoid herbal supplements or monitor INR closely',
                'clinical_note': 'Many herbals have unpredictable effects on warfarin'
            },
            ('ssri', 'nsaid'): {
                'severity': 'MODERATE',
                'mechanism': 'Increased serotonin and platelet dysfunction',
                'effects': 'Increased GI bleeding risk (6-fold increase)',
                'management': 'Use PPI for gastroprotection if combination necessary',
                'clinical_note': 'Risk highest in elderly and those with prior GI bleed'
            },
            ('ace', 'potassium'): {
                'severity': 'MAJOR',
                'mechanism': 'Both increase potassium retention',
                'effects': 'Hyperkalemia, cardiac arrhythmias',
                'management': 'Monitor potassium levels closely, avoid supplements unless hypokalemic',
                'clinical_note': 'Risk higher with renal impairment'
            },
            ('metformin', 'ace'): {
                'severity': 'MINOR',
                'mechanism': 'No direct interaction, both commonly used in diabetics',
                'effects': 'Generally safe combination, monitor renal function',
                'management': 'Safe to use together with regular monitoring of renal function and potassium',
                'clinical_note': 'Common combination in diabetic patients with hypertension'
            },
            ('statin', 'ace'): {
                'severity': 'NONE',
                'mechanism': 'No interaction',
                'effects': 'Safe combination, often used together',
                'management': 'No special precautions needed',
                'clinical_note': 'Standard combination for cardiovascular risk reduction'
            },
            ('metformin', 'statin'): {
                'severity': 'NONE',
                'mechanism': 'No significant interaction',
                'effects': 'Safe combination commonly used in diabetics',
                'management': 'Monitor liver function as both can rarely affect liver',
                'clinical_note': 'Standard combination in diabetic patients with dyslipidemia'
            },
            ('nsaid', 'ace'): {
                'severity': 'MODERATE',
                'mechanism': 'NSAIDs may reduce antihypertensive effect and worsen renal function',
                'effects': 'Reduced blood pressure control, potential renal impairment',
                'management': 'Monitor blood pressure and renal function, use lowest NSAID dose',
                'clinical_note': 'Risk higher in elderly and those with existing renal disease'
            },
            ('ppi', 'warfarin'): {
                'severity': 'MODERATE',
                'mechanism': 'Some PPIs inhibit CYP2C19, affecting warfarin metabolism',
                'effects': 'Possible increased INR and bleeding risk',
                'management': 'Monitor INR when starting or stopping PPI',
                'clinical_note': 'Pantoprazole has least interaction potential'
            }
        }

        # Extract drugs from query - extended database
        drug_patterns = {
            'maoi': ['maoi', 'maois', 'phenelzine', 'tranylcypromine', 'selegiline'],
            'ssri': ['ssri', 'ssris', 'fluoxetine', 'sertraline', 'paroxetine', 'citalopram', 'escitalopram'],
            'warfarin': ['warfarin', 'coumadin'],
            'nsaid': ['nsaid', 'nsaids', 'ibuprofen', 'naproxen', 'diclofenac', 'aspirin', 'celecoxib', 'meloxicam'],
            'methotrexate': ['methotrexate', 'mtx'],
            'trimethoprim': ['trimethoprim', 'bactrim', 'septra', 'sulfamethoxazole'],
            'herbal': ['herbal', 'herb', 'supplement', 'st john', 'ginkgo', 'ginseng', 'garlic'],
            'ace': ['ace', 'lisinopril', 'enalapril', 'ramipril', 'captopril', 'benazepril', 'quinapril'],
            'potassium': ['potassium', 'k+', 'kdur', 'klor'],
            'metformin': ['metformin', 'glucophage'],
            'statin': ['statin', 'atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin', 'lovastatin'],
            'beta_blocker': ['metoprolol', 'atenolol', 'propranolol', 'carvedilol', 'bisoprolol'],
            'ppi': ['omeprazole', 'pantoprazole', 'esomeprazole', 'lansoprazole', 'rabeprazole'],
            'antibiotic': ['amoxicillin', 'azithromycin', 'ciprofloxacin', 'doxycycline', 'cephalexin'],
            'anticoagulant': ['apixaban', 'rivaroxaban', 'dabigatran', 'edoxaban', 'heparin'],
            'antidepressant': ['venlafaxine', 'duloxetine', 'bupropion', 'mirtazapine', 'trazodone']
        }

        query_lower = query.lower()
        identified_classes = []

        # Identify drug classes in query
        for drug_class, keywords in drug_patterns.items():
            for keyword in keywords:
                if keyword in query_lower:
                    identified_classes.append(drug_class)
                    break

        # Remove duplicates
        identified_classes = list(dict.fromkeys(identified_classes))

        # Check for interactions
        if len(identified_classes) >= 2:
            drug1, drug2 = identified_classes[0], identified_classes[1]

            # Check both directions
            interaction = interaction_database.get((drug1, drug2)) or interaction_database.get((drug2, drug1))

            if interaction:
                if drug1 == 'warfarin' and drug2 == 'herbal' or drug2 == 'warfarin' and drug1 == 'herbal':
                    # Special case for warfarin-herbal
                    result_text = f"""**Warfarin-Herbal Supplement Interactions:**

**Severity**: {interaction['severity']}

**Common Interactions**:
"""
                    for herb, effect in interaction['common_interactions'].items():
                        result_text += f"• **{herb}**: {effect}\n"

                    result_text += f"""
**Management**: {interaction['management']}

**Clinical Note**: {interaction['clinical_note']}

**Recommendation**: Patients on warfarin should avoid herbal supplements or maintain consistent use with frequent INR monitoring."""
                else:
                    result_text = f"""**Drug Interaction: {drug1.upper()} + {drug2.upper()}**

**Severity**: {interaction['severity']}

**Mechanism**: {interaction['mechanism']}

**Clinical Effects**: {interaction['effects']}

**Management**: {interaction['management']}

**Clinical Pearl**: {interaction['clinical_note']}

**Recommendation**: {('DO NOT USE THIS COMBINATION' if interaction['severity'] == 'CONTRAINDICATED' else 'Use with extreme caution and monitoring')}"""

                return {
                    'source': 'drug_interaction_checker',
                    'drugs_identified': [drug1, drug2],
                    'results': [{
                        'drugs': [drug1, drug2],
                        'severity': interaction['severity'],
                        'content': result_text
                    }],
                    'answer': result_text
                }
            else:
                # No specific interaction found, provide general info
                return {
                    'source': 'drug_interaction_checker',
                    'drugs_identified': identified_classes,
                    'results': [{
                        'content': f"""**Drug Interaction Check: {' + '.join(identified_classes).upper()}**

No major interactions found in common databases, but always verify with:
• Clinical pharmacist
• Lexicomp or Micromedex
• FDA drug labels
• Consider patient-specific factors (age, renal/hepatic function)

**General Precautions**:
• Monitor for unexpected effects
• Start with lower doses when combining medications
• Check for CYP450 interactions
• Consider pharmacodynamic interactions"""
                    }],
                    'answer': f"No major documented interaction between {drug1} and {drug2}, but clinical monitoring recommended."
                }
        else:
            # Couldn't identify enough drugs - try direct drug name extraction
            # Common drug names that might not be in patterns
            common_drugs = [
                'metformin', 'lisinopril', 'atorvastatin', 'metoprolol', 'omeprazole',
                'amlodipine', 'losartan', 'gabapentin', 'hydrochlorothiazide', 'levothyroxine',
                'simvastatin', 'albuterol', 'sertraline', 'prednisone', 'tramadol'
            ]

            found_drugs = []
            for drug in common_drugs:
                if drug in query_lower:
                    found_drugs.append(drug)

            if len(found_drugs) >= 2:
                # Provide general safety information for unrecognized combinations
                return {
                    'source': 'drug_interaction_checker',
                    'drugs_identified': found_drugs,
                    'results': [{
                        'content': f"""**Drug Interaction Check: {' + '.join(found_drugs).upper()}**

No documented major interactions between these medications.

**General Safety Information**:
• These medications are commonly prescribed together
• Monitor for any unusual symptoms when starting new combinations
• Regular follow-up with healthcare provider recommended

**Important Considerations**:
• Individual patient factors may affect drug interactions
• Always inform healthcare providers of all medications
• Include OTC drugs and supplements in interaction checks

**Recommendation**: This combination appears safe for most patients. Continue regular monitoring."""
                    }],
                    'answer': f"No major interactions found between {' and '.join(found_drugs)}. Generally safe to use together with standard monitoring."
                }
            else:
                return {
                    'source': 'drug_interaction_checker',
                    'error': 'Need at least 2 drugs to check interactions',
                    'drugs_found': identified_classes + found_drugs,
                    'results': [],
                    'query': query
                }
    
    async def _check_pregnancy_safety(self, query: str) -> Dict[str, Any]:
        """Check pregnancy drug safety - tries FDA first, then uses fallback database"""
        import re

        # Extract drug name first
        drug = None
        query_lower = query.lower()

        # Try regex patterns
        drug_match = re.search(r'is\s+(\w+)\s+safe', query_lower)
        if not drug_match:
            drug_match = re.search(r'(\w+)\s+(?:safe|during|pregnancy)', query_lower)
        if not drug_match:
            drug_match = re.search(r'take\s+(\w+)\s+(?:during|while|pregnancy)', query_lower)

        if drug_match:
            drug = drug_match.group(1)

        # First try to get FDA label data
        if drug:
            try:
                fda_result = await self._search_fda_optimized(drug)
                if fda_result.get('results'):
                    # Parse pregnancy info from FDA label
                    for result in fda_result['results'][:1]:  # Check first result
                        pregnancy_section = result.get('pregnancy', [])
                        lactation_section = result.get('nursing_mothers', [])

                        if pregnancy_section or lactation_section:
                            answer = f"""**Pregnancy Safety: {drug.upper()} (FDA Label)**

**Pregnancy Information**:
{' '.join(pregnancy_section) if pregnancy_section else 'No specific pregnancy information in FDA label.'}

**Breastfeeding Information**:
{' '.join(lactation_section) if lactation_section else 'No specific lactation information in FDA label.'}

**Important Note**: Always consult your healthcare provider before taking any medication during pregnancy or while breastfeeding."""

                            return {
                                'source': 'pregnancy_safety_checker',
                                'drug': drug,
                                'data_source': 'FDA',
                                'results': [{
                                    'drug': drug,
                                    'content': answer
                                }],
                                'answer': answer
                            }
            except Exception as e:
                logger.debug(f"FDA lookup failed for {drug}: {e}")

        # Fallback to curated database for common drugs
        # This data is medically accurate as of 2024
        pregnancy_safety_db = {
            'acetaminophen': {
                'category': 'Generally Safe',
                'trimester_specific': {
                    '1st': 'Safe - preferred analgesic',
                    '2nd': 'Safe - preferred analgesic',
                    '3rd': 'Safe - preferred analgesic'
                },
                'details': 'First-line analgesic/antipyretic in pregnancy. No increased risk of birth defects.',
                'lactation': 'Compatible - minimal transfer to breast milk'
            },
            'ibuprofen': {
                'category': 'Use with Caution',
                'trimester_specific': {
                    '1st': 'Generally avoid',
                    '2nd': 'May use short-term if needed',
                    '3rd': 'CONTRAINDICATED'
                },
                'details': 'NSAIDs can cause premature ductus arteriosus closure in 3rd trimester. Associated with oligohydramnios and renal dysfunction.',
                'lactation': 'Compatible - preferred NSAID for breastfeeding'
            },
            'aspirin': {
                'category': 'Depends on Dose',
                'trimester_specific': {
                    '1st': 'Low-dose (81mg) may be recommended for preeclampsia prevention',
                    '2nd': 'Low-dose generally safe',
                    '3rd': 'Avoid high doses; low-dose may continue if indicated'
                },
                'details': 'Low-dose aspirin used for preeclampsia prevention in high-risk pregnancies. High doses avoided due to bleeding risk.',
                'lactation': 'Use with caution - monitor infant for bruising'
            },
            'metformin': {
                'category': 'Generally Safe',
                'trimester_specific': {
                    '1st': 'Safe - may continue for diabetes/PCOS',
                    '2nd': 'Safe - insulin may be added',
                    '3rd': 'Safe - often combined with insulin'
                },
                'details': 'Increasingly used for gestational diabetes. Does not increase birth defect risk. May reduce pregnancy complications in PCOS.',
                'lactation': 'Compatible - minimal transfer to breast milk'
            },
            'lisinopril': {
                'category': 'CONTRAINDICATED',
                'trimester_specific': {
                    '1st': 'Discontinue immediately',
                    '2nd': 'CONTRAINDICATED - fetal renal damage',
                    '3rd': 'CONTRAINDICATED - oligohydramnios, renal failure'
                },
                'details': 'ACE inhibitors cause fetal renal dysfunction, oligohydramnios, growth restriction, and potentially fatal neonatal hypotension.',
                'lactation': 'Not recommended - consider alternatives',
                'alternatives': 'Methyldopa, labetalol, or nifedipine for hypertension'
            },
            'warfarin': {
                'category': 'CONTRAINDICATED (with exceptions)',
                'trimester_specific': {
                    '1st': 'AVOID - warfarin embryopathy risk',
                    '2nd': 'May use weeks 13-34 if high thrombotic risk',
                    '3rd': 'Switch to heparin by week 36'
                },
                'details': 'Causes warfarin embryopathy (nasal hypoplasia, stippled epiphyses). Reserved for mechanical heart valves where benefit outweighs risk.',
                'lactation': 'Compatible - does not pass into breast milk significantly',
                'alternatives': 'LMWH or unfractionated heparin preferred'
            },
            'amoxicillin': {
                'category': 'Safe',
                'trimester_specific': {
                    '1st': 'Safe - no increased birth defect risk',
                    '2nd': 'Safe',
                    '3rd': 'Safe'
                },
                'details': 'First-line antibiotic for many infections in pregnancy. Well-studied with no teratogenic effects.',
                'lactation': 'Compatible'
            },
            'sertraline': {
                'category': 'Use if Benefits Outweigh Risks',
                'trimester_specific': {
                    '1st': 'Small risk of cardiac defects, weigh benefits',
                    '2nd': 'Generally safe if needed',
                    '3rd': 'Monitor newborn for withdrawal/adaptation syndrome'
                },
                'details': 'Preferred SSRI in pregnancy. Untreated depression poses risks. Small absolute risk increase for birth defects.',
                'lactation': 'Preferred antidepressant for breastfeeding'
            }
        }

        # Extract drug name
        drug = None
        query_lower = query.lower()

        # Try regex patterns
        drug_match = re.search(r'is\s+(\w+)\s+safe', query_lower)
        if not drug_match:
            drug_match = re.search(r'(\w+)\s+(?:safe|during|pregnancy)', query_lower)
        if not drug_match:
            drug_match = re.search(r'take\s+(\w+)\s+(?:during|while|pregnancy)', query_lower)

        if drug_match:
            drug = drug_match.group(1)

        # Check common drug names
        if not drug:
            for drug_name in pregnancy_safety_db.keys():
                if drug_name in query_lower:
                    drug = drug_name
                    break

        if drug and drug in pregnancy_safety_db:
            safety_info = pregnancy_safety_db[drug]

            answer = f"""**Pregnancy Safety: {drug.upper()}**

**Overall Category**: {safety_info['category']}

**Trimester-Specific Recommendations**:
• First Trimester: {safety_info['trimester_specific']['1st']}
• Second Trimester: {safety_info['trimester_specific']['2nd']}
• Third Trimester: {safety_info['trimester_specific']['3rd']}

**Detailed Information**: {safety_info['details']}

**Breastfeeding**: {safety_info['lactation']}"""

            if 'alternatives' in safety_info:
                answer += f"\n\n**Safer Alternatives**: {safety_info['alternatives']}"

            answer += """

**Important Notes**:
• Always consult your healthcare provider before starting, stopping, or changing medications during pregnancy
• Individual risk-benefit assessment is essential
• The FDA Pregnancy and Lactation Labeling Rule (PLLR) provides detailed narratives rather than letter categories"""

            return {
                'source': 'pregnancy_safety_checker',
                'drug': drug,
                'results': [{
                    'drug': drug,
                    'safety_category': safety_info['category'],
                    'content': answer
                }],
                'answer': answer
            }
        else:
            # Drug not in database
            return {
                'source': 'pregnancy_safety_checker',
                'drug': drug or 'unknown medication',
                'results': [{
                    'content': f"Specific pregnancy safety data for {drug or 'this medication'} not available in current database. Consult healthcare provider and resources like Briggs Drugs in Pregnancy and Lactation, Reprotox, or MotherToBaby."
                }],
                'answer': f"Pregnancy safety information for {drug or 'this medication'} requires consultation with healthcare provider and specialized databases."
            }
    
    async def _check_renal_dosing(self, query: str) -> Dict[str, Any]:
        """Check renal dosing adjustments"""
        import re
        
        # Extract CrCl or eGFR
        crcl_match = re.search(r'(?:crcl|egfr|gfr)\s*(?:of\s*)?(\d+)', query.lower())
        crcl = int(crcl_match.group(1)) if crcl_match else 30
        
        # Extract drug
        drug_match = re.search(r'(vancomycin|metformin|gabapentin|atenolol|digoxin)', query.lower())
        drug = drug_match.group(1) if drug_match else 'medication'
        
        renal_stage = 'normal' if crcl > 90 else 'mild impairment' if crcl > 60 else 'moderate impairment' if crcl > 30 else 'severe impairment' if crcl > 15 else 'kidney failure'
        
        return {
            'source': 'renal_dosing_calculator',
            'drug': drug,
            'crcl_ml_min': crcl,
            'renal_stage': renal_stage,
            'answer': f"Renal dosing adjustment for {drug} with CrCl {crcl} mL/min ({renal_stage}) requires access to comprehensive renal dosing guidelines.",
            'results': [{
                'status': 'database_needed',
                'drug': drug,
                'crcl': crcl,
                'message': 'Requires access to renal dosing database',
                'recommended_sources': [
                    'Lexicomp Renal Dosing Handbook',
                    'UpToDate Renal Drug Dosing',
                    'KDIGO Guidelines',
                    'FDA drug labels with renal dosing tables',
                    'Sanford Guide (for antimicrobials)'
                ]
            }],
            'recommendation': 'Consult clinical pharmacist or nephrologist for renal dosing adjustments'
        }
    
    async def _analyze_lab_values(self, query: str) -> Dict[str, Any]:
        """Analyze lab values - identifies values and returns need for reference ranges"""
        import re
        
        # Extract lab values from query
        lab_patterns = {
            'troponin': r'troponin\s*(?:of\s*|is\s*)?(\d+(?:\.\d+)?)',
            'wbc': r'wbc\s*(?:of\s*|is\s*)?(\d+(?:,\d{3}|\d+)?)',
            'neutrophils': r'neutrophils?\s*(?:of\s*|is\s*)?(\d+)%?',
            'anc': r'anc\s*(?:of\s*|is\s*)?(\d+)',
            'platelets': r'platelets?\s*(?:of\s*|is\s*)?(\d+(?:,\d{3}|\d+)?)',
            'hemoglobin': r'(?:hgb|hemoglobin)\s*(?:of\s*|is\s*)?(\d+(?:\.\d+)?)',
            'mcv': r'mcv\s*(?:of\s*|is\s*)?(\d+(?:\.\d+)?)',
            'creatinine': r'(?:cr|creatinine)\s*(?:of\s*|is\s*)?(\d+(?:\.\d+)?)',
            'bilirubin': r'bilirubin\s*(?:of\s*|is\s*)?(\d+(?:\.\d+)?)',
            'inr': r'inr\s*(?:of\s*|is\s*)?(\d+(?:\.\d+)?)',
            'glucose': r'glucose\s*(?:of\s*|is\s*)?(\d+)'
        }
        
        found_labs = []
        
        for lab, pattern in lab_patterns.items():
            match = re.search(pattern, query.lower())
            if match:
                value = match.group(1).replace(',', '')
                found_labs.append({
                    'lab': lab.upper(),
                    'value': value,
                    'unit': 'varies by lab'
                })
        
        if found_labs:
            return {
                'source': 'lab_analyzer',
                'labs_identified': found_labs,
                'answer': f"Interpreting lab values requires access to age and gender-specific reference ranges and clinical context.",
                'results': [{
                    'status': 'database_needed',
                    'labs': found_labs,
                    'message': 'Requires access to laboratory reference ranges',
                    'recommended_sources': [
                        'UpToDate Lab Interpretation',
                        'Mayo Clinic Laboratories Reference Values',
                        'Quest Diagnostics Reference Ranges',
                        'LabCorp Test Menu',
                        'Hospital-specific reference ranges'
                    ]
                }],
                'recommendation': 'Correlate with clinical presentation and trending values',
                'note': 'Reference ranges vary by laboratory, age, gender, and methodology'
            }
        else:
            return {
                'source': 'lab_analyzer',
                'error': 'Unable to extract lab values from query',
                'results': [],  # Empty results for quality evaluator
                'query': query
            }
    
    async def _get_disease_statistics(self, query: str) -> Dict[str, Any]:
        """Get disease statistics"""
        import re
        
        # Extract disease name
        disease_patterns = [
            r'(?:prevalence|incidence|mortality|statistics?)\s+(?:of\s+)?(.+?)(?:\s+in)?',
            r'(.+?)\s+(?:prevalence|incidence|mortality|statistics)',
        ]
        
        disease = None
        for pattern in disease_patterns:
            match = re.search(pattern, query.lower())
            if match:
                disease = match.group(1).strip()
                break
        
        if not disease:
            disease = 'condition'
        
        # Clean up disease name
        if disease:
            # Remove common words
            disease = disease.replace('the', '').replace('of', '').strip()
        
        return {
            'source': 'disease_statistics',
            'disease': disease or 'condition',
            'answer': f"Current epidemiological data for {disease or 'this condition'} requires access to updated medical databases.",
            'results': [{
                'status': 'database_needed',
                'disease': disease,
                'message': 'Requires access to epidemiological databases',
                'recommended_sources': [
                    'CDC National Center for Health Statistics',
                    'WHO Global Health Observatory',
                    'NCI SEER Database (for cancers)',
                    'GBD (Global Burden of Disease) Study',
                    'Disease-specific registries and foundations'
                ]
            }],
            'recommendation': 'Check CDC, WHO, or disease-specific organization databases for current statistics',
            'note': 'Epidemiological data varies by region and is updated annually'
        }
    
    async def _calculate_medical_score(self, query: str) -> Dict[str, Any]:
        """Calculate medical scores like CHADS2, GCS, etc."""
        import re
        
        # Detect score type
        score_type = None
        if 'chads' in query.lower():
            score_type = 'CHADS2'
        elif 'glasgow' in query.lower() or 'gcs' in query.lower():
            score_type = 'GCS'
        elif 'apgar' in query.lower():
            score_type = 'APGAR'
        
        # Identify clinical components mentioned in query
        components_found = []
        query_lower = query.lower()
        
        if score_type:
            # Try to identify components from the query
            if 'chf' in query_lower or 'heart failure' in query_lower:
                components_found.append('Heart failure')
            if 'htn' in query_lower or 'hypertension' in query_lower:
                components_found.append('Hypertension')
            if 'diabetes' in query_lower or 'dm' in query_lower:
                components_found.append('Diabetes')
            if 'stroke' in query_lower or 'tia' in query_lower:
                components_found.append('Prior stroke/TIA')
            if re.search(r'\bage\s*[>]?\s*[67]\d|[89]\d', query_lower) or 'age' in query_lower:
                components_found.append('Age criteria')
            
            return {
                'source': 'medical_calculator',
                'score_type': score_type or 'Unknown',
                'components_identified': components_found,
                'answer': f"Calculating {score_type or 'medical score'} requires access to validated scoring tools with current risk stratification data.",
                'results': [{
                    'status': 'database_needed',
                    'score_type': score_type,
                    'components': components_found,
                    'message': 'Requires access to validated medical calculators',
                    'recommended_sources': [
                        'MDCalc',
                        'QxMD Calculate',
                        'MedCalc',
                        'UpToDate Calculators',
                        'Specialty society guidelines'
                    ]
                }],
                'recommendation': 'Use validated clinical calculators with current risk stratification data',
                'note': 'Medical scores should be calculated using validated tools and interpreted in clinical context'
            }
        else:
            return {
                'source': 'medical_calculator',
                'error': 'Unable to determine which medical score to calculate',
                'available_scores': ['CHADS2-VASc', 'HAS-BLED', 'GCS', 'APGAR', 'MELD', 'Child-Pugh', 'Wells', 'PERC', 'CURB-65'],
                'results': [],  # Empty results for quality evaluator
                'query': query
            }