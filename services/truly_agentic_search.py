"""
Truly Agentic Medical Search System
- Evaluates result quality
- Automatically retries with different tools
- Expands queries when needed
- Chains tools based on context
"""

import asyncio
import json
import time
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from services.optimized_search_fixed import OptimizedMedicalSearchFixed

# Optional: Import OpenAI for LLM tool calling
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import FDA service if available
try:
    from services.fda_api_service import FDAApiService
    FDA_AVAILABLE = True
except ImportError:
    FDA_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResultQuality(Enum):
    """Quality levels for search results"""
    EXCELLENT = "excellent"  # Comprehensive, relevant results
    GOOD = "good"           # Adequate information
    POOR = "poor"           # Some results but insufficient
    EMPTY = "empty"         # No useful results


@dataclass
class SearchState:
    """Tracks the state of an agentic search"""
    query: str
    original_query: str
    tools_tried: List[str]
    results_collected: Dict[str, Any]
    quality_scores: Dict[str, ResultQuality]
    iteration: int
    max_iterations: int = 5  # Increased for thoroughness
    total_time: float = 0
    
    def should_continue(self) -> bool:
        """Determine if we should continue searching"""
        # Stop if we have excellent results
        if ResultQuality.EXCELLENT in self.quality_scores.values():
            return False
        
        # Stop if we've tried too many times
        if self.iteration >= self.max_iterations:
            return False
            
        # Stop if we've taken too long (but allow more time for thoroughness)
        if self.total_time > 30:  # 30 second max for thorough search
            return False
            
        # Continue if we have poor/empty results
        poor_count = sum(1 for q in self.quality_scores.values() 
                        if q in [ResultQuality.POOR, ResultQuality.EMPTY])
        return poor_count > len(self.quality_scores) / 2


class TrulyAgenticSearch(OptimizedMedicalSearchFixed):
    """
    Truly agentic search that adapts based on results
    """
    
    def __init__(self, llm_service=None, use_llm_tool_calling=False):
        super().__init__(llm_service)
        self.use_llm_tool_calling = use_llm_tool_calling and OPENAI_AVAILABLE
        
        # Initialize OpenAI client if requested
        if self.use_llm_tool_calling:
            self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.model = "gpt-4o-mini"  # Fast and efficient for tool calling
        
        # ALL available medical tools
        self.all_tools = [
            'search_pubmed',           # PubMed research articles
            'search_fda',              # FDA drug information
            'search_clinical_trials',  # ClinicalTrials.gov
            'web_search',              # General medical web search
            'calculate_pediatric_dose', # Pediatric dosing calculator
            'check_drug_interactions', # Drug interaction checker
            'analyze_source',          # Analyze medical sources
            'get_statistics',          # Get medical statistics
            'search_clinical_guidelines', # Clinical practice guidelines
            'get_drug_information',    # Detailed drug info
            'calculate_medical_score', # Medical scoring calculators
            'get_disease_statistics',  # Disease prevalence data
            'analyze_lab_values',      # Lab result interpretation
            'check_pregnancy_safety',  # Pregnancy drug safety
            'check_renal_dosing'       # Renal dosing adjustments
        ]
    
    async def execute_agentic(self, query: str, streaming_callback=None) -> Dict[str, Any]:
        """
        Execute a truly agentic search with iterative refinement
        Can use either pattern matching or LLM tool calling
        """
        # If LLM tool calling is enabled, use that approach
        if self.use_llm_tool_calling:
            return await self._execute_with_llm_tools(query, streaming_callback)
        
        # Otherwise use the existing pattern-based approach
        start_time = time.time()
        
        # Initialize search state
        state = SearchState(
            query=query,
            original_query=query,
            tools_tried=[],
            results_collected={},
            quality_scores={},
            iteration=0
        )
        
        # Main agentic loop
        while state.iteration < state.max_iterations:
            state.iteration += 1
            
            if streaming_callback:
                await streaming_callback(f"Iteration {state.iteration}: Analyzing query and selecting tools...")
            
            # Step 1: Determine what tools to use based on current state
            tools_to_use = await self._select_next_tools(state)
            
            if not tools_to_use:
                logger.info("No more tools to try")
                break
            
            if streaming_callback:
                await streaming_callback(f"Trying tools: {', '.join(tools_to_use)}")
            
            # Step 2: Execute the selected tools
            results = await self.search_parallel(state.query, tools_to_use)
            
            # Step 3: Evaluate result quality
            for tool, result in results.items():
                quality = self._evaluate_result_quality(result)
                state.quality_scores[tool] = quality
                state.results_collected[tool] = result
                state.tools_tried.append(tool)
                
                if streaming_callback:
                    await streaming_callback(f"{tool}: {quality.value} quality")
            
            # Step 4: Check if we need to refine the query
            if self._should_refine_query(state):
                state.query = await self._refine_query(state)
                if streaming_callback:
                    await streaming_callback(f"Refined query: {state.query}")
            
            # Step 5: Check if we have good enough results to stop
            if self._has_sufficient_results(state):
                if streaming_callback:
                    await streaming_callback("Found sufficient results, completing search...")
                break
            
            # Step 6: Check if we should try additional tools
            if self._should_expand_search(state):
                if streaming_callback:
                    await streaming_callback("Expanding search to additional sources...")
            
            state.total_time = time.time() - start_time
        
        # Step 6: Synthesize final answer from all collected results
        if streaming_callback:
            await streaming_callback("Synthesizing comprehensive answer from all sources...")
        
        final_answer = await self._synthesize_comprehensive(state)
        
        return {
            'answer': final_answer,
            'tools_used': state.tools_tried,
            'iterations': state.iteration,
            'quality_assessment': {k: v.value for k, v in state.quality_scores.items()},
            'raw_results': state.results_collected,
            'execution_time': time.time() - start_time,
            'was_refined': state.query != state.original_query
        }
    
    async def _select_next_tools(self, state: SearchState) -> List[str]:
        """
        Intelligently select which tools to try next based on current state
        """
        untried_tools = [t for t in self.all_tools if t not in state.tools_tried]
        
        if state.iteration == 1:
            # First iteration: Smart tool selection based on query patterns
            return await self._smart_tool_selection(state.query)
        
        # Subsequent iterations: Choose based on what's missing
        if not untried_tools:
            return []
        
        # Analyze what type of information is missing
        missing_info = self._analyze_gaps(state)
        
        # Map missing info to appropriate tools
        tools_to_try = []
        
        if "clinical_data" in missing_info and "search_clinical_trials" in untried_tools:
            tools_to_try.append("search_clinical_trials")
            
        if "drug_info" in missing_info and "search_fda" in untried_tools:
            tools_to_try.append("search_fda")
            
        if "research" in missing_info and "search_pubmed" in untried_tools:
            tools_to_try.append("search_pubmed")
            
        if not tools_to_try and "web_search" in untried_tools:
            tools_to_try.append("web_search")
        
        return tools_to_try[:2]  # Try at most 2 new tools per iteration
    
    async def _smart_tool_selection(self, query: str) -> List[str]:
        """
        Smart initial tool selection based on query pattern matching
        """
        query_lower = query.lower()
        tools = []
        
        # Check for pediatric dosing
        if any(word in query_lower for word in ['pediatric', 'child', 'kg', 'dose', 'dosing']) and \
           any(word in query_lower for word in ['amoxicillin', 'ibuprofen', 'acetaminophen', 'medication']):
            tools.append('calculate_pediatric_dose')
        
        # Check for drug interactions
        if any(phrase in query_lower for phrase in ['interaction', 'with', 'between', 'can i take']):
            tools.append('check_drug_interactions')
        
        # Check for pregnancy
        if any(word in query_lower for word in ['pregnant', 'pregnancy', 'safe during pregnancy']):
            tools.append('check_pregnancy_safety')
        
        # Check for renal dosing
        if any(word in query_lower for word in ['crcl', 'egfr', 'renal', 'kidney', 'creatinine clearance']):
            tools.append('check_renal_dosing')
        
        # Check for lab values
        if any(word in query_lower for word in ['troponin', 'wbc', 'neutrophil', 'elevated', 'lab', 'blood test']):
            tools.append('analyze_lab_values')
        
        # Check for disease statistics
        if any(word in query_lower for word in ['prevalence', 'incidence', 'mortality', 'statistics', 'rate']):
            tools.append('get_disease_statistics')
        
        # Check for medical scores
        if any(word in query_lower for word in ['chads', 'glasgow', 'gcs', 'apgar', 'score', 'calculate']):
            tools.append('calculate_medical_score')
        
        # If no specialized tool matched, use general search based on content
        if not tools:
            if 'drug' in query_lower or 'medication' in query_lower:
                tools.extend(['search_fda', 'search_pubmed'])
            elif 'trial' in query_lower or 'study' in query_lower:
                tools.extend(['search_clinical_trials', 'search_pubmed'])
            elif 'guideline' in query_lower or 'treatment' in query_lower:
                tools.extend(['search_pubmed', 'web_search'])
            else:
                # Default fallback
                tools.extend(['search_pubmed', 'web_search'])
        
        # Limit to 3 tools initially
        return tools[:3]
    
    def _evaluate_result_quality(self, result: Dict[str, Any]) -> ResultQuality:
        """
        Evaluate the quality of a search result more thoroughly
        """
        if 'error' in result:
            return ResultQuality.EMPTY

        # Check source type for specialized tools
        source = result.get('source', '')

        # Special handling for drug tools
        if source in ['pediatric_calculator', 'drug_interaction_checker']:
            # These tools provide focused, specific information
            if 'answer' in result and result['answer'] and len(result['answer']) > 100:
                # Check if it's actual medical info vs placeholder
                answer_lower = result['answer'].lower()
                if 'database_needed' in answer_lower or 'requires access' in answer_lower:
                    return ResultQuality.EMPTY
                return ResultQuality.GOOD
            if 'results' in result and result['results']:
                for r in result['results']:
                    if isinstance(r, dict) and 'content' in r and len(r['content']) > 100:
                        return ResultQuality.GOOD
            return ResultQuality.POOR

        if 'results' not in result and 'answer' not in result:
            return ResultQuality.EMPTY

        # Check result count
        results = result.get('results', [])

        if not results:
            # Special case: web_search might have an answer without results array
            if 'answer' in result and result['answer'] and len(result['answer']) > 50:
                return ResultQuality.GOOD
            return ResultQuality.EMPTY

        # Check if results have actual content (not just IDs)
        content_score = 0
        valid_results = 0

        for r in results[:5]:  # Check up to 5 results
            if isinstance(r, dict):
                # Check for meaningful content fields with better logic
                # Handle both strings and lists for FDA API compatibility
                def get_content_safely(field):
                    value = r.get(field, '')
                    if isinstance(value, list):
                        # For FDA results which return lists
                        return ' '.join(str(v) for v in value)
                    return str(value)

                has_abstract = 'abstract' in r and get_content_safely('abstract').strip()
                has_content = 'content' in r and get_content_safely('content').strip()
                has_text = 'text' in r and get_content_safely('text').strip()
                has_title = 'title' in r and get_content_safely('title').strip()
                has_description = 'description' in r and get_content_safely('description').strip()

                # FDA specific fields
                has_fda_data = any(field in r for field in ['indications_and_usage', 'dosage_and_administration', 'warnings'])

                # Score based on content richness
                if has_fda_data:
                    # FDA data is rich and structured
                    content_score += 3  # FDA labels are comprehensive
                    valid_results += 1
                elif has_abstract or has_content or has_text:
                    # Full content - abstracts are valuable even if short
                    content_text = get_content_safely('abstract') + get_content_safely('content') + get_content_safely('text')
                    content_length = len(content_text)
                    if content_length > 200:  # Substantial content
                        content_score += 3
                    elif content_length > 50:   # Moderate content
                        content_score += 2
                    else:  # Some content
                        content_score += 1
                    valid_results += 1
                elif has_title or has_description:
                    content_score += 1  # Partial content
                    valid_results += 1
                elif 'pmid' in r or 'nct_id' in r or 'doi' in r:
                    content_score += 0.5  # Just identifiers but valid references
                    valid_results += 1

        # Evaluate based on content richness and number of valid results
        if content_score >= 8 and valid_results >= 3:  # Multiple high-quality results
            return ResultQuality.EXCELLENT
        elif content_score >= 4 and valid_results >= 2:  # Some good content
            return ResultQuality.GOOD
        elif content_score >= 1 and valid_results >= 1:  # Limited but some content
            return ResultQuality.POOR
        else:
            return ResultQuality.EMPTY
    
    def _should_refine_query(self, state: SearchState) -> bool:
        """
        Determine if query refinement would help
        """
        # Refine if most results are poor/empty and we haven't refined yet
        poor_ratio = sum(1 for q in state.quality_scores.values() 
                        if q in [ResultQuality.POOR, ResultQuality.EMPTY]) / max(len(state.quality_scores), 1)
        
        return poor_ratio > 0.7 and state.query == state.original_query
    
    async def _refine_query(self, state: SearchState) -> str:
        """
        Refine the query to get better results
        """
        if not self.llm_service:
            # Simple fallback: add context words
            if "latest" not in state.query.lower():
                return f"latest research {state.query}"
            elif "treatment" not in state.query.lower():
                return f"{state.query} treatment guidelines"
            else:
                return f"{state.query} clinical studies"
        
        # Use LLM to refine
        prompt = f"""
        Original query: "{state.original_query}"
        
        The search returned insufficient results. Please refine this query to be more specific
        and likely to return medical information. Add relevant medical terms or context.
        
        Return only the refined query, nothing else.
        """
        
        try:
            refined = await self.llm_service.chat(prompt)
            return refined.strip()
        except:
            return state.query
    
    def _has_sufficient_results(self, state: SearchState) -> bool:
        """
        Check if we have sufficient results to stop searching
        """
        # Check if we have at least one excellent result
        if ResultQuality.EXCELLENT in state.quality_scores.values():
            return True
        
        # Check if we have multiple good results
        good_count = sum(1 for q in state.quality_scores.values() 
                        if q in [ResultQuality.EXCELLENT, ResultQuality.GOOD])
        return good_count >= 2
    
    def _should_expand_search(self, state: SearchState) -> bool:
        """
        Determine if we should try additional tools
        """
        # Expand if we don't have good results yet
        has_good = any(q in [ResultQuality.EXCELLENT, ResultQuality.GOOD] 
                      for q in state.quality_scores.values())
        
        # And we haven't tried all tools
        return not has_good and len(state.tools_tried) < len(self.all_tools)
    
    def _summarize_quality(self, results_collected: Dict) -> str:
        """Summarize current result quality for display"""
        if not results_collected:
            return "No results yet"

        quality_counts = {'excellent': 0, 'good': 0, 'poor': 0, 'empty': 0}
        for tool, result in results_collected.items():
            quality = self._evaluate_result_quality(result)
            quality_counts[quality.value] += 1

        parts = []
        if quality_counts['excellent'] > 0:
            parts.append(f"{quality_counts['excellent']} excellent")
        if quality_counts['good'] > 0:
            parts.append(f"{quality_counts['good']} good")
        if quality_counts['poor'] > 0:
            parts.append(f"{quality_counts['poor']} poor")
        if quality_counts['empty'] > 0:
            parts.append(f"{quality_counts['empty']} empty")

        if parts:
            return f"Found {', '.join(parts)} result{'s' if sum(quality_counts.values()) > 1 else ''}"
        return "Gathering initial results"

    def _analyze_gaps(self, state: SearchState) -> List[str]:
        """
        Analyze what type of information is missing
        """
        gaps = []
        
        # Check what we have
        has_clinical = any("clinical" in tool for tool in state.tools_tried)
        has_drug = any("fda" in tool for tool in state.tools_tried)
        has_research = any("pubmed" in tool for tool in state.tools_tried)
        
        if not has_clinical:
            gaps.append("clinical_data")
        if not has_drug and "drug" in state.query.lower():
            gaps.append("drug_info")
        if not has_research:
            gaps.append("research")
            
        return gaps
    
    async def _execute_with_llm_tools(self, query: str, streaming_callback=None) -> Dict[str, Any]:
        """
        Execute search using OpenAI GPT-4o-mini with function calling
        The LLM decides which tools to call
        """
        start_time = time.time()
        
        if streaming_callback:
            await streaming_callback("🤖 Starting AI-powered medical search...")
            await streaming_callback(f"  Query: \"{query[:100]}...\"" if len(query) > 100 else f"  Query: \"{query}\"")
        
        # Define tools for OpenAI function calling
        tools = self._get_openai_tool_definitions()
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": """You are a medical information assistant with access to various medical tools.
                IMPORTANT: You MUST use tools to gather information for ALL medical queries.
                Even for common conditions, search for the latest guidelines and evidence-based information.

                MANDATORY MINIMUM SEARCHES:
                - For ANY medical query, you MUST ALWAYS use BOTH:
                  1. search_pubmed - for scientific evidence and research
                  2. web_search - for practical guidelines and current practices
                - These are the MINIMUM - you should use additional tools as needed

                Additional tool guidelines:
                - For medications: Also use search_fda and check_drug_interactions
                - For ongoing studies: Also use search_clinical_trials
                - For dosing: Use calculate_pediatric_dose if relevant
                - For pregnancy: Use check_pregnancy_safety if relevant

                ITERATION DECISION MAKING:
                - In your FIRST iteration, call AT LEAST search_pubmed AND web_search
                - You can call them together in parallel for efficiency
                - After getting results from both, decide if you need more tools
                - Continue searching if you need more specific details or verification
                - Stop when you have sufficient information from multiple sources
                - Balance thoroughness with efficiency

                IMPORTANT: Never provide medical advice based on just one source. Always cross-reference PubMed (scientific) with web search (practical guidelines).

                CRITICAL: When encountering medical abbreviations or informal language:
                - Expand abbreviations to full medical terms (e.g., FH → Familial Hypercholesterolemia, HTN → Hypertension, AFib → Atrial Fibrillation)
                - Convert informal queries to proper medical terminology before searching
                - Use the expanded terms when calling search tools, NOT the abbreviations
                - Consider context from previous queries when interpreting abbreviations
                - For personal/family history queries, search for both the condition AND inheritance patterns

                Example: If user asks about "FH", search for "familial hypercholesterolemia" not "FH"
                Example: If user mentions "my dad has FH", search for "familial hypercholesterolemia inheritance genetic risk"

                Important: Always verify drug names, dosages, and medical information from multiple sources when possible."""
            },
            {"role": "user", "content": query}
        ]
        
        tools_used = []
        results_collected = {}
        iterations = 0
        max_iterations = 5  # Safety limit, but we'll stop when satisfied
        satisfied = False

        while iterations < max_iterations and not satisfied:
            iterations += 1

            if streaming_callback:
                if iterations == 1:
                    await streaming_callback(f"\n🔍 Beginning search...")
                else:
                    # Show why we're continuing
                    if results_collected:
                        quality_summary = self._summarize_quality(results_collected)
                        await streaming_callback(f"\n🔄 Continuing search (Iteration {iterations})")
                        await streaming_callback(f"  Current status: {quality_summary}")
                    else:
                        await streaming_callback(f"\n🔄 Iteration {iterations}")
                await streaming_callback(f"  Analyzing query and selecting tools...")
            
            try:
                # Call OpenAI with tools
                if streaming_callback:
                    await streaming_callback(f"  Selecting optimal tools for this query...")

                response = await self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=1000
                )
                
                message = response.choices[0].message
                
                # Check if LLM wants to call tools
                if message.tool_calls:
                    if streaming_callback:
                        tools_to_call = [tc.function.name.replace('_', ' ').title() for tc in message.tool_calls]
                        await streaming_callback(f"  Selected {len(message.tool_calls)} tool{'s' if len(message.tool_calls) > 1 else ''}: {', '.join(tools_to_call)}")
                        await streaming_callback(f"\n🔧 Executing tools...")

                    # Execute tool calls
                    tool_results = await self._execute_llm_tool_calls(
                        message.tool_calls, streaming_callback
                    )
                    
                    # Track tools used
                    for tc in message.tool_calls:
                        tool_name = tc.function.name
                        tools_used.append(tool_name)
                        results_collected[tool_name] = tool_results[tc.id]
                    
                    # Add assistant message with tool calls
                    messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in message.tool_calls
                        ]
                    })
                    
                    # Add tool results to messages
                    for tc_id, result in tool_results.items():
                        # Limit result size to avoid token limits
                        result_str = json.dumps(result)
                        if len(result_str) > 4000:
                            # Truncate large results
                            if isinstance(result, dict) and 'results' in result:
                                truncated = result.copy()
                                truncated['results'] = result['results'][:3]  # Keep first 3
                                truncated['truncated'] = True
                                result_str = json.dumps(truncated)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": result_str
                        })
                    
                    if streaming_callback:
                        tool_names = [tc.function.name.replace('_', ' ').title() for tc in message.tool_calls]
                        await streaming_callback(f"  ✓ Completed {len(message.tool_calls)} tool{'s' if len(message.tool_calls) > 1 else ''}: {', '.join(tool_names)}")
                else:
                    # LLM has no more tools to call - this means IT decided to stop
                    satisfied = True
                    if streaming_callback:
                        await streaming_callback(f"  ✓ AI determined sufficient information gathered")
                    break

                # Optional: Add quality information to context for LLM's next decision
                # But DON'T force stop based on our rules - let the LLM decide
                if results_collected and streaming_callback:
                    quality_scores = []
                    for tool, result in results_collected.items():
                        quality = self._evaluate_result_quality(result)
                        quality_scores.append(quality)

                    excellent_count = sum(1 for q in quality_scores if q == ResultQuality.EXCELLENT)
                    good_count = sum(1 for q in quality_scores if q == ResultQuality.GOOD)
                    empty_count = sum(1 for q in quality_scores if q == ResultQuality.EMPTY)

                    # Just report the quality, don't make decisions
                    quality_summary = []
                    if excellent_count > 0:
                        quality_summary.append(f"{excellent_count} excellent")
                    if good_count > 0:
                        quality_summary.append(f"{good_count} good")
                    if empty_count > 0:
                        quality_summary.append(f"{empty_count} empty")

                    if quality_summary:
                        await streaming_callback(f"  Quality: {', '.join(quality_summary)} results")

                    # Only stop if ALL attempts have failed completely
                    if len(quality_scores) >= 3 and empty_count == len(quality_scores):
                        # This is a safety check - if everything is empty, no point continuing
                        satisfied = True
                        if streaming_callback:
                            await streaming_callback(f"  ⚠️ No results found from any sources, stopping")
                        break
                    
            except Exception as e:
                logger.error(f"Error in LLM tool calling: {str(e)}")
                if streaming_callback:
                    await streaming_callback(f"  ⚠️ Error: {str(e)[:100]}")
                # Don't break on error if we have some results
                if results_collected:
                    satisfied = True
                break
        
        # Get final synthesis from LLM
        if streaming_callback:
            total_results = sum(len(r.get('results', [])) if isinstance(r, dict) else 0 for r in results_collected.values())
            await streaming_callback(f"\n📝 Final Step: Synthesizing comprehensive answer from {len(tools_used)} source{'s' if len(tools_used) != 1 else ''} ({total_results} total results)...")
        
        try:
            final_response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages + [
                    {"role": "user", "content": "Please provide a comprehensive answer based on all the information gathered."}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            final_answer = final_response.choices[0].message.content
        except:
            # Fallback to existing synthesis method
            state = SearchState(
                query=query,
                original_query=query,
                tools_tried=tools_used,
                results_collected=results_collected,
                quality_scores={},
                iteration=iterations
            )
            final_answer = await self._synthesize_results(query, results_collected)
        
        # Evaluate quality of results for display
        quality_assessment = {}
        for tool, result in results_collected.items():
            quality = self._evaluate_result_quality(result)
            quality_assessment[tool] = quality.value

        return {
            'answer': final_answer,
            'tools_used': tools_used,
            'iterations': iterations,
            'quality_assessment': quality_assessment,
            'execution_time': time.time() - start_time,
            'method': 'llm_tool_calling',
            'raw_results': results_collected
        }
    
    def _get_openai_tool_definitions(self) -> List[Dict]:
        """
        Get tool definitions in OpenAI function calling format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_pubmed",
                    "description": "Search PubMed for medical research articles. IMPORTANT: Use full medical terms, not abbreviations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query using full medical terms (e.g., 'familial hypercholesterolemia' not 'FH')"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_fda",
                    "description": "Search FDA database for drug information, labels, and safety data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Drug name using full terms (e.g., 'atorvastatin' not shortcuts)"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_drug_interactions",
                    "description": "Check for interactions between medications",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Query mentioning drugs to check"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_pediatric_dose",
                    "description": "Calculate pediatric medication dosing",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Query with drug and weight"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_renal_dosing",
                    "description": "Check renal dosing adjustments for medications",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Query with drug and renal function"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_pregnancy_safety",
                    "description": "Check if medications are safe during pregnancy",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Query about drug and pregnancy"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_lab_values",
                    "description": "Interpret laboratory test results",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Query with lab values"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_medical_score",
                    "description": "Calculate medical risk scores (CHADS2, MELD, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Query with score type and parameters"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_clinical_trials",
                    "description": "Search for clinical trials. IMPORTANT: Use full medical terms, not abbreviations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Condition or treatment using full medical terms (e.g., 'familial hypercholesterolemia' not 'FH')"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "General medical web search for additional information. IMPORTANT: Use full medical terms, not abbreviations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query using full medical terms (e.g., 'familial hypercholesterolemia inheritance' not 'FH genetic')"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    async def _execute_llm_tool_calls(self, tool_calls, streaming_callback=None) -> Dict[str, Any]:
        """
        Execute tool calls requested by the LLM with detailed progress
        """
        results = {}

        for i, tool_call in enumerate(tool_calls, 1):
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # Extract the query/search terms being used
            search_query = tool_args.get('query', tool_args.get('condition', tool_args.get('drug_name', '')))

            # Format tool name for display
            display_name = tool_name.replace('_', ' ').title()

            if streaming_callback:
                # Show detailed progress with search terms
                await streaming_callback(f"Step {i}/{len(tool_calls)}: {display_name}")
                if search_query:
                    await streaming_callback(f"  → Search terms: \"{search_query[:100]}...\" " if len(search_query) > 100 else f"  → Search terms: \"{search_query}\"")
            
            # Map to actual tool execution
            try:
                # Special handling for FDA tools if available
                if tool_name == "search_fda" and FDA_AVAILABLE:
                    async with FDAApiService() as fda:
                        drug_name = tool_args.get('query', '').replace('search_fda ', '')
                        result = await fda.search_drug_label(drug_name)
                        results[tool_call.id] = result
                elif tool_name == "check_drug_interaction" and FDA_AVAILABLE:
                    # Extract two drugs from query
                    query = tool_args.get('query', '')
                    # Simple extraction - can be improved
                    drugs = re.findall(r'\b(\w+)\b', query.lower())
                    if len(drugs) >= 2:
                        async with FDAApiService() as fda:
                            result = await fda.check_drug_interactions(drugs[0], drugs[1])
                            results[tool_call.id] = result
                    else:
                        results[tool_call.id] = {'error': 'Need at least 2 drugs to check interactions'}
                elif tool_name in self.all_tools:
                    # Use existing tool implementations
                    if streaming_callback:
                        await streaming_callback(f"  → Searching...")

                    result = await self.search_parallel(
                        tool_args.get('query', ''),
                        [tool_name]
                    )
                    results[tool_call.id] = result.get(tool_name, {'error': 'No result'})

                    # Report result quality
                    if streaming_callback:
                        quality = self._evaluate_result_quality(result.get(tool_name, {}))
                        quality_emoji = {
                            ResultQuality.EXCELLENT: '🟢',
                            ResultQuality.GOOD: '🟡',
                            ResultQuality.POOR: '🟠',
                            ResultQuality.EMPTY: '🔴'
                        }.get(quality, '⚪')
                        result_count = len(result.get(tool_name, {}).get('results', []))
                        await streaming_callback(f"  → Found {result_count} results {quality_emoji} ({quality.value})")
                else:
                    results[tool_call.id] = {'error': f'Unknown tool: {tool_name}'}
                    if streaming_callback:
                        await streaming_callback(f"  ⚠️ Unknown tool: {tool_name}")
            except Exception as e:
                results[tool_call.id] = {'error': str(e)}
                if streaming_callback:
                    await streaming_callback(f"  ❌ Error: {str(e)[:50]}")
        
        return results
    
    async def _synthesize_comprehensive(self, state: SearchState) -> str:
        """
        Synthesize a comprehensive answer from all collected results
        """
        # Use ALL results, not just "good" ones - even poor results may have valuable info
        all_results = state.results_collected

        # But prioritize good results by including quality context
        good_results = {
            tool: data for tool, data in state.results_collected.items()
            if state.quality_scores.get(tool, ResultQuality.EMPTY) in
            [ResultQuality.EXCELLENT, ResultQuality.GOOD]
        }

        poor_results = {
            tool: data for tool, data in state.results_collected.items()
            if state.quality_scores.get(tool, ResultQuality.EMPTY) in
            [ResultQuality.POOR]
        }

        if not all_results:
            return "I was unable to find sufficient information to answer your query. Please try rephrasing or being more specific."

        # Combine good and poor results, prioritizing good ones
        results_to_synthesize = {}
        results_to_synthesize.update(good_results)  # Add good results first

        # Add poor results if we don't have enough good ones
        if len(good_results) < 2:
            results_to_synthesize.update(poor_results)

        if not results_to_synthesize:
            # Last resort - use everything we have
            results_to_synthesize = all_results

        # Add quality context to help LLM understand source reliability
        for tool, data in results_to_synthesize.items():
            quality = state.quality_scores.get(tool, ResultQuality.EMPTY)
            data['_quality_score'] = quality.value  # Add quality metadata

        # Use parent class synthesis with enhanced results
        return await self._synthesize_results(state.original_query, results_to_synthesize)