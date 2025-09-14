#!/usr/bin/env python
"""
Improved Agentic Medical Search CLI with Rich markdown rendering
Enhanced with better performance and visual output
"""

import asyncio
import sys
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
from dotenv import load_dotenv

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.progress import track, Progress, SpinnerColumn, TextColumn
from rich.tree import Tree
from rich.syntax import Syntax
from rich.live import Live
from rich.layout import Layout
from rich.prompt import Prompt, Confirm
from rich import print as rprint

# Load environment variables
load_dotenv()

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.truly_agentic_search import TrulyAgenticSearch
from services.optimized_search_fixed import OptimizedMedicalSearchFixed
from services.llm import LLMService


class ImprovedAgenticCLI:
    """Enhanced CLI with Rich library for beautiful output"""
    
    def __init__(self, use_optimized=True, use_streaming=True, use_truly_agentic=True, use_llm_tool_calling=True):
        self.console = Console()
        self.llm_service = None
        self.optimized_search = None
        self.truly_agentic_search = None
        self.use_optimized = use_optimized
        self.use_streaming = use_streaming
        self.use_truly_agentic = use_truly_agentic
        self.use_llm_tool_calling = use_llm_tool_calling
        self.history = []
        self.session_start = datetime.now()
        self.verbose = True
        
    async def initialize(self):
        """Initialize services with progress display"""
        with self.console.status("[bold green]Initializing Agentic Medical Search System...") as status:
            try:
                # Initialize LLM service
                status.update("[yellow]Initializing LLM Service...")
                self.llm_service = LLMService()
                self.console.print("[green][OK] LLM Service initialized[/green]")
                
                # Initialize search system
                if self.use_truly_agentic:
                    status.update("[yellow]Initializing Truly Agentic Search System...")
                    self.truly_agentic_search = TrulyAgenticSearch(
                        llm_service=self.llm_service,
                        use_llm_tool_calling=self.use_llm_tool_calling
                    )
                    self.console.print("[green][OK] Truly Agentic Search System initialized[/green]")
                    if self.use_llm_tool_calling:
                        self.console.print("[green][OK] LLM Tool Calling (GPT-4o-mini) enabled[/green]")
                    # Also initialize optimized for fallback
                    self.optimized_search = self.truly_agentic_search  # Inherits from OptimizedMedicalSearchFixed
                elif self.use_optimized:
                    status.update("[yellow]Initializing Optimized Search System...")
                    # Use the fixed version with better memory management
                    self.optimized_search = OptimizedMedicalSearchFixed(llm_service=self.llm_service)
                    self.console.print("[green][OK] Optimized Search System (Fixed) initialized[/green]")
                else:
                    # Fallback to truly agentic search
                    status.update("[yellow]Initializing Truly Agentic Search System (fallback)...")
                    self.truly_agentic_search = TrulyAgenticSearch(
                        llm_service=self.llm_service,
                        use_llm_tool_calling=self.use_llm_tool_calling
                    )
                    self.console.print("[green][OK] Truly Agentic Search System initialized (fallback)[/green]")
                
                return True
                
            except Exception as e:
                self.console.print(f"[red]X Initialization failed: {str(e)}[/red]")
                return False
    
    def print_header(self):
        """Print beautiful header with Rich"""
        header = Panel.fit(
            "[bold cyan]AGENTIC MEDICAL SEARCH SYSTEM[/bold cyan]\n" +
            "[yellow]Enhanced with Markdown Rendering[/yellow]\n" +
            f"[dim]Session: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(header)
        self.console.print("\nType [bold green]help[/bold green] for commands, [bold red]exit[/bold red] to quit\n")
    
    def print_help(self):
        """Print help with Rich formatting"""
        help_md = """
# Available Commands

## Navigation
- **help** - Show this help message
- **clear** - Clear the screen  
- **exit** - Exit the program

## Features
- **history** - Show conversation history
- **tools** - List available tools
- **sources** - List search sources
- **verbose** - Toggle detailed output
- **streaming** - Toggle streaming responses (show progress)
- **save** - Save conversation to file

## Example Queries

### Drug Information
- What are the side effects of metformin?
- Drug interactions between warfarin and aspirin

### Treatment Guidelines  
- First-line treatment for type 2 diabetes
- Compare ACE inhibitors vs ARBs for hypertension

### Clinical Trials
- Latest clinical trials for Alzheimer's disease
- CAR-T therapy trials for solid tumors

### Pediatric Dosing
- Calculate amoxicillin dose for 25kg child
- Vaccine schedule for immunocompromised children
        """
        self.console.print(Markdown(help_md))
    
    def print_tools(self):
        """Print available tools as a table"""
        table = Table(title="Available Medical Search Tools", show_header=True, header_style="bold cyan")
        table.add_column("Tool", style="green", width=20)
        table.add_column("Description", style="white")
        table.add_column("Source", style="yellow")
        
        tools = [
            ("search_pubmed", "Search PubMed medical database", "NCBI"),
            ("search_fda", "Search FDA drug information", "FDA"),
            ("search_clinical_trials", "Search clinical trials database", "ClinicalTrials.gov"),
            ("check_drug_interactions", "Check drug-drug interactions", "Internal DB"),
            ("calculate_pediatric_dose", "Calculate pediatric dosing", "Calculator"),
            ("web_search", "General medical web search", "Web"),
        ]
        
        for tool, desc, source in tools:
            table.add_row(tool, desc, source)
        
        self.console.print(table)
    
    def format_tool_execution(self, tool_name: str, params: Dict, result: Any) -> Tree:
        """Format tool execution as a tree"""
        tree = Tree(f"[bold cyan]Tool: {tool_name}[/bold cyan]")
        
        # Add parameters
        params_branch = tree.add("Parameters")
        for key, value in params.items():
            params_branch.add(f"[yellow]{key}:[/yellow] {value}")
        
        # Add result summary
        result_branch = tree.add("Result")
        if isinstance(result, dict):
            if 'results' in result:
                result_branch.add(f"[green]Found {len(result.get('results', []))} results[/green]")
            elif 'answer' in result:
                result_branch.add("[green]Generated answer[/green]")
            else:
                result_branch.add("[yellow]Data retrieved[/yellow]")
        else:
            result_branch.add(f"[dim]{str(result)[:100]}...[/dim]")
        
        return tree
    
    async def process_query(self, query: str):
        """Process query with beautiful progress display"""
        start_time = time.time()

        # Don't repeat the query since it's already shown in the prompt
        
        if self.use_truly_agentic and self.truly_agentic_search:
            # Use truly agentic search with iterative refinement
            await self.process_query_truly_agentic(query, start_time)
        elif self.use_streaming and self.use_optimized:
            # Stream results as they arrive
            await self.process_query_streaming(query, start_time)
        else:
            # Original non-streaming behavior
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("[cyan]Analyzing query...", total=None)
                
                try:
                    # Update progress
                    progress.update(task, description="[yellow]Planning search strategy...")
                    
                    # Execute search
                    if self.use_optimized and self.optimized_search:
                        result = await self.optimized_search.execute_with_reasoning(query)
                    elif self.truly_agentic_search:
                        result = await self.truly_agentic_search.execute_agentic(query)
                    else:
                        raise Exception("No search system initialized")
                    
                    progress.update(task, description="[green]Generating response...")
                    
                    # Store in history
                    self.history.append({
                        'type': 'user',
                        'content': query,
                        'timestamp': datetime.now()
                    })
                    
                    self.history.append({
                        'type': 'agent', 
                        'content': result,
                        'timestamp': datetime.now()
                    })
                    
                    progress.stop()
                    
                    # Display results
                    await self.display_results(result, time.time() - start_time)
                    
                except Exception as e:
                    progress.stop()
                    self.console.print(f"\n[red]Error: {str(e)}[/red]")
                    if self.verbose:
                        self.console.print_exception()
    
    async def display_results(self, result: Dict, execution_time: float):
        """Display results with Rich formatting"""

        # Skip execution time here as it's shown above
        pass
        
        # Show performance stats if using optimized search
        if self.use_optimized and 'performance_stats' in result:
            stats = result['performance_stats']
            cache_eff = result.get('cache_efficiency', 0)
            self.console.print(f"[dim]Cache: {cache_eff:.1f}% | Requests: {stats.get('total_requests', 0)} | Errors: {stats.get('errors', 0)}[/dim]")
        
        # Show tool usage if verbose
        if self.verbose and 'reasoning_trace' in result:
            trace = result['reasoning_trace']
            
            # Show reasoning tree
            if 'initial_reasoning' in trace:
                reasoning = trace['initial_reasoning']
                tree = Tree("[bold]Reasoning Process[/bold]")
                tree.add(f"Query Type: [yellow]{reasoning.get('query_type', 'unknown')}[/yellow]")
                
                if 'key_concepts' in reasoning:
                    concepts = tree.add("Key Concepts")
                    for concept in reasoning['key_concepts'][:5]:
                        concepts.add(f"[cyan]{concept}[/cyan]")
                
                self.console.print(tree)
            
            # Show tool executions
            if 'execution_history' in trace:
                exec_tree = Tree("[bold]Tool Executions[/bold]")
                for exec_item in trace['execution_history']:
                    if 'tool_name' in exec_item:
                        tool_branch = exec_tree.add(f"[cyan]{exec_item['tool_name']}[/cyan]")
                        if 'interpretation' in exec_item:
                            interp = exec_item['interpretation']
                            if 'key_findings' in interp:
                                for finding in interp['key_findings'][:2]:
                                    tool_branch.add(f"[green]+[/green] {finding[:80]}...")
                
                self.console.print(exec_tree)
        
        # Display main answer without panel, just markdown
        if 'answer' in result and result['answer']:
            self.console.print("\n[bold]Answer:[/bold]\n")
            self.console.print(Markdown(result['answer']))
        else:
            self.console.print("[yellow]No answer generated. Try rephrasing your query.[/yellow]")
        
        # Show sources if available
        if 'sources' in result and result['sources']:
            self.console.print("\n[bold]Sources:[/bold]")
            for source in result['sources'][:5]:
                self.console.print(f"  - [blue]{source}[/blue]")
    
    def show_history(self):
        """Show conversation history with formatting"""
        if not self.history:
            self.console.print("[yellow]No conversation history yet.[/yellow]")
            return
        
        for item in self.history[-10:]:  # Last 10 items
            timestamp = item['timestamp'].strftime('%H:%M:%S')
            if item['type'] == 'user':
                self.console.print(f"\n[dim]{timestamp}[/dim] [bold cyan]You:[/bold cyan]")
                self.console.print(f"  {item['content']}")
            else:
                self.console.print(f"\n[dim]{timestamp}[/dim] [bold green]Agent:[/bold green]")
                if 'answer' in item['content']:
                    # Show truncated answer
                    answer = item['content']['answer'][:200] + "..."
                    self.console.print(f"  {answer}")
    
    async def process_query_truly_agentic(self, query: str, start_time: float):
        """Process query with truly agentic iterative search"""
        try:
            # Simple chat-like display without panels
            self.console.print("\n[bold green]Assistant:[/bold green]")

            async def streaming_callback(message: str):
                """Callback to show agentic progress"""
                # Format messages based on type
                if "🤖" in message or "Starting" in message:
                    self.console.print(f"[bold cyan]{message}[/bold cyan]")
                elif "🔍" in message or "Beginning" in message:
                    self.console.print(f"[bold green]{message}[/bold green]")
                elif "🔄" in message or "Continuing" in message:
                    self.console.print(f"[bold yellow]{message}[/bold yellow]")
                elif "🔧" in message or "Executing" in message:
                    self.console.print(f"[bold green]{message}[/bold green]")
                elif "Step" in message:
                    self.console.print(f"[bold magenta]{message}[/bold magenta]")
                elif "→ Search terms:" in message:
                    self.console.print(f"[dim cyan]{message}[/dim cyan]")
                elif "→ Searching..." in message:
                    self.console.print(f"[dim yellow]{message}[/dim yellow]")
                elif "→ Found" in message:
                    if "🟢" in message:
                        self.console.print(f"[green]{message}[/green]")
                    elif "🟡" in message:
                        self.console.print(f"[yellow]{message}[/yellow]")
                    elif "🟠" in message:
                        self.console.print(f"[orange1]{message}[/orange1]")
                    elif "🔴" in message:
                        self.console.print(f"[red]{message}[/red]")
                    else:
                        self.console.print(f"[dim]{message}[/dim]")
                elif "✓" in message:
                    self.console.print(f"[green]{message}[/green]")
                elif "📝" in message:
                    self.console.print(f"[bold blue]{message}[/bold blue]")
                elif "Selected" in message:
                    self.console.print(f"[yellow]{message}[/yellow]")
                elif "❌" in message or "Error" in message:
                    self.console.print(f"[red]{message}[/red]")
                elif "⚠️" in message:
                    self.console.print(f"[orange1]{message}[/orange1]")
                else:
                    self.console.print(f"[dim]{message}[/dim]")

            # Execute truly agentic search
            result = await self.truly_agentic_search.execute_agentic(query, streaming_callback)

            # Final summary without panel
            execution_time = time.time() - start_time

            self.console.print(f"\n[bold green]✅ Search Complete![/bold green]")
            self.console.print(f"[cyan]Total time:[/cyan] {execution_time:.1f}s")
            self.console.print(f"[cyan]Iterations:[/cyan] {result.get('iterations', 1)}")
            self.console.print(f"[cyan]Tools used:[/cyan] {', '.join(result.get('tools_used', []))}")

            # Show quality assessment if available
            if 'quality_assessment' in result:
                self.console.print(f"\n[bold]Quality Assessment:[/bold]")
                for tool, quality in result['quality_assessment'].items():
                    emoji = {
                        'excellent': '🟢',
                        'good': '🟡',
                        'poor': '🟠',
                        'empty': '🔴'
                    }.get(quality, '⚪')
                    self.console.print(f"  {emoji} {tool}: {quality}")

            if result.get('was_refined'):
                self.console.print("\n[yellow]Note: Query was refined for better results[/yellow]")
            
            # Store in history
            self.history.append({
                'type': 'user',
                'content': query,
                'timestamp': datetime.now()
            })
            
            self.history.append({
                'type': 'agent', 
                'content': result,
                'timestamp': datetime.now()
            })
            
            # Display final results
            await self.display_results(result, execution_time)
            
        except Exception as e:
            self.console.print(f"\n[red]Error during agentic search: {str(e)}[/red]")
            if self.verbose:
                self.console.print_exception()
    
    async def process_query_streaming(self, query: str, start_time: float):
        """Process query with streaming responses showing progress in real-time"""
        try:
            # Simple streaming without panels
            self.console.print("\n[bold green]Assistant:[/bold green]")
            self.console.print("\n[yellow]Analyzing query and selecting tools...[/yellow]")

            # Step 1: Determine tools
            tools_to_use = await self.optimized_search._determine_tools(query)

            # Update display with tool selection
            tools_display = ", ".join([f"[green]{tool}[/green]" for tool in tools_to_use])
            self.console.print(f"[green]Tools selected:[/green] {tools_display}")
            self.console.print(f"\n[yellow]Searching sources in parallel...[/yellow]")

            # Step 2: Execute tools in parallel with progress tracking
            search_results = {}
            completed_tools = []

            # Create tasks for parallel execution
            async def execute_with_tracking(tool):
                result = await self.optimized_search._execute_tool_safe(tool, query)
                completed_tools.append(tool)

                # Update display as each tool completes
                self.console.print(f"  [green]✓[/green] Completed: {tool}")
                return tool, result

            # Execute all tools in parallel
            tasks = [execute_with_tracking(tool) for tool in tools_to_use]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for item in results:
                if isinstance(item, tuple):
                    tool, result = item
                    search_results[tool] = result
                elif isinstance(item, Exception):
                    self.console.print(f"[red]Error in tool execution: {item}[/red]")

            self.console.print(f"\n[green]Search completed![/green] Found results from {len(completed_tools)} sources")
            self.console.print(f"[yellow]Synthesizing medical information...[/yellow]")

            # Step 3: Synthesize results
            answer = await self.optimized_search._synthesize_results(query, search_results)

            # Final update
            execution_time = time.time() - start_time
            self.console.print(f"\n[green]✅ Answer generated![/green]")
            self.console.print(f"[dim]Total time: {execution_time:.1f} seconds[/dim]")
            
            # Build final result
            result = {
                'answer': answer,
                'tools_used': tools_to_use,
                'raw_results': search_results,
                'execution_time': execution_time,
                'performance_stats': self.optimized_search.performance_stats.copy() if hasattr(self.optimized_search, 'performance_stats') else {},
                'cache_efficiency': self.optimized_search._calculate_cache_efficiency() if hasattr(self.optimized_search, '_calculate_cache_efficiency') else 0
            }
            
            # Store in history
            self.history.append({
                'type': 'user',
                'content': query,
                'timestamp': datetime.now()
            })
            
            self.history.append({
                'type': 'agent', 
                'content': result,
                'timestamp': datetime.now()
            })
            
            # Display final results
            await self.display_results(result, execution_time)
            
        except Exception as e:
            self.console.print(f"\n[red]Error during streaming: {str(e)}[/red]")
            if self.verbose:
                self.console.print_exception()
    
    def save_conversation(self):
        """Save conversation to file"""
        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(self.history, f, indent=2, default=str)
            self.console.print(f"[green]Conversation saved to {filename}[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to save: {e}[/red]")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.optimized_search:
            await self.optimized_search.cleanup()
            self.console.print("[dim]Cleaned up resources[/dim]")
    
    async def run(self):
        """Main CLI loop"""
        if not await self.initialize():
            return
        
        self.print_header()
        
        try:
            while True:
                try:
                    # Get user input with Rich prompt
                    query = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                    
                    # Handle commands
                    if query.lower() == 'exit':
                        if Confirm.ask("Are you sure you want to exit?"):
                            self.console.print("\n[yellow]Goodbye![/yellow]\n")
                            break
                            
                    elif query.lower() == 'help':
                        self.print_help()
                        
                    elif query.lower() == 'clear':
                        os.system('cls' if os.name == 'nt' else 'clear')
                        self.print_header()
                        
                    elif query.lower() == 'tools':
                        self.print_tools()
                        
                    elif query.lower() == 'history':
                        self.show_history()
                        
                    elif query.lower() == 'verbose':
                        self.verbose = not self.verbose
                        status = "enabled" if self.verbose else "disabled"
                        self.console.print(f"Verbose mode [yellow]{status}[/yellow]")
                        
                    elif query.lower() == 'streaming':
                        self.use_streaming = not self.use_streaming
                        status = "enabled" if self.use_streaming else "disabled"
                        self.console.print(f"Streaming mode [yellow]{status}[/yellow]")
                        
                    elif query.lower() == 'save':
                        self.save_conversation()
                        
                    elif query.lower() == 'sources':
                        sources = ["PubMed", "FDA", "ClinicalTrials.gov", "Medical Literature"]
                        self.console.print("\n[bold]Available Sources:[/bold]")
                        for source in sources:
                            self.console.print(f"  - [blue]{source}[/blue]")
                    
                    elif query.strip():
                        # Process medical query
                        await self.process_query(query)
                        
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Use 'exit' command to quit[/yellow]")
                except Exception as e:
                    self.console.print(f"\n[red]Error: {str(e)}[/red]")
                    if self.verbose:
                        self.console.print_exception()
        finally:
            # Clean up resources on exit
            await self.cleanup()


def main():
    """Main entry point"""
    # Enable truly agentic search by default for best results
    cli = ImprovedAgenticCLI(use_optimized=True, use_streaming=True, use_truly_agentic=True)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()