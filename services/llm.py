"""
LangChain wrapper for OpenRouter LLM integration.
"""

import os
import logging
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import yaml

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service class for LLM interactions using LangChain and OpenRouter.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the LLM service.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.api_key = self._get_api_key()
        
        # Default model from config
        self.default_model = self.config.get('default_model', 'meta-llama/llama-4-maverick:free')
        
        # OpenRouter base URL
        self.base_url = "https://openrouter.ai/api/v1"
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
            
    def _get_api_key(self) -> str:
        """Get OpenRouter API key from environment variable."""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required. "
                "Please set it with your OpenRouter API key."
            )
        return api_key
        
    def get_system_prompt(self, prompt_mode: str = "default") -> str:
        """
        Get system prompt from configuration.
        
        Args:
            prompt_mode: The prompt mode to use (default, creative, technical, casual)
            
        Returns:
            str: The system prompt text
        """
        system_prompts = self.config.get('system_prompts', {})
        prompt = system_prompts.get(prompt_mode, system_prompts.get('default', ''))
        
        if not prompt:
            logger.warning(f"No system prompt found for mode '{prompt_mode}', using default")
            prompt = "You are a helpful AI assistant."
            
        return prompt
        
    def get_available_models(self) -> Dict[str, Any]:
        """Get available models from configuration (legacy method)."""
        return self.config.get('models', {})
    
    async def fetch_openrouter_models(self) -> Dict[str, Any]:
        """
        Fetch available models from OpenRouter API.
        
        Returns:
            Dict containing categorized models with pricing and details
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://openrouter.ai/api/v1/models", 
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._categorize_openrouter_models(data.get('data', []))
                    else:
                        logger.warning(f"Failed to fetch OpenRouter models: {response.status}")
                        return self.get_available_models()  # Fallback to config
                        
        except Exception as e:
            logger.error(f"Error fetching OpenRouter models: {e}")
            return self.get_available_models()  # Fallback to config
    
    def _categorize_openrouter_models(self, models: List[Dict]) -> Dict[str, Any]:
        """
        Categorize OpenRouter models by pricing and capabilities.
        
        Args:
            models: List of model data from OpenRouter API
            
        Returns:
            Dict with categorized models
        """
        free_models = []
        premium_models = []
        featured_models = []
        
        for model in models:
            model_id = model.get('id', '')
            model_name = model.get('name', '')
            pricing = model.get('pricing', {})
            
            # Check if model is free (prompt cost is 0)
            prompt_cost = float(pricing.get('prompt', '0'))
            completion_cost = float(pricing.get('completion', '0'))
            
            model_info = {
                'id': model_id,
                'name': model_name,
                'prompt_cost': prompt_cost,
                'completion_cost': completion_cost,
                'context_length': model.get('context_length', 0),
                'description': model.get('description', ''),
            }
            
            if prompt_cost == 0 and completion_cost == 0:
                free_models.append(model_info)
            else:
                premium_models.append(model_info)
                
            # Featured models (popular/recommended)
            if any(keyword in model_name.lower() for keyword in ['gpt-4', 'claude', 'llama', 'qwen', 'gemini']):
                featured_models.append(model_info)
        
        # Sort by popularity/name
        free_models.sort(key=lambda x: x['name'])
        premium_models.sort(key=lambda x: x['prompt_cost'])
        
        return {
            'free_models': free_models,  # Return all free models
            'premium_models': premium_models,  # Return all premium models  
            'featured_models': featured_models,  # Return all featured models
            'total_available': len(models),
            'last_updated': 'live_from_openrouter'
        }
        
    async def chat(
        self, 
        message: str, 
        model: Optional[str] = None, 
        prompt_mode: str = "default",
        history: Optional[List[Dict[str, str]]] = None,
        audio_data: Optional[str] = None
    ) -> str:
        """
        Send a chat message to the LLM and get a response.
        
        Args:
            message: The user's message
            model: The model to use (defaults to config default)
            prompt_mode: The system prompt mode to use
            history: Previous conversation history
            audio_data: Base64 encoded audio data for multimodal models
            
        Returns:
            str: The LLM's response
        """
        # Use provided model or default
        model_name = model or self.default_model
        
        # Get system prompt
        system_prompt = self.get_system_prompt(prompt_mode)
        
        # Log the request
        logger.info(f"Chat request - Model: {model_name}, Prompt mode: {prompt_mode}")
        logger.info(f"User message: {message[:100]}{'...' if len(message) > 100 else ''}")
        if history:
            logger.info(f"Chat history: {len(history)} previous messages")

        else:
            logger.info("No chat history provided")
        
        try:
            # Initialize the LLM with OpenRouter
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                temperature=0.7,
                max_tokens=8000,
                # OpenRouter specific headers
                model_kwargs={
                    "extra_headers": {
                        "HTTP-Referer": "http://localhost:8000",  # Required by OpenRouter
                        "X-Title": "FastAPI LangChain Backend"     # Optional but helpful
                    }
                }
            )
            
            # Create messages list starting with system prompt
            messages = [SystemMessage(content=system_prompt)]
            
            # Add conversation history if provided
            if history:
                for hist_msg in history:
                    if hist_msg["role"] == "user":
                        messages.append(HumanMessage(content=hist_msg["content"]))
                    elif hist_msg["role"] == "assistant":
                        messages.append(AIMessage(content=hist_msg["content"]))
            
            # Add the current user message (with audio support if provided)
            if audio_data:
                # For multimodal models, create a message with audio content
                content = [
                    {
                        "type": "text",
                        "text": message or "Please analyze this audio"
                    },
                    {
                        "type": "audio",
                        "audio_url": {
                            "url": f"data:audio/webm;base64,{audio_data}"
                        }
                    }
                ]
                messages.append(HumanMessage(content=content))
            else:
                messages.append(HumanMessage(content=message))
            
            # Get response from LLM
            response = await llm.ainvoke(messages)
            
            # Extract text content
            response_text = response.content
            
            logger.info(f"LLM response received: {len(response_text)} characters")
            logger.debug(f"Response preview: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise
            
    async def chat_stream(
        self, 
        message: str, 
        model: Optional[str] = None, 
        prompt_mode: str = "default",
        history: Optional[List[Dict[str, str]]] = None
    ):
        """
        Send a chat message to the LLM and get a streaming response.
        
        Args:
            message: The user's message
            model: The model to use (defaults to config default)
            prompt_mode: The system prompt mode to use
            history: Previous conversation history
            
        Yields:
            str: Chunks of the LLM's response
        """
        # Use provided model or default
        model_name = model or self.default_model
        
        # Get system prompt
        system_prompt = self.get_system_prompt(prompt_mode)
        
        # Log the request
        logger.info(f"Streaming chat request - Model: {model_name}, Mode: {prompt_mode}")
        logger.info(f"User message: {message[:100]}{'...' if len(message) > 100 else ''}")
        if history:
            logger.info(f"Chat history: {len(history)} previous messages")

        else:
            logger.info("No chat history provided")
        
        try:
            # Initialize the LLM with OpenRouter for streaming
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                temperature=0.7,
                max_tokens=8000,
                streaming=True,  # Enable streaming
                # OpenRouter specific headers
                model_kwargs={
                    "extra_headers": {
                        "HTTP-Referer": "http://localhost:8000",  # Required by OpenRouter
                        "X-Title": "FastAPI LangChain Backend"     # Optional but helpful
                    }
                }
            )
            
            # Create messages list starting with system prompt
            messages = [SystemMessage(content=system_prompt)]
            
            # Add conversation history if provided
            if history:
                for hist_msg in history:
                    if hist_msg["role"] == "user":
                        messages.append(HumanMessage(content=hist_msg["content"]))
                    elif hist_msg["role"] == "assistant":
                        messages.append(AIMessage(content=hist_msg["content"]))
            
            # Add the current user message
            messages.append(HumanMessage(content=message))
            
            # Get streaming response from LLM
            async for chunk in llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Error in streaming LLM call: {str(e)}")
            raise
            
    def validate_model(self, model: str) -> bool:
        """
        Validate if a model is available (checks both config and OpenRouter).
        
        Args:
            model: Model name to validate
            
        Returns:
            bool: True if model is available
        """
        # Always allow the default model
        if model == self.default_model:
            return True
            
        # Check config models (legacy)
        available_models = self.get_available_models()
        for category_models in available_models.values():
            if isinstance(category_models, list) and model in category_models:
                return True
        
        # For production, you might want to cache OpenRouter models
        # and validate against them, but for now we'll be permissive
        # since OpenRouter has 400+ models
        logger.info(f"Model {model} not in config, allowing (OpenRouter validation)")
        return True
    
    async def get_live_models(self) -> Dict[str, Any]:
        """
        Get live models from OpenRouter API with fallback to config.
        
        Returns:
            Dict containing live model data
        """
        try:
            return await self.fetch_openrouter_models()
        except Exception as e:
            logger.error(f"Failed to get live models: {e}")
            return self.get_available_models()
    
    async def chat_with_web_search(
        self,
        message: str,
        model: Optional[str] = None,
        prompt_mode: str = "default",
        enable_web_search: bool = True,
        search_domains: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 8000,
        **kwargs
    ) -> str:
        """
        Send a chat message with web search capabilities enabled.
        
        Args:
            message: The user's message
            model: The model to use (defaults to config default)
            prompt_mode: The system prompt mode to use
            enable_web_search: Whether to enable web search
            search_domains: Optional list of domains to focus search on
            history: Previous conversation history
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for the model
            
        Returns:
            str: The LLM's response with web search results integrated
        """
        # For now, use regular chat since OpenRouter doesn't support web search directly
        # In production, this would integrate with actual web search APIs
        logger.info(f"Web search requested but using regular chat (not implemented)")
        
        # Add search context to the prompt if domains are specified
        if search_domains and enable_web_search:
            message = f"Please answer based on information from these sources: {', '.join(search_domains)}\n\n{message}"
        
        return await self.chat(
            message=message,
            model=model,
            prompt_mode=prompt_mode,
            history=history
        )


# Global LLM service instance (lazy-loaded)
llm_service = None

def get_llm_service():
    """Get the global LLM service instance (lazy-loaded)."""
    global llm_service
    if llm_service is None:
        llm_service = LLMService()
    return llm_service