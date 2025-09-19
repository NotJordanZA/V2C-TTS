"""
Ollama-based LLM processor for character text transformation.

This module provides an OllamaProcessor class that uses Ollama for local
language model inference, offering an easier alternative to llama.cpp.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama package not available. Ollama functionality will be disabled.")


class OllamaProcessor:
    """
    Handles local LLM inference using Ollama for character text transformation.
    
    Ollama is easier to set up than llama.cpp and handles model management automatically.
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2:1b",
        host: str = "http://localhost:11434",
        timeout: float = 30.0,
        **kwargs
    ):
        """
        Initialize Ollama processor.
        
        Args:
            model_name: Name of the Ollama model to use (e.g., "llama3.2:1b", "phi3:mini")
            host: Ollama server host URL
            timeout: Request timeout in seconds
            **kwargs: Additional arguments for Ollama client
        """
        self.model_name = model_name
        self.host = host
        self.timeout = timeout
        self.kwargs = kwargs
        
        self._client: Optional[ollama.Client] = None
        self._model_loaded = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ollama")
        self._lock = threading.Lock()
        
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "ollama package is not installed. "
                "Install it with: pip install ollama"
            )
    
    def load_model(self) -> None:
        """
        Initialize Ollama client and ensure model is available.
        
        Raises:
            RuntimeError: If Ollama server is not running or model loading fails
        """
        if self._model_loaded:
            logger.info("Ollama client already initialized")
            return
        
        try:
            logger.info(f"Initializing Ollama client for model {self.model_name}")
            
            # Initialize Ollama client
            self._client = ollama.Client(host=self.host, **self.kwargs)
            
            # Check if Ollama server is running
            try:
                models = self._client.list()
                logger.info(f"Connected to Ollama server, found {len(models['models'])} models")
            except Exception as e:
                raise RuntimeError(
                    f"Cannot connect to Ollama server at {self.host}. "
                    f"Make sure Ollama is installed and running. Error: {e}"
                )
            
            # Check if model exists, pull if not
            model_list = models.models if hasattr(models, 'models') else models.get('models', [])
            model_exists = any(
                getattr(model, 'model', model.get('name', '')) == self.model_name 
                for model in model_list
            )
            
            if not model_exists:
                logger.info(f"Model {self.model_name} not found, pulling...")
                try:
                    self._client.pull(self.model_name)
                    logger.info(f"Successfully pulled model {self.model_name}")
                except Exception as e:
                    raise RuntimeError(f"Failed to pull model {self.model_name}: {e}")
            
            self._model_loaded = True
            logger.info(f"Ollama processor ready with model {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama processor: {e}")
            raise RuntimeError(f"Failed to initialize Ollama processor: {e}")
    
    def unload_model(self) -> None:
        """Cleanup Ollama client."""
        with self._lock:
            if self._client is not None:
                self._client = None
                self._model_loaded = False
                logger.info("Ollama client cleaned up")
    
    def _generate_sync(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Synchronous text generation using Ollama.
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop: List of stop sequences
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If model is not loaded or generation fails
        """
        if not self._model_loaded or self._client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            with self._lock:
                # Prepare options for Ollama
                options = {
                    'num_predict': max_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    'top_k': top_k,
                    **kwargs
                }
                
                # Add stop sequences if provided
                if stop:
                    options['stop'] = stop
                
                response = self._client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options=options,
                    stream=False
                )
            
            generated_text = response['response'].strip()
            logger.debug(f"Generated {len(generated_text)} characters with Ollama")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Ollama text generation failed: {e}")
            raise RuntimeError(f"Ollama text generation failed: {e}")
    
    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Asynchronous text generation using Ollama.
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop: List of stop sequences
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        loop = asyncio.get_event_loop()
        
        return await loop.run_in_executor(
            self._executor,
            self._generate_sync,
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            stop,
            **kwargs
        )
    
    def create_character_prompt(
        self,
        original_text: str,
        character_prompt: str,
        intensity: float = 1.0
    ) -> str:
        """
        Create a prompt for character transformation.
        
        Args:
            original_text: Original text to transform
            character_prompt: Character-specific transformation prompt template
            intensity: Transformation intensity (0.0-2.0)
            
        Returns:
            Formatted prompt for LLM
        """
        # Adjust intensity in the prompt
        intensity_descriptions = {
            0.0: "very subtly",
            0.5: "slightly", 
            1.0: "moderately",
            1.5: "strongly",
            2.0: "very strongly"
        }
        
        # Find closest intensity description
        closest_intensity = min(intensity_descriptions.keys(), 
                              key=lambda x: abs(x - intensity))
        intensity_desc = intensity_descriptions[closest_intensity]
        
        # Format the character prompt with intensity
        formatted_prompt = character_prompt.format(
            text=original_text,
            intensity=intensity_desc
        )
        
        # Add system instructions for better output
        system_prompt = (
            "You are a text transformation assistant. Transform the given text "
            "according to the character description. Keep the core meaning intact "
            "but change the style, vocabulary, and tone to match the character. "
            "Respond with ONLY the transformed text, no explanations or quotes.\n\n"
        )
        
        return system_prompt + formatted_prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Ollama model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.model_name,
            "model_loaded": self._model_loaded,
            "host": self.host,
            "timeout": self.timeout,
            "processor_type": "ollama"
        }
        
        if self._model_loaded and self._client:
            try:
                models = self._client.list()
                model_list = models.models if hasattr(models, 'models') else models.get('models', [])
                model_info = next(
                    (m for m in model_list if getattr(m, 'model', m.get('name', '')) == self.model_name),
                    None
                )
                if model_info:
                    info.update({
                        "model_size": getattr(model_info, 'size', model_info.get('size', 'unknown')),
                        "model_modified": getattr(model_info, 'modified_at', model_info.get('modified_at', 'unknown'))
                    })
            except Exception as e:
                logger.warning(f"Could not get detailed model info: {e}")
        
        return info
    
    def list_available_models(self) -> List[str]:
        """
        List available models on the Ollama server.
        
        Returns:
            List of available model names
        """
        if not self._model_loaded or self._client is None:
            return []
        
        try:
            models = self._client.list()
            model_list = models.models if hasattr(models, 'models') else models.get('models', [])
            return [getattr(model, 'model', model.get('name', '')) for model in model_list]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_model()
        self._executor.shutdown(wait=True)
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.unload_model()
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors during deletion


def create_ollama_processor(model_name: str = "llama3.2:1b", **kwargs) -> OllamaProcessor:
    """
    Factory function to create Ollama processor.
    
    Args:
        model_name: Name of the Ollama model to use
        **kwargs: Additional arguments for OllamaProcessor
        
    Returns:
        OllamaProcessor instance
        
    Raises:
        ImportError: If ollama package is not available
    """
    if not OLLAMA_AVAILABLE:
        raise ImportError(
            "ollama package is not installed. "
            "Install it with: pip install ollama"
        )
    
    return OllamaProcessor(model_name, **kwargs)