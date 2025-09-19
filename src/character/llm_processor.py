"""
Local LLM integration for character text transformation.

This module provides the LLMProcessor class that handles local language model
inference using llama.cpp Python bindings for character text transformation.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not available. LLM functionality will be disabled.")


class LLMProcessor:
    """
    Handles local LLM inference for character text transformation.
    
    Uses llama.cpp Python bindings to run language models locally with
    GPU acceleration and quantization support.
    """
    
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize LLM processor.
        
        Args:
            model_path: Path to the GGUF model file
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_ctx: Context window size
            n_threads: Number of CPU threads to use (None for auto)
            verbose: Enable verbose logging
            **kwargs: Additional arguments for Llama initialization
        """
        self.model_path = Path(model_path)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.verbose = verbose
        self.kwargs = kwargs
        
        self._llama: Optional[Llama] = None
        self._model_loaded = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
        self._lock = threading.Lock()
        
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python"
            )
    
    def load_model(self) -> None:
        """
        Load the LLM model.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        if self._model_loaded:
            logger.info("Model already loaded")
            return
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            logger.info(f"Loading LLM model from {self.model_path}")
            
            # Determine optimal thread count if not specified
            n_threads = self.n_threads
            if n_threads is None:
                import os
                n_threads = min(8, os.cpu_count() or 4)
            
            self._llama = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_threads=n_threads,
                verbose=self.verbose,
                **self.kwargs
            )
            
            self._model_loaded = True
            logger.info(f"Successfully loaded LLM model with {n_threads} threads")
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise RuntimeError(f"Failed to load LLM model: {e}")
    
    def unload_model(self) -> None:
        """Unload the LLM model to free memory."""
        with self._lock:
            if self._llama is not None:
                # llama.cpp doesn't have explicit cleanup, but we can clear the reference
                self._llama = None
                self._model_loaded = False
                logger.info("Unloaded LLM model")
    
    def _generate_sync(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Synchronous text generation.
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repeat_penalty: Repetition penalty
            stop: List of stop sequences
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If model is not loaded or generation fails
        """
        if not self._model_loaded or self._llama is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            with self._lock:
                response = self._llama(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=stop or [],
                    echo=False,
                    **kwargs
                )
            
            generated_text = response['choices'][0]['text'].strip()
            logger.debug(f"Generated {len(generated_text)} characters")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {e}")
    
    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Asynchronous text generation.
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repeat_penalty: Repetition penalty
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
            repeat_penalty,
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
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_path": str(self.model_path),
            "model_loaded": self._model_loaded,
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "model_exists": self.model_path.exists()
        }
    
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


class MockLLMProcessor:
    """
    Mock LLM processor for testing and development when llama.cpp is not available.
    """
    
    def __init__(self, model_path: str, **kwargs):
        """Initialize mock LLM processor."""
        self.model_path = Path(model_path)
        self._model_loaded = False
        logger.info("Using mock LLM processor (llama.cpp not available)")
    
    def load_model(self) -> None:
        """Mock model loading."""
        self._model_loaded = True
        logger.info("Mock: Model loaded")
    
    def unload_model(self) -> None:
        """Mock model unloading."""
        self._model_loaded = False
        logger.info("Mock: Model unloaded")
    
    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        **kwargs
    ) -> str:
        """
        Mock text generation that returns a simple transformation.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens (ignored)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Mock transformed text
        """
        # Simple mock transformation - just add some character flair
        if "anime" in prompt.lower() or "kawaii" in prompt.lower():
            return "Kawaii transformed text desu~"
        elif "patriotic" in prompt.lower() or "american" in prompt.lower():
            return "FREEDOM transformed text, America!"
        elif "drunk" in prompt.lower() or "slur" in prompt.lower():
            return "Shransformed texsht, ya know..."
        else:
            return "Transformed text (mock)"
    
    def create_character_prompt(
        self,
        original_text: str,
        character_prompt: str,
        intensity: float = 1.0
    ) -> str:
        """Mock prompt creation."""
        return f"Mock prompt for: {original_text}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "model_path": str(self.model_path),
            "model_loaded": self._model_loaded,
            "mock": True
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_model()


def create_llm_processor(model_path: str, processor_type: str = "auto", **kwargs):
    """
    Factory function to create LLM processor.
    
    Args:
        model_path: Path to model file (for llama.cpp) or model name (for ollama)
        processor_type: Type of processor ("auto", "llama_cpp", "ollama", "mock")
        **kwargs: Additional arguments for processor
        
    Returns:
        Appropriate LLM processor instance
    """
    if processor_type == "mock":
        return MockLLMProcessor(model_path, **kwargs)
    elif processor_type == "llama_cpp":
        if LLAMA_CPP_AVAILABLE:
            return LLMProcessor(model_path, **kwargs)
        else:
            logger.warning("llama.cpp not available, falling back to mock processor")
            return MockLLMProcessor(model_path, **kwargs)
    elif processor_type == "ollama":
        try:
            from .ollama_processor import create_ollama_processor
            return create_ollama_processor(model_path, **kwargs)
        except ImportError:
            logger.warning("Ollama not available, falling back to mock processor")
            return MockLLMProcessor(model_path, **kwargs)
    else:  # processor_type == "auto"
        # Try Ollama first (easier to use), then llama.cpp, then mock
        try:
            from .ollama_processor import create_ollama_processor
            return create_ollama_processor(model_path, **kwargs)
        except ImportError:
            if LLAMA_CPP_AVAILABLE:
                return LLMProcessor(model_path, **kwargs)
            else:
                logger.info("No LLM processors available, using mock processor")
                return MockLLMProcessor(model_path, **kwargs)