"""
Character transformation module for voice transformation system.

This module provides character profile management and text transformation
capabilities for converting user speech into character-specific speech patterns.
"""

from .profile import CharacterProfile, CharacterProfileManager
from .llm_processor import LLMProcessor, MockLLMProcessor, create_llm_processor
from .transformer import CharacterTransformer, TransformationResult, TransformationCache

# Try to import Ollama processor
try:
    from .ollama_processor import OllamaProcessor, create_ollama_processor
    _ollama_available = True
except ImportError:
    _ollama_available = False

__all__ = [
    'CharacterProfile',
    'CharacterProfileManager',
    'LLMProcessor',
    'MockLLMProcessor',
    'create_llm_processor',
    'CharacterTransformer',
    'TransformationResult',
    'TransformationCache',
]

if _ollama_available:
    __all__.extend(['OllamaProcessor', 'create_ollama_processor'])