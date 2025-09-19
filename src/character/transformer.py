"""
Character text transformation system.

This module provides the CharacterTransformer class that combines character profiles
and LLM processing to transform text according to character-specific styles and patterns.
"""

import logging
import asyncio
import hashlib
import time
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from functools import lru_cache

from .profile import CharacterProfile, CharacterProfileManager
from .llm_processor import LLMProcessor, create_llm_processor

logger = logging.getLogger(__name__)


@dataclass
class TransformationResult:
    """
    Result of text transformation.
    
    Attributes:
        original_text: Original input text
        transformed_text: Character-transformed text
        character_name: Name of character used for transformation
        intensity: Transformation intensity used
        processing_time_ms: Time taken for transformation in milliseconds
        cached: Whether result was retrieved from cache
    """
    original_text: str
    transformed_text: str
    character_name: str
    intensity: float
    processing_time_ms: float
    cached: bool = False


class TransformationCache:
    """
    Cache for character text transformations to improve performance.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize transformation cache.
        
        Args:
            max_size: Maximum number of cached transformations
            ttl_seconds: Time-to-live for cached entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[str, float]] = {}  # key -> (result, timestamp)
        self._access_order: Dict[str, float] = {}  # key -> last_access_time
    
    def _generate_key(
        self, 
        text: str, 
        character_name: str, 
        intensity: float
    ) -> str:
        """
        Generate cache key for transformation parameters.
        
        Args:
            text: Input text
            character_name: Character name
            intensity: Transformation intensity
            
        Returns:
            Cache key string
        """
        # Create a hash of the parameters for consistent key generation
        key_data = f"{text}|{character_name}|{intensity:.2f}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()
    
    def get(
        self, 
        text: str, 
        character_name: str, 
        intensity: float
    ) -> Optional[str]:
        """
        Get cached transformation result.
        
        Args:
            text: Input text
            character_name: Character name
            intensity: Transformation intensity
            
        Returns:
            Cached transformed text or None if not found/expired
        """
        key = self._generate_key(text, character_name, intensity)
        current_time = time.time()
        
        if key in self._cache:
            result, timestamp = self._cache[key]
            
            # Check if entry has expired
            if current_time - timestamp > self.ttl_seconds:
                self._remove_key(key)
                return None
            
            # Update access time
            self._access_order[key] = current_time
            logger.debug(f"Cache hit for key: {key[:8]}...")
            return result
        
        return None
    
    def put(
        self, 
        text: str, 
        character_name: str, 
        intensity: float, 
        result: str
    ) -> None:
        """
        Store transformation result in cache.
        
        Args:
            text: Input text
            character_name: Character name
            intensity: Transformation intensity
            result: Transformed text result
        """
        key = self._generate_key(text, character_name, intensity)
        current_time = time.time()
        
        # Remove oldest entries if cache is full
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        self._cache[key] = (result, current_time)
        self._access_order[key] = current_time
        
        logger.debug(f"Cached result for key: {key[:8]}...")
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache and access order."""
        self._cache.pop(key, None)
        self._access_order.pop(key, None)
    
    def _evict_oldest(self) -> None:
        """Evict the least recently used cache entry."""
        if not self._access_order:
            return
        
        # Find the key with the oldest (smallest) access time
        oldest_key = min(self._access_order.keys(), 
                        key=lambda k: self._access_order[k])
        self._remove_key(oldest_key)
        logger.debug(f"Evicted oldest cache entry: {oldest_key[:8]}...")
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Cleared transformation cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        current_time = time.time()
        expired_count = sum(
            1 for _, timestamp in self._cache.values()
            if current_time - timestamp > self.ttl_seconds
        )
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "expired_entries": expired_count,
            "ttl_seconds": self.ttl_seconds
        }


class CharacterTransformer:
    """
    Main character text transformation system.
    
    Combines character profiles and LLM processing to transform text according
    to character-specific styles, patterns, and personalities.
    """
    
    def __init__(
        self,
        llm_model_path: str,
        profiles_directory: str = "characters",
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        **llm_kwargs
    ):
        """
        Initialize character transformer.
        
        Args:
            llm_model_path: Path to LLM model file
            profiles_directory: Directory containing character profiles
            cache_size: Maximum number of cached transformations
            cache_ttl: Cache time-to-live in seconds
            **llm_kwargs: Additional arguments for LLM processor
        """
        self.llm_model_path = llm_model_path
        self.llm_kwargs = llm_kwargs
        
        # Initialize components
        self.profile_manager = CharacterProfileManager(profiles_directory)
        self.llm_processor: Optional[LLMProcessor] = None
        self.cache = TransformationCache(cache_size, cache_ttl)
        
        # Current state
        self.current_character: Optional[CharacterProfile] = None
        self.current_intensity: float = 1.0
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize the transformer by loading the LLM model.
        
        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            logger.info("Transformer already initialized")
            return
        
        try:
            logger.info("Initializing character transformer...")
            
            # Create and load LLM processor if not already set
            if self.llm_processor is None:
                self.llm_processor = create_llm_processor(
                    self.llm_model_path, 
                    **self.llm_kwargs
                )
            
            self.llm_processor.load_model()
            
            # Load default character
            self.current_character = self.profile_manager.get_default_profile()
            
            self._initialized = True
            logger.info("Character transformer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize character transformer: {e}")
            raise RuntimeError(f"Failed to initialize character transformer: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the transformer and cleanup resources."""
        if self.llm_processor:
            self.llm_processor.unload_model()
            self.llm_processor = None
        
        self.cache.clear()
        self._initialized = False
        logger.info("Character transformer shutdown complete")
    
    def set_character(self, character_name: str) -> None:
        """
        Set the current character for transformations.
        
        Args:
            character_name: Name of character profile to use
            
        Raises:
            FileNotFoundError: If character profile doesn't exist
            ValueError: If character profile is invalid
        """
        try:
            character = self.profile_manager.load_profile(character_name)
            self.current_character = character
            logger.info(f"Set current character to: {character_name}")
        except Exception as e:
            logger.error(f"Failed to set character {character_name}: {e}")
            raise
    
    def set_intensity(self, intensity: float) -> None:
        """
        Set transformation intensity.
        
        Args:
            intensity: Transformation intensity (0.0-2.0)
            
        Raises:
            ValueError: If intensity is out of range
        """
        if not (0.0 <= intensity <= 2.0):
            raise ValueError("Intensity must be between 0.0 and 2.0")
        
        self.current_intensity = intensity
        logger.debug(f"Set transformation intensity to: {intensity}")
    
    def get_available_characters(self) -> list[str]:
        """
        Get list of available character names.
        
        Returns:
            List of character profile names
        """
        return self.profile_manager.get_available_profiles()
    
    def get_current_character(self) -> Optional[str]:
        """
        Get current character name.
        
        Returns:
            Current character name or None if not set
        """
        return self.current_character.name if self.current_character else None
    
    def get_current_intensity(self) -> float:
        """
        Get current transformation intensity.
        
        Returns:
            Current intensity value
        """
        return self.current_intensity
    
    async def transform_text(
        self, 
        text: str, 
        character_name: Optional[str] = None,
        intensity: Optional[float] = None
    ) -> TransformationResult:
        """
        Transform text using character profile and LLM.
        
        Args:
            text: Original text to transform
            character_name: Character to use (uses current if None)
            intensity: Transformation intensity (uses current if None)
            
        Returns:
            TransformationResult with original and transformed text
            
        Raises:
            RuntimeError: If transformer is not initialized
            ValueError: If parameters are invalid
        """
        if not self._initialized:
            raise RuntimeError("Transformer not initialized. Call initialize() first.")
        
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        # Use provided parameters or current settings
        char_name = character_name or (
            self.current_character.name if self.current_character else "default"
        )
        intensity_val = intensity if intensity is not None else self.current_intensity
        
        # Validate intensity
        if not (0.0 <= intensity_val <= 2.0):
            raise ValueError("Intensity must be between 0.0 and 2.0")
        
        start_time = time.time()
        
        # Check cache first
        cached_result = self.cache.get(text, char_name, intensity_val)
        if cached_result:
            processing_time = (time.time() - start_time) * 1000
            return TransformationResult(
                original_text=text,
                transformed_text=cached_result,
                character_name=char_name,
                intensity=intensity_val,
                processing_time_ms=processing_time,
                cached=True
            )
        
        try:
            # Load character profile if different from current
            if not self.current_character or self.current_character.name != char_name:
                character = self.profile_manager.load_profile(char_name)
            else:
                character = self.current_character
            
            # Apply intensity adjustment to character
            adjusted_intensity = intensity_val * character.intensity_multiplier
            adjusted_intensity = min(2.0, max(0.0, adjusted_intensity))  # Clamp to valid range
            
            # Handle special case for zero intensity (no transformation)
            if adjusted_intensity == 0.0:
                transformed_text = text
            else:
                # Create prompt for LLM
                prompt = self.llm_processor.create_character_prompt(
                    text, 
                    character.transformation_prompt, 
                    adjusted_intensity
                )
                
                # Generate transformed text
                transformed_text = await self.llm_processor.generate_async(
                    prompt,
                    max_tokens=min(256, len(text) * 2),  # Adaptive token limit
                    temperature=0.7,
                    top_p=0.9,
                    stop=["Human:", "Assistant:", "\n\n"]
                )
                
                # Apply simple speech pattern replacements as post-processing
                transformed_text = self._apply_speech_patterns(
                    transformed_text, 
                    character.speech_patterns,
                    adjusted_intensity
                )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Cache the result
            self.cache.put(text, char_name, intensity_val, transformed_text)
            
            logger.debug(
                f"Transformed text for {char_name} "
                f"(intensity: {intensity_val:.2f}) in {processing_time:.1f}ms"
            )
            
            return TransformationResult(
                original_text=text,
                transformed_text=transformed_text,
                character_name=char_name,
                intensity=intensity_val,
                processing_time_ms=processing_time,
                cached=False
            )
            
        except Exception as e:
            logger.error(f"Text transformation failed: {e}")
            # Return original text as fallback
            processing_time = (time.time() - start_time) * 1000
            return TransformationResult(
                original_text=text,
                transformed_text=text,  # Fallback to original
                character_name=char_name,
                intensity=intensity_val,
                processing_time_ms=processing_time,
                cached=False
            )
    
    def _apply_speech_patterns(
        self, 
        text: str, 
        patterns: Dict[str, str], 
        intensity: float
    ) -> str:
        """
        Apply character-specific speech patterns to text.
        
        Args:
            text: Text to modify
            patterns: Dictionary of pattern replacements
            intensity: Intensity for pattern application
            
        Returns:
            Text with patterns applied
        """
        if not patterns or intensity == 0.0:
            return text
        
        modified_text = text
        
        # Apply patterns with intensity-based probability
        for pattern, replacement in patterns.items():
            if intensity >= 1.0:
                # Full intensity - apply all patterns
                modified_text = modified_text.replace(pattern, replacement)
            else:
                # Partial intensity - apply patterns probabilistically
                # This is a simple implementation; could be more sophisticated
                import random
                if random.random() < intensity:
                    modified_text = modified_text.replace(pattern, replacement)
        
        return modified_text
    
    def get_transformation_stats(self) -> Dict[str, Any]:
        """
        Get transformation system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        stats = {
            "initialized": self._initialized,
            "current_character": self.get_current_character(),
            "current_intensity": self.current_intensity,
            "available_characters": len(self.get_available_characters()),
            "cache_stats": self.cache.get_stats()
        }
        
        if self.llm_processor:
            stats["llm_info"] = self.llm_processor.get_model_info()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the transformation cache."""
        self.cache.clear()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()