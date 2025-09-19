"""
Unit tests for character transformer system.
"""

import pytest
import tempfile
import asyncio
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.character.transformer import (
    CharacterTransformer,
    TransformationResult,
    TransformationCache
)
from src.character.profile import CharacterProfile


class TestTransformationCache:
    """Test cases for TransformationCache."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = TransformationCache(max_size=100, ttl_seconds=300)
        
        assert cache.max_size == 100
        assert cache.ttl_seconds == 300
        assert len(cache._cache) == 0
        assert len(cache._access_order) == 0
    
    def test_cache_put_and_get(self):
        """Test basic cache put and get operations."""
        cache = TransformationCache()
        
        # Put item in cache
        cache.put("hello", "anime", 1.0, "hello desu~")
        
        # Get item from cache
        result = cache.get("hello", "anime", 1.0)
        assert result == "hello desu~"
    
    def test_cache_miss(self):
        """Test cache miss for non-existent item."""
        cache = TransformationCache()
        
        result = cache.get("hello", "anime", 1.0)
        assert result is None
    
    def test_cache_key_generation(self):
        """Test that different parameters generate different keys."""
        cache = TransformationCache()
        
        cache.put("hello", "anime", 1.0, "result1")
        cache.put("hello", "anime", 1.5, "result2")
        cache.put("hello", "patriotic", 1.0, "result3")
        cache.put("goodbye", "anime", 1.0, "result4")
        
        assert cache.get("hello", "anime", 1.0) == "result1"
        assert cache.get("hello", "anime", 1.5) == "result2"
        assert cache.get("hello", "patriotic", 1.0) == "result3"
        assert cache.get("goodbye", "anime", 1.0) == "result4"
    
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        cache = TransformationCache(ttl_seconds=1)  # 1 second TTL
        
        cache.put("hello", "anime", 1.0, "hello desu~")
        
        # Should be available immediately
        result = cache.get("hello", "anime", 1.0)
        assert result == "hello desu~"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired now
        result = cache.get("hello", "anime", 1.0)
        assert result is None
    
    def test_cache_eviction(self):
        """Test cache eviction when max size is reached."""
        cache = TransformationCache(max_size=2)
        
        # Fill cache to capacity
        cache.put("text1", "char1", 1.0, "result1")
        time.sleep(0.01)  # Small delay to ensure different timestamps
        cache.put("text2", "char2", 1.0, "result2")
        
        # Access first item to make it more recent
        time.sleep(0.01)
        cache.get("text1", "char1", 1.0)
        
        # Add third item (should evict text2)
        time.sleep(0.01)
        cache.put("text3", "char3", 1.0, "result3")
        
        # text1 should still be there (more recently accessed)
        assert cache.get("text1", "char1", 1.0) == "result1"
        
        # text2 should be evicted
        assert cache.get("text2", "char2", 1.0) is None
        
        # text3 should be there
        assert cache.get("text3", "char3", 1.0) == "result3"
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = TransformationCache()
        
        cache.put("hello", "anime", 1.0, "hello desu~")
        assert cache.get("hello", "anime", 1.0) == "hello desu~"
        
        cache.clear()
        assert cache.get("hello", "anime", 1.0) is None
        assert len(cache._cache) == 0
        assert len(cache._access_order) == 0
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = TransformationCache(max_size=10, ttl_seconds=300)
        
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 10
        assert stats["ttl_seconds"] == 300
        assert stats["expired_entries"] == 0
        
        # Add some entries
        cache.put("text1", "char1", 1.0, "result1")
        cache.put("text2", "char2", 1.0, "result2")
        
        stats = cache.get_stats()
        assert stats["size"] == 2


class TestTransformationResult:
    """Test cases for TransformationResult dataclass."""
    
    def test_transformation_result_creation(self):
        """Test creating TransformationResult."""
        result = TransformationResult(
            original_text="Hello world",
            transformed_text="Hello world desu~",
            character_name="anime",
            intensity=1.0,
            processing_time_ms=150.5,
            cached=False
        )
        
        assert result.original_text == "Hello world"
        assert result.transformed_text == "Hello world desu~"
        assert result.character_name == "anime"
        assert result.intensity == 1.0
        assert result.processing_time_ms == 150.5
        assert result.cached is False
    
    def test_transformation_result_defaults(self):
        """Test TransformationResult with default values."""
        result = TransformationResult(
            original_text="Hello",
            transformed_text="Hello~",
            character_name="test",
            intensity=1.0,
            processing_time_ms=100.0
        )
        
        assert result.cached is False  # Default value


class TestCharacterTransformer:
    """Test cases for CharacterTransformer."""
    
    @pytest.fixture
    def temp_profiles_dir(self):
        """Create temporary directory for test profiles."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test character profile
            profile_data = {
                "name": "test_character",
                "description": "Test character",
                "personality_traits": ["friendly"],
                "speech_patterns": {"hello": "hi there"},
                "vocabulary_preferences": {},
                "transformation_prompt": "Transform to test style: {text}",
                "voice_model_path": "",
                "intensity_multiplier": 1.0
            }
            
            import json
            profile_path = Path(temp_dir) / "test_character.json"
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f)
            
            yield temp_dir
    
    @pytest.fixture
    def mock_llm_processor(self):
        """Create mock LLM processor."""
        mock_processor = Mock()
        mock_processor.load_model = Mock()
        mock_processor.unload_model = Mock()
        mock_processor.create_character_prompt = Mock(return_value="mock prompt")
        mock_processor.generate_async = AsyncMock(return_value="transformed text")
        mock_processor.get_model_info = Mock(return_value={"mock": True})
        return mock_processor
    
    def test_transformer_initialization(self, temp_profiles_dir):
        """Test CharacterTransformer initialization."""
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        assert transformer.llm_model_path == "test_model.gguf"
        assert transformer.current_character is None
        assert transformer.current_intensity == 1.0
        assert not transformer._initialized
    
    @patch('src.character.transformer.create_llm_processor')
    @pytest.mark.asyncio
    async def test_transformer_initialize(self, mock_create_llm, temp_profiles_dir, mock_llm_processor):
        """Test transformer initialization."""
        mock_create_llm.return_value = mock_llm_processor
        
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        await transformer.initialize()
        
        assert transformer._initialized
        assert transformer.current_character is not None
        assert transformer.current_character.name == "default"
        mock_llm_processor.load_model.assert_called_once()
    
    @patch('src.character.transformer.create_llm_processor')
    @pytest.mark.asyncio
    async def test_transformer_initialize_failure(self, mock_create_llm, temp_profiles_dir):
        """Test transformer initialization failure."""
        mock_create_llm.side_effect = Exception("Model loading failed")
        
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        with pytest.raises(RuntimeError, match="Failed to initialize character transformer"):
            await transformer.initialize()
        
        assert not transformer._initialized
    
    @patch('src.character.transformer.create_llm_processor')
    @pytest.mark.asyncio
    async def test_transformer_shutdown(self, mock_create_llm, temp_profiles_dir, mock_llm_processor):
        """Test transformer shutdown."""
        mock_create_llm.return_value = mock_llm_processor
        
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        await transformer.initialize()
        await transformer.shutdown()
        
        assert not transformer._initialized
        mock_llm_processor.unload_model.assert_called_once()
    
    @patch('src.character.transformer.create_llm_processor')
    @pytest.mark.asyncio
    async def test_set_character(self, mock_create_llm, temp_profiles_dir, mock_llm_processor):
        """Test setting character."""
        mock_create_llm.return_value = mock_llm_processor
        
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        await transformer.initialize()
        
        # Set to test character
        transformer.set_character("test_character")
        assert transformer.current_character.name == "test_character"
        assert transformer.get_current_character() == "test_character"
    
    def test_set_character_nonexistent(self, temp_profiles_dir):
        """Test setting nonexistent character."""
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        with pytest.raises(FileNotFoundError):
            transformer.set_character("nonexistent")
    
    def test_set_intensity(self, temp_profiles_dir):
        """Test setting transformation intensity."""
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        transformer.set_intensity(1.5)
        assert transformer.current_intensity == 1.5
        assert transformer.get_current_intensity() == 1.5
    
    def test_set_intensity_invalid(self, temp_profiles_dir):
        """Test setting invalid intensity."""
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        with pytest.raises(ValueError, match="Intensity must be between 0.0 and 2.0"):
            transformer.set_intensity(-0.1)
        
        with pytest.raises(ValueError, match="Intensity must be between 0.0 and 2.0"):
            transformer.set_intensity(2.1)
    
    @patch('src.character.transformer.create_llm_processor')
    @pytest.mark.asyncio
    async def test_transform_text_basic(self, mock_create_llm, temp_profiles_dir, mock_llm_processor):
        """Test basic text transformation."""
        mock_create_llm.return_value = mock_llm_processor
        
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        await transformer.initialize()
        
        result = await transformer.transform_text("Hello world")
        
        assert isinstance(result, TransformationResult)
        assert result.original_text == "Hello world"
        assert result.transformed_text == "transformed text"
        assert result.character_name == "default"
        assert result.intensity == 1.0
        assert result.processing_time_ms >= 0
        assert not result.cached
    
    @patch('src.character.transformer.create_llm_processor')
    @pytest.mark.asyncio
    async def test_transform_text_not_initialized(self, mock_create_llm, temp_profiles_dir):
        """Test text transformation without initialization."""
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        with pytest.raises(RuntimeError, match="Transformer not initialized"):
            await transformer.transform_text("Hello world")
    
    @patch('src.character.transformer.create_llm_processor')
    @pytest.mark.asyncio
    async def test_transform_text_empty(self, mock_create_llm, temp_profiles_dir, mock_llm_processor):
        """Test text transformation with empty text."""
        mock_create_llm.return_value = mock_llm_processor
        
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        await transformer.initialize()
        
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            await transformer.transform_text("")
        
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            await transformer.transform_text("   ")
    
    @patch('src.character.transformer.create_llm_processor')
    @pytest.mark.asyncio
    async def test_transform_text_zero_intensity(self, mock_create_llm, temp_profiles_dir, mock_llm_processor):
        """Test text transformation with zero intensity."""
        mock_create_llm.return_value = mock_llm_processor
        
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        await transformer.initialize()
        transformer.set_intensity(0.0)
        
        result = await transformer.transform_text("Hello world")
        
        # With zero intensity, text should be unchanged
        assert result.transformed_text == "Hello world"
        assert result.intensity == 0.0
    
    @patch('src.character.transformer.create_llm_processor')
    @pytest.mark.asyncio
    async def test_transform_text_with_parameters(self, mock_create_llm, temp_profiles_dir, mock_llm_processor):
        """Test text transformation with specific parameters."""
        mock_create_llm.return_value = mock_llm_processor
        
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        await transformer.initialize()
        
        result = await transformer.transform_text(
            "Hello world",
            character_name="test_character",
            intensity=1.5
        )
        
        assert result.character_name == "test_character"
        assert result.intensity == 1.5
    
    @patch('src.character.transformer.create_llm_processor')
    @pytest.mark.asyncio
    async def test_transform_text_caching(self, mock_create_llm, temp_profiles_dir, mock_llm_processor):
        """Test text transformation caching."""
        mock_create_llm.return_value = mock_llm_processor
        
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        await transformer.initialize()
        
        # First transformation
        result1 = await transformer.transform_text("Hello world")
        assert not result1.cached
        
        # Second transformation (should be cached)
        result2 = await transformer.transform_text("Hello world")
        assert result2.cached
        assert result2.transformed_text == result1.transformed_text
        
        # LLM should only be called once
        assert mock_llm_processor.generate_async.call_count == 1
    
    @patch('src.character.transformer.create_llm_processor')
    @pytest.mark.asyncio
    async def test_transform_text_llm_failure(self, mock_create_llm, temp_profiles_dir, mock_llm_processor):
        """Test text transformation with LLM failure."""
        mock_llm_processor.generate_async.side_effect = Exception("LLM failed")
        mock_create_llm.return_value = mock_llm_processor
        
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        await transformer.initialize()
        
        # Should fallback to original text on failure
        result = await transformer.transform_text("Hello world")
        assert result.transformed_text == "Hello world"  # Fallback
    
    def test_apply_speech_patterns(self, temp_profiles_dir):
        """Test speech pattern application."""
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        patterns = {"hello": "hi there", "world": "universe"}
        
        # Full intensity
        result = transformer._apply_speech_patterns("hello world", patterns, 1.0)
        assert result == "hi there universe"
        
        # Zero intensity
        result = transformer._apply_speech_patterns("hello world", patterns, 0.0)
        assert result == "hello world"
    
    @patch('src.character.transformer.create_llm_processor')
    @pytest.mark.asyncio
    async def test_get_available_characters(self, mock_create_llm, temp_profiles_dir, mock_llm_processor):
        """Test getting available characters."""
        mock_create_llm.return_value = mock_llm_processor
        
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        characters = transformer.get_available_characters()
        assert "default" in characters
        assert "test_character" in characters
    
    @patch('src.character.transformer.create_llm_processor')
    @pytest.mark.asyncio
    async def test_get_transformation_stats(self, mock_create_llm, temp_profiles_dir, mock_llm_processor):
        """Test getting transformation statistics."""
        mock_create_llm.return_value = mock_llm_processor
        
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        # Before initialization
        stats = transformer.get_transformation_stats()
        assert not stats["initialized"]
        assert stats["current_character"] is None
        
        # After initialization
        await transformer.initialize()
        stats = transformer.get_transformation_stats()
        assert stats["initialized"]
        assert stats["current_character"] == "default"
        assert stats["current_intensity"] == 1.0
        assert "cache_stats" in stats
        assert "llm_info" in stats
    
    @patch('src.character.transformer.create_llm_processor')
    @pytest.mark.asyncio
    async def test_clear_cache(self, mock_create_llm, temp_profiles_dir, mock_llm_processor):
        """Test clearing transformation cache."""
        mock_create_llm.return_value = mock_llm_processor
        
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        await transformer.initialize()
        
        # Transform text to populate cache
        await transformer.transform_text("Hello world")
        
        # Clear cache
        transformer.clear_cache()
        
        # Next transformation should not be cached
        result = await transformer.transform_text("Hello world")
        assert not result.cached
    
    @patch('src.character.transformer.create_llm_processor')
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_create_llm, temp_profiles_dir, mock_llm_processor):
        """Test CharacterTransformer as async context manager."""
        mock_create_llm.return_value = mock_llm_processor
        
        transformer = CharacterTransformer(
            llm_model_path="test_model.gguf",
            profiles_directory=temp_profiles_dir
        )
        
        async with transformer as t:
            assert t._initialized
            result = await t.transform_text("Hello world")
            assert isinstance(result, TransformationResult)
        
        assert not transformer._initialized