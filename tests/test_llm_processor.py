"""
Unit tests for LLM processor.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.character.llm_processor import (
    LLMProcessor, 
    MockLLMProcessor, 
    create_llm_processor,
    LLAMA_CPP_AVAILABLE
)


class TestMockLLMProcessor:
    """Test cases for MockLLMProcessor."""
    
    def test_mock_llm_processor_initialization(self):
        """Test MockLLMProcessor initialization."""
        processor = MockLLMProcessor("test_model.gguf")
        
        assert processor.model_path == Path("test_model.gguf")
        assert not processor._model_loaded
    
    def test_mock_load_unload_model(self):
        """Test mock model loading and unloading."""
        processor = MockLLMProcessor("test_model.gguf")
        
        # Initially not loaded
        assert not processor._model_loaded
        
        # Load model
        processor.load_model()
        assert processor._model_loaded
        
        # Unload model
        processor.unload_model()
        assert not processor._model_loaded
    
    @pytest.mark.asyncio
    async def test_mock_generate_async_anime(self):
        """Test mock text generation for anime character."""
        processor = MockLLMProcessor("test_model.gguf")
        processor.load_model()
        
        result = await processor.generate_async("Transform this anime text")
        assert "Kawaii" in result
        assert "desu~" in result
    
    @pytest.mark.asyncio
    async def test_mock_generate_async_patriotic(self):
        """Test mock text generation for patriotic character."""
        processor = MockLLMProcessor("test_model.gguf")
        processor.load_model()
        
        result = await processor.generate_async("Transform this patriotic American text")
        assert "FREEDOM" in result
        assert "America" in result
    
    @pytest.mark.asyncio
    async def test_mock_generate_async_drunk(self):
        """Test mock text generation for drunk character."""
        processor = MockLLMProcessor("test_model.gguf")
        processor.load_model()
        
        result = await processor.generate_async("Transform this drunk text")
        assert "Shransformed" in result
        assert "ya know" in result
    
    @pytest.mark.asyncio
    async def test_mock_generate_async_default(self):
        """Test mock text generation for default character."""
        processor = MockLLMProcessor("test_model.gguf")
        processor.load_model()
        
        result = await processor.generate_async("Transform this text")
        assert "Transformed text (mock)" in result
    
    def test_mock_create_character_prompt(self):
        """Test mock character prompt creation."""
        processor = MockLLMProcessor("test_model.gguf")
        
        prompt = processor.create_character_prompt(
            "Hello world",
            "Transform to character style: {text}",
            1.0
        )
        
        assert "Mock prompt for: Hello world" == prompt
    
    def test_mock_get_model_info(self):
        """Test getting mock model information."""
        processor = MockLLMProcessor("test_model.gguf")
        
        info = processor.get_model_info()
        
        assert info["model_path"] == "test_model.gguf"
        assert info["model_loaded"] is False
        assert info["mock"] is True
    
    def test_mock_context_manager(self):
        """Test mock LLM processor as context manager."""
        processor = MockLLMProcessor("test_model.gguf")
        
        with processor as p:
            assert p._model_loaded
        
        assert not processor._model_loaded


@pytest.mark.skipif(not LLAMA_CPP_AVAILABLE, reason="llama-cpp-python not available")
class TestLLMProcessor:
    """Test cases for LLMProcessor (only if llama.cpp is available)."""
    
    @pytest.fixture
    def temp_model_file(self):
        """Create temporary model file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"fake model data")
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    def test_llm_processor_initialization(self, temp_model_file):
        """Test LLMProcessor initialization."""
        processor = LLMProcessor(temp_model_file)
        
        assert processor.model_path == Path(temp_model_file)
        assert processor.n_gpu_layers == -1
        assert processor.n_ctx == 2048
        assert not processor._model_loaded
    
    def test_llm_processor_initialization_with_params(self, temp_model_file):
        """Test LLMProcessor initialization with custom parameters."""
        processor = LLMProcessor(
            temp_model_file,
            n_gpu_layers=10,
            n_ctx=4096,
            n_threads=4,
            verbose=True
        )
        
        assert processor.n_gpu_layers == 10
        assert processor.n_ctx == 4096
        assert processor.n_threads == 4
        assert processor.verbose is True
    
    def test_llm_processor_nonexistent_model(self):
        """Test LLMProcessor with nonexistent model file."""
        processor = LLMProcessor("nonexistent_model.gguf")
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            processor.load_model()
    
    @patch('src.character.llm_processor.Llama')
    def test_llm_processor_load_model_success(self, mock_llama, temp_model_file):
        """Test successful model loading."""
        mock_llama_instance = Mock()
        mock_llama.return_value = mock_llama_instance
        
        processor = LLMProcessor(temp_model_file)
        processor.load_model()
        
        assert processor._model_loaded
        mock_llama.assert_called_once()
    
    @patch('src.character.llm_processor.Llama')
    def test_llm_processor_load_model_failure(self, mock_llama, temp_model_file):
        """Test model loading failure."""
        mock_llama.side_effect = Exception("Model loading failed")
        
        processor = LLMProcessor(temp_model_file)
        
        with pytest.raises(RuntimeError, match="Failed to load LLM model"):
            processor.load_model()
        
        assert not processor._model_loaded
    
    @patch('src.character.llm_processor.Llama')
    def test_llm_processor_load_model_already_loaded(self, mock_llama, temp_model_file):
        """Test loading model when already loaded."""
        mock_llama_instance = Mock()
        mock_llama.return_value = mock_llama_instance
        
        processor = LLMProcessor(temp_model_file)
        processor.load_model()
        processor.load_model()  # Second call should not reload
        
        # Should only be called once
        mock_llama.assert_called_once()
    
    @patch('src.character.llm_processor.Llama')
    def test_llm_processor_unload_model(self, mock_llama, temp_model_file):
        """Test model unloading."""
        mock_llama_instance = Mock()
        mock_llama.return_value = mock_llama_instance
        
        processor = LLMProcessor(temp_model_file)
        processor.load_model()
        assert processor._model_loaded
        
        processor.unload_model()
        assert not processor._model_loaded
        assert processor._llama is None
    
    @patch('src.character.llm_processor.Llama')
    def test_llm_processor_generate_sync_success(self, mock_llama, temp_model_file):
        """Test successful synchronous text generation."""
        mock_llama_instance = Mock()
        mock_llama_instance.return_value = {
            'choices': [{'text': '  Generated text  '}]
        }
        mock_llama.return_value = mock_llama_instance
        
        processor = LLMProcessor(temp_model_file)
        processor.load_model()
        
        result = processor._generate_sync("Test prompt")
        
        assert result == "Generated text"
        mock_llama_instance.assert_called_once()
    
    @patch('src.character.llm_processor.Llama')
    def test_llm_processor_generate_sync_not_loaded(self, mock_llama, temp_model_file):
        """Test synchronous generation without loaded model."""
        processor = LLMProcessor(temp_model_file)
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            processor._generate_sync("Test prompt")
    
    @patch('src.character.llm_processor.Llama')
    def test_llm_processor_generate_sync_failure(self, mock_llama, temp_model_file):
        """Test synchronous generation failure."""
        mock_llama_instance = Mock()
        mock_llama_instance.side_effect = Exception("Generation failed")
        mock_llama.return_value = mock_llama_instance
        
        processor = LLMProcessor(temp_model_file)
        processor.load_model()
        
        with pytest.raises(RuntimeError, match="Text generation failed"):
            processor._generate_sync("Test prompt")
    
    @patch('src.character.llm_processor.Llama')
    @pytest.mark.asyncio
    async def test_llm_processor_generate_async(self, mock_llama, temp_model_file):
        """Test asynchronous text generation."""
        mock_llama_instance = Mock()
        mock_llama_instance.return_value = {
            'choices': [{'text': 'Generated async text'}]
        }
        mock_llama.return_value = mock_llama_instance
        
        processor = LLMProcessor(temp_model_file)
        processor.load_model()
        
        result = await processor.generate_async("Test prompt")
        
        assert result == "Generated async text"
    
    def test_create_character_prompt_basic(self, temp_model_file):
        """Test basic character prompt creation."""
        processor = LLMProcessor(temp_model_file)
        
        prompt = processor.create_character_prompt(
            "Hello world",
            "Transform this text: {text}",
            1.0
        )
        
        assert "Hello world" in prompt
        assert "Transform this text" in prompt
        assert "moderately" in prompt  # intensity 1.0 maps to "moderately"
    
    def test_create_character_prompt_intensity_mapping(self, temp_model_file):
        """Test character prompt creation with different intensities."""
        processor = LLMProcessor(temp_model_file)
        
        # Test different intensity levels
        test_cases = [
            (0.0, "very subtly"),
            (0.5, "slightly"),
            (1.0, "moderately"),
            (1.5, "strongly"),
            (2.0, "very strongly"),
            (0.3, "very subtly"),  # Should map to closest (0.0)
            (1.7, "strongly"),     # Should map to closest (1.5)
        ]
        
        for intensity, expected_desc in test_cases:
            prompt = processor.create_character_prompt(
                "Test text",
                "Transform {intensity}: {text}",
                intensity
            )
            assert expected_desc in prompt
    
    def test_get_model_info(self, temp_model_file):
        """Test getting model information."""
        processor = LLMProcessor(temp_model_file, n_gpu_layers=5, n_ctx=1024)
        
        info = processor.get_model_info()
        
        assert info["model_path"] == temp_model_file
        assert info["model_loaded"] is False
        assert info["n_gpu_layers"] == 5
        assert info["n_ctx"] == 1024
        assert info["model_exists"] is True
    
    @patch('src.character.llm_processor.Llama')
    def test_llm_processor_context_manager(self, mock_llama, temp_model_file):
        """Test LLMProcessor as context manager."""
        mock_llama_instance = Mock()
        mock_llama.return_value = mock_llama_instance
        
        processor = LLMProcessor(temp_model_file)
        
        with processor as p:
            assert p._model_loaded
        
        assert not processor._model_loaded


class TestLLMProcessorFactory:
    """Test cases for LLM processor factory function."""
    
    def test_create_llm_processor_with_llama_cpp(self):
        """Test factory function when llama.cpp is available."""
        if LLAMA_CPP_AVAILABLE:
            processor = create_llm_processor("test_model.gguf")
            assert isinstance(processor, LLMProcessor)
        else:
            processor = create_llm_processor("test_model.gguf")
            assert isinstance(processor, MockLLMProcessor)
    
    def test_create_llm_processor_without_llama_cpp(self):
        """Test factory function when llama.cpp is not available."""
        with patch('src.character.llm_processor.LLAMA_CPP_AVAILABLE', False):
            processor = create_llm_processor("test_model.gguf")
            assert isinstance(processor, MockLLMProcessor)


class TestLLMProcessorImportError:
    """Test cases for import error handling."""
    
    def test_llm_processor_import_error(self):
        """Test LLMProcessor initialization when llama.cpp is not available."""
        with patch('src.character.llm_processor.LLAMA_CPP_AVAILABLE', False):
            with pytest.raises(ImportError, match="llama-cpp-python is not installed"):
                LLMProcessor("test_model.gguf")