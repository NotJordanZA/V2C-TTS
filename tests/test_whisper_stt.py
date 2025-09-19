"""
Unit tests for WhisperSTT implementation.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import torch

from src.stt.whisper_stt import WhisperSTT
from src.core.interfaces import PipelineConfig, PipelineError, PipelineStage


@pytest.fixture
def config():
    """Create a test configuration."""
    return PipelineConfig(
        audio_device_id=0,
        sample_rate=16000,
        stt_model_size="base",
        gpu_device="cuda"
    )


@pytest.fixture
def cpu_config():
    """Create a CPU-only test configuration."""
    return PipelineConfig(
        audio_device_id=0,
        sample_rate=16000,
        stt_model_size="base",
        gpu_device="cpu"
    )


@pytest.fixture
def sample_audio():
    """Create sample audio data for testing."""
    # Generate 1 second of sine wave at 440Hz
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio


class TestWhisperSTT:
    """Test cases for WhisperSTT class."""
    
    def test_init_valid_model_size(self, config):
        """Test initialization with valid model size."""
        stt = WhisperSTT(config)
        assert stt.model_size == "base"
        assert stt.device == "cuda"
        assert stt.compute_type == "float16"
        assert not stt.is_initialized
    
    def test_init_invalid_model_size(self):
        """Test initialization with invalid model size defaults to base."""
        config = PipelineConfig(
            audio_device_id=0,
            stt_model_size="invalid_model"
        )
        stt = WhisperSTT(config)
        assert stt.model_size == "base"
    
    def test_init_cpu_device(self, cpu_config):
        """Test initialization with CPU device."""
        stt = WhisperSTT(cpu_config)
        assert stt.device == "cpu"
        assert stt.compute_type == "int8"
    
    @patch('src.stt.whisper_stt.torch.cuda.is_available')
    @patch('src.stt.whisper_stt.WhisperModel')
    async def test_initialize_success(self, mock_whisper_model, mock_cuda_available, config):
        """Test successful initialization."""
        mock_cuda_available.return_value = True
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        stt = WhisperSTT(config)
        await stt.initialize()
        
        assert stt.is_initialized
        assert stt.model == mock_model
        mock_whisper_model.assert_called_once_with(
            "base",
            device="cuda",
            compute_type="float16",
            cpu_threads=0,
            num_workers=1
        )
    
    @patch('src.stt.whisper_stt.torch.cuda.is_available')
    @patch('src.stt.whisper_stt.WhisperModel')
    async def test_initialize_cuda_fallback(self, mock_whisper_model, mock_cuda_available, config):
        """Test fallback to CPU when CUDA is not available."""
        mock_cuda_available.return_value = False
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        stt = WhisperSTT(config)
        await stt.initialize()
        
        assert stt.device == "cpu"
        assert stt.compute_type == "int8"
        mock_whisper_model.assert_called_once_with(
            "base",
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            num_workers=1
        )
    
    @patch('src.stt.whisper_stt.WhisperModel')
    async def test_initialize_failure(self, mock_whisper_model, config):
        """Test initialization failure handling."""
        mock_whisper_model.side_effect = Exception("Model loading failed")
        
        stt = WhisperSTT(config)
        
        with pytest.raises(PipelineError) as exc_info:
            await stt.initialize()
        
        assert exc_info.value.stage == PipelineStage.SPEECH_TO_TEXT
        assert "Failed to initialize Whisper model" in str(exc_info.value)
        assert not exc_info.value.recoverable
    
    @patch('src.stt.whisper_stt.torch.cuda.empty_cache')
    async def test_unload_model(self, mock_empty_cache, config):
        """Test model unloading."""
        stt = WhisperSTT(config)
        stt.model = Mock()
        
        await stt.unload_model()
        
        assert stt.model is None
        mock_empty_cache.assert_called_once()
    
    def test_preprocess_audio_float32(self, config, sample_audio):
        """Test audio preprocessing with float32 input."""
        stt = WhisperSTT(config)
        processed = stt._preprocess_audio(sample_audio)
        
        assert processed.dtype == np.float32
        assert len(processed.shape) == 1  # Mono
        assert np.max(np.abs(processed)) <= 1.0
    
    def test_preprocess_audio_int16(self, config):
        """Test audio preprocessing with int16 input."""
        stt = WhisperSTT(config)
        # Create int16 audio data
        audio_int16 = np.array([1000, -1000, 500, -500], dtype=np.int16)
        processed = stt._preprocess_audio(audio_int16)
        
        assert processed.dtype == np.float32
        assert np.max(np.abs(processed)) <= 1.0
    
    def test_preprocess_audio_stereo(self, config):
        """Test audio preprocessing with stereo input."""
        stt = WhisperSTT(config)
        # Create stereo audio (2 channels)
        stereo_audio = np.random.randn(1000, 2).astype(np.float32)
        processed = stt._preprocess_audio(stereo_audio)
        
        assert len(processed.shape) == 1  # Should be mono
        assert len(processed) == 1000
    
    def test_preprocess_audio_normalization(self, config):
        """Test audio normalization for values > 1.0."""
        stt = WhisperSTT(config)
        # Create audio with values > 1.0
        loud_audio = np.array([2.0, -3.0, 1.5, -2.5], dtype=np.float32)
        processed = stt._preprocess_audio(loud_audio)
        
        assert np.max(np.abs(processed)) <= 1.0
        # Check that relative amplitudes are preserved (normalized by max absolute value which is 3.0)
        # Original: [2.0, -3.0, 1.5, -2.5] -> Normalized: [0.667, -1.0, 0.5, -0.833]
        assert abs(processed[1]) == 1.0  # Largest absolute value should be 1.0
        assert processed[0] > processed[2]  # 2.0 > 1.5 -> 0.667 > 0.5
        assert abs(processed[3]) > processed[2]  # |-2.5| > 1.5 -> 0.833 > 0.5
    
    async def test_transcribe_not_initialized(self, config, sample_audio):
        """Test transcription when model is not initialized."""
        stt = WhisperSTT(config)
        
        with pytest.raises(PipelineError) as exc_info:
            await stt.transcribe(sample_audio)
        
        assert exc_info.value.stage == PipelineStage.SPEECH_TO_TEXT
        assert "not initialized" in str(exc_info.value)
        assert exc_info.value.recoverable
    
    @patch('src.stt.whisper_stt.WhisperModel')
    async def test_transcribe_success(self, mock_whisper_model, config, sample_audio):
        """Test successful transcription."""
        # Mock the model and its transcribe method
        mock_model = Mock()
        mock_segment1 = Mock()
        mock_segment1.text = " Hello "
        mock_segment2 = Mock()
        mock_segment2.text = " world "
        
        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        
        mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)
        mock_whisper_model.return_value = mock_model
        
        stt = WhisperSTT(config)
        await stt.initialize()
        
        result = await stt.transcribe(sample_audio)
        
        assert result == "Hello world"
        mock_model.transcribe.assert_called_once()
    
    @patch('src.stt.whisper_stt.WhisperModel')
    async def test_transcribe_failure(self, mock_whisper_model, config, sample_audio):
        """Test transcription failure handling."""
        mock_model = Mock()
        mock_model.transcribe.side_effect = Exception("Transcription failed")
        mock_whisper_model.return_value = mock_model
        
        stt = WhisperSTT(config)
        await stt.initialize()
        
        with pytest.raises(PipelineError) as exc_info:
            await stt.transcribe(sample_audio)
        
        assert exc_info.value.stage == PipelineStage.SPEECH_TO_TEXT
        assert "Transcription failed" in str(exc_info.value)
        assert exc_info.value.recoverable
    
    def test_get_model_info(self, config):
        """Test getting model information."""
        stt = WhisperSTT(config)
        info = stt.get_model_info()
        
        assert info["model_size"] == "base"
        assert info["device"] == "cuda"
        assert info["compute_type"] == "float16"
        assert info["initialized"] == False
        assert "available_models" in info
    
    def test_set_model_size_valid(self, config):
        """Test setting valid model size."""
        stt = WhisperSTT(config)
        stt.set_model_size("small")
        
        assert stt.model_size == "small"
    
    def test_set_model_size_invalid(self, config):
        """Test setting invalid model size raises error."""
        stt = WhisperSTT(config)
        
        with pytest.raises(ValueError):
            stt.set_model_size("invalid_model")
    
    def test_set_model_size_same(self, config):
        """Test setting same model size doesn't change anything."""
        stt = WhisperSTT(config)
        original_size = stt.model_size
        stt.set_model_size(original_size)
        
        assert stt.model_size == original_size
    
    async def test_cleanup(self, config):
        """Test cleanup method."""
        stt = WhisperSTT(config)
        stt._initialized = True
        stt.model = Mock()
        
        await stt.cleanup()
        
        assert not stt.is_initialized
        assert stt.model is None


@pytest.mark.integration
class TestWhisperSTTIntegration:
    """Integration tests for WhisperSTT (require actual model loading)."""
    
    @pytest.mark.slow
    async def test_real_model_loading(self):
        """Test loading a real Whisper model (slow test)."""
        config = PipelineConfig(
            audio_device_id=0,
            stt_model_size="tiny",  # Use smallest model for faster testing
            gpu_device="cpu"  # Use CPU to avoid GPU requirements in CI
        )
        
        stt = WhisperSTT(config)
        
        try:
            await stt.initialize()
            assert stt.is_initialized
            assert stt.model is not None
            
            # Test with simple audio
            sample_rate = 16000
            duration = 0.5  # Short duration for faster test
            t = np.linspace(0, duration, int(sample_rate * duration))
            # Create silence (should transcribe to empty or minimal text)
            audio = np.zeros_like(t, dtype=np.float32)
            
            result = await stt.transcribe(audio)
            assert isinstance(result, str)
            
        finally:
            await stt.cleanup()