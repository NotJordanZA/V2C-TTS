"""
Unit tests for CoquiTTS implementation.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import tempfile
import os

from src.tts.coqui_tts import CoquiTTS
from src.core.interfaces import PipelineConfig, VoiceModel, PipelineError, PipelineStage


@pytest.fixture
def config():
    """Create a test configuration."""
    return PipelineConfig(
        audio_device_id=0,
        sample_rate=16000,
        chunk_size=1024,
        stt_model_size="base",
        llm_model_path="test_llm_path",
        tts_model_path="test_tts_path",
        gpu_device="cuda",
        max_latency_ms=2000
    )


@pytest.fixture
def voice_model():
    """Create a test voice model."""
    return VoiceModel(
        name="test_voice",
        model_path="test_voice.wav",
        sample_rate=22050,
        language="en",
        gender="female"
    )


@pytest.fixture
def coqui_tts(config):
    """Create a CoquiTTS instance for testing."""
    return CoquiTTS(config)


class TestCoquiTTS:
    """Test cases for CoquiTTS class."""
    
    def test_init(self, config):
        """Test CoquiTTS initialization."""
        tts = CoquiTTS(config)
        
        assert tts.config == config
        assert tts.tts_model is None
        assert not tts.is_initialized
        assert tts.loaded_voices == {}
        assert tts.model_name == "tts_models/multilingual/multi-dataset/xtts_v2"
    
    @patch('torch.cuda.is_available')
    def test_get_device_cuda_available(self, mock_cuda_available, config):
        """Test device selection when CUDA is available."""
        mock_cuda_available.return_value = True
        config.gpu_device = "cuda"
        
        tts = CoquiTTS(config)
        assert tts.device == "cuda"
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_get_device_mps_available(self, mock_mps_available, mock_cuda_available, config):
        """Test device selection when MPS is available."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True
        config.gpu_device = "mps"
        
        tts = CoquiTTS(config)
        assert tts.device == "mps"
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_get_device_cpu_fallback(self, mock_mps_available, mock_cuda_available, config):
        """Test fallback to CPU when no GPU is available."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        
        tts = CoquiTTS(config)
        assert tts.device == "cpu"
    
    @pytest.mark.asyncio
    @patch('src.tts.coqui_tts.TTS')
    async def test_initialize_success(self, mock_tts_class, coqui_tts):
        """Test successful initialization."""
        mock_tts_instance = Mock()
        mock_tts_instance.synthesizer = Mock()
        mock_tts_instance.synthesizer.tts_model = Mock()
        mock_tts_instance.synthesizer.tts_model.to = Mock()
        mock_tts_class.return_value = mock_tts_instance
        
        await coqui_tts.initialize()
        
        assert coqui_tts.is_initialized
        assert coqui_tts.tts_model == mock_tts_instance
        mock_tts_class.assert_called_once_with(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=(coqui_tts.device == "cuda")
        )
    
    @pytest.mark.asyncio
    @patch('src.tts.coqui_tts.TTS')
    async def test_initialize_failure(self, mock_tts_class, coqui_tts):
        """Test initialization failure."""
        mock_tts_class.side_effect = Exception("Model loading failed")
        
        with pytest.raises(PipelineError) as exc_info:
            await coqui_tts.initialize()
        
        assert exc_info.value.stage == PipelineStage.TEXT_TO_SPEECH
        assert "TTS initialization failed" in str(exc_info.value)
        assert not exc_info.value.recoverable
        assert not coqui_tts.is_initialized
    
    @pytest.mark.asyncio
    @patch('torch.cuda.empty_cache')
    async def test_cleanup(self, mock_empty_cache, coqui_tts):
        """Test cleanup method."""
        # Set up initialized state
        coqui_tts.tts_model = Mock()
        coqui_tts._initialized = True
        coqui_tts.loaded_voices = {"test": Mock()}
        coqui_tts.device = "cuda"
        
        await coqui_tts.cleanup()
        
        assert coqui_tts.tts_model is None
        assert not coqui_tts.is_initialized
        assert coqui_tts.loaded_voices == {}
        mock_empty_cache.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_synthesize_not_initialized(self, coqui_tts, voice_model):
        """Test synthesis when not initialized."""
        with pytest.raises(PipelineError) as exc_info:
            await coqui_tts.synthesize("Hello world", voice_model)
        
        assert exc_info.value.stage == PipelineStage.TEXT_TO_SPEECH
        assert "not initialized" in str(exc_info.value)
        assert exc_info.value.recoverable
    
    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self, coqui_tts, voice_model):
        """Test synthesis with empty text."""
        coqui_tts._initialized = True
        coqui_tts.tts_model = Mock()
        
        result = await coqui_tts.synthesize("", voice_model)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    @pytest.mark.asyncio
    @patch('asyncio.get_event_loop')
    async def test_synthesize_success(self, mock_get_loop, coqui_tts, voice_model):
        """Test successful synthesis."""
        # Setup
        coqui_tts._initialized = True
        coqui_tts.tts_model = Mock()
        
        mock_loop = Mock()
        mock_get_loop.return_value = mock_loop
        
        expected_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_loop.run_in_executor = AsyncMock(return_value=expected_audio)
        
        # Test
        result = await coqui_tts.synthesize("Hello world", voice_model)
        
        # Verify
        assert np.array_equal(result, expected_audio)
        mock_loop.run_in_executor.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('asyncio.get_event_loop')
    async def test_synthesize_failure(self, mock_get_loop, coqui_tts, voice_model):
        """Test synthesis failure."""
        # Setup
        coqui_tts._initialized = True
        coqui_tts.tts_model = Mock()
        
        mock_loop = Mock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_in_executor = AsyncMock(side_effect=Exception("Synthesis failed"))
        
        # Test
        with pytest.raises(PipelineError) as exc_info:
            await coqui_tts.synthesize("Hello world", voice_model)
        
        assert exc_info.value.stage == PipelineStage.TEXT_TO_SPEECH
        assert "Speech synthesis failed" in str(exc_info.value)
        assert exc_info.value.recoverable
    
    def test_synthesize_sync_with_voice_cloning(self, coqui_tts, voice_model):
        """Test synchronous synthesis with voice cloning."""
        # Setup
        coqui_tts.tts_model = Mock()
        expected_audio = [0.1, 0.2, 0.3]
        coqui_tts.tts_model.tts.return_value = expected_audio
        
        # Create temporary file for voice model
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            voice_model.model_path = tmp_file.name
        
        try:
            # Test
            result = coqui_tts._synthesize_sync("Hello world", voice_model)
            
            # Verify
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            coqui_tts.tts_model.tts.assert_called_once_with(
                text="Hello world",
                speaker_wav=voice_model.model_path,
                language=voice_model.language
            )
        finally:
            os.unlink(tmp_file.name)
    
    def test_synthesize_sync_builtin_speaker(self, coqui_tts, voice_model):
        """Test synchronous synthesis with built-in speaker."""
        # Setup
        coqui_tts.tts_model = Mock()
        expected_audio = [0.1, 0.2, 0.3]
        coqui_tts.tts_model.tts.return_value = expected_audio
        voice_model.model_path = "nonexistent_path.wav"
        
        # Test
        result = coqui_tts._synthesize_sync("Hello world", voice_model)
        
        # Verify
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        coqui_tts.tts_model.tts.assert_called_once_with(
            text="Hello world",
            language=voice_model.language
        )
    
    def test_load_voice_model_success(self, coqui_tts):
        """Test successful voice model loading."""
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(b"fake audio data")
            tmp_path = tmp_file.name
        
        try:
            # Test
            voice_model = coqui_tts.load_voice_model(tmp_path)
            
            # Verify
            assert isinstance(voice_model, VoiceModel)
            assert voice_model.name == Path(tmp_path).stem
            assert voice_model.model_path == tmp_path
            assert voice_model.language == "en"
            assert voice_model.sample_rate == 22050
            
            # Check caching
            assert voice_model.name in coqui_tts.loaded_voices
            
        finally:
            os.unlink(tmp_path)
    
    def test_load_voice_model_not_found(self, coqui_tts):
        """Test voice model loading with non-existent file."""
        with pytest.raises(PipelineError) as exc_info:
            coqui_tts.load_voice_model("nonexistent_file.wav")
        
        assert exc_info.value.stage == PipelineStage.TEXT_TO_SPEECH
        assert "Voice model loading failed" in str(exc_info.value)
        assert exc_info.value.recoverable
    
    def test_get_available_voices_empty(self, coqui_tts):
        """Test getting available voices when none are loaded."""
        voices = coqui_tts.get_available_voices()
        assert isinstance(voices, list)
        assert len(voices) == 0
    
    def test_get_available_voices_with_loaded(self, coqui_tts, voice_model):
        """Test getting available voices with loaded models."""
        coqui_tts.loaded_voices["test_voice"] = voice_model
        
        voices = coqui_tts.get_available_voices()
        
        assert len(voices) >= 1
        assert voice_model in voices
    
    def test_get_available_voices_with_builtin(self, coqui_tts):
        """Test getting available voices with built-in speakers."""
        # Setup mock TTS model with speakers
        mock_tts = Mock()
        mock_tts.speakers = ["speaker1", "speaker2"]
        coqui_tts.tts_model = mock_tts
        
        voices = coqui_tts.get_available_voices()
        
        # Should have built-in voices
        builtin_voices = [v for v in voices if v.name.startswith("builtin_")]
        assert len(builtin_voices) == 2
    
    def test_get_model_info_not_initialized(self, coqui_tts):
        """Test getting model info when not initialized."""
        info = coqui_tts.get_model_info()
        assert info == {}
    
    def test_get_model_info_initialized(self, coqui_tts):
        """Test getting model info when initialized."""
        coqui_tts._initialized = True
        coqui_tts.tts_model = Mock()
        coqui_tts.loaded_voices = {"voice1": Mock(), "voice2": Mock()}
        
        info = coqui_tts.get_model_info()
        
        assert info["model_name"] == coqui_tts.model_name
        assert info["device"] == coqui_tts.device
        assert info["initialized"] is True
        assert info["loaded_voices"] == 2
    
    @pytest.mark.asyncio
    async def test_warmup_not_initialized(self, coqui_tts):
        """Test warmup when not initialized."""
        # Should not raise exception, just log warning
        await coqui_tts.warmup()
        # Test passes if no exception is raised
    
    @pytest.mark.asyncio
    @patch.object(CoquiTTS, 'synthesize')
    async def test_warmup_success(self, mock_synthesize, coqui_tts):
        """Test successful warmup."""
        coqui_tts._initialized = True
        mock_synthesize.return_value = np.array([0.1, 0.2])
        
        await coqui_tts.warmup("Test warmup")
        
        mock_synthesize.assert_called_once()
        args, kwargs = mock_synthesize.call_args
        assert args[0] == "Test warmup"
        assert isinstance(args[1], VoiceModel)
        assert args[1].name == "warmup"
    
    @pytest.mark.asyncio
    @patch.object(CoquiTTS, 'synthesize')
    async def test_warmup_failure(self, mock_synthesize, coqui_tts):
        """Test warmup failure."""
        coqui_tts._initialized = True
        mock_synthesize.side_effect = Exception("Warmup failed")
        
        # Should not raise exception, just log warning
        await coqui_tts.warmup()
        # Test passes if no exception is raised


if __name__ == "__main__":
    pytest.main([__file__])