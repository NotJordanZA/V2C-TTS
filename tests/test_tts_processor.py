"""
Unit tests for TTSProcessor.
"""

import pytest
import asyncio
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import time

from src.tts.processor import TTSProcessor, TTSRequest, TTSResult, TTSProcessingState
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
def tts_processor(config):
    """Create a TTSProcessor instance for testing."""
    return TTSProcessor(config)


@pytest.fixture
def sample_audio():
    """Create sample audio data."""
    return np.random.rand(1000).astype(np.float32)


class TestTTSProcessor:
    """Test cases for TTSProcessor class."""
    
    def test_init(self, config):
        """Test TTSProcessor initialization."""
        processor = TTSProcessor(config)
        
        assert processor.config == config
        assert processor.tts_engine is None
        assert processor.voice_manager is None
        assert not processor.is_running
        assert processor.processing_state == TTSProcessingState.IDLE
        assert processor.total_requests == 0
        assert processor.target_sample_rate == config.sample_rate
    
    @pytest.mark.asyncio
    @patch('src.tts.processor.CoquiTTS')
    @patch('src.tts.processor.VoiceModelManager')
    async def test_initialize_success(self, mock_voice_manager_class, mock_tts_class, tts_processor):
        """Test successful initialization."""
        # Mock TTS engine
        mock_tts = AsyncMock()
        mock_tts_class.return_value = mock_tts
        
        # Mock voice manager
        mock_voice_manager = Mock()
        mock_voice_manager_class.return_value = mock_voice_manager
        
        await tts_processor.initialize()
        
        assert tts_processor.is_initialized
        assert tts_processor.tts_engine == mock_tts
        assert tts_processor.voice_manager == mock_voice_manager
        assert tts_processor.is_running
        mock_tts.initialize.assert_called_once()
        mock_voice_manager.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.tts.processor.CoquiTTS')
    async def test_initialize_failure(self, mock_tts_class, tts_processor):
        """Test initialization failure."""
        mock_tts = AsyncMock()
        mock_tts.initialize.side_effect = Exception("TTS init failed")
        mock_tts_class.return_value = mock_tts
        
        with pytest.raises(PipelineError) as exc_info:
            await tts_processor.initialize()
        
        assert exc_info.value.stage == PipelineStage.TEXT_TO_SPEECH
        assert "TTS processor initialization failed" in str(exc_info.value)
        assert not tts_processor.is_initialized
    
    @pytest.mark.asyncio
    async def test_cleanup(self, tts_processor):
        """Test cleanup method."""
        # Set up initialized state
        mock_tts = AsyncMock()
        tts_processor.tts_engine = mock_tts
        tts_processor.voice_manager = Mock()
        tts_processor._initialized = True
        tts_processor.is_running = True
        
        await tts_processor.cleanup()
        
        assert not tts_processor.is_initialized
        assert not tts_processor.is_running
        assert tts_processor.tts_engine is None
        assert tts_processor.voice_manager is None
        mock_tts.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_stop_processing(self, tts_processor):
        """Test starting and stopping processing worker."""
        # Start processing
        await tts_processor.start_processing()
        assert tts_processor.is_running
        assert tts_processor.worker_task is not None
        
        # Stop processing
        await tts_processor.stop_processing()
        assert not tts_processor.is_running
        assert tts_processor.processing_state == TTSProcessingState.IDLE
    
    @pytest.mark.asyncio
    async def test_synthesize_async_not_initialized(self, tts_processor, voice_model):
        """Test async synthesis when not initialized."""
        with pytest.raises(PipelineError) as exc_info:
            await tts_processor.synthesize_async("Hello world", voice_model)
        
        assert exc_info.value.stage == PipelineStage.TEXT_TO_SPEECH
        assert "not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_synthesize_async_empty_text(self, tts_processor, voice_model):
        """Test async synthesis with empty text."""
        tts_processor._initialized = True
        tts_processor.tts_engine = Mock()
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await tts_processor.synthesize_async("", voice_model)
    
    @pytest.mark.asyncio
    async def test_synthesize_async_success(self, tts_processor, voice_model):
        """Test successful async synthesis."""
        tts_processor._initialized = True
        tts_processor.tts_engine = Mock()
        
        request_id = await tts_processor.synthesize_async("Hello world", voice_model)
        
        assert request_id.startswith("tts_")
        assert tts_processor.request_queue.qsize() == 1
        assert tts_processor.total_requests == 1
    
    @pytest.mark.asyncio
    async def test_synthesize_async_queue_full(self, tts_processor, voice_model):
        """Test async synthesis when queue is full."""
        tts_processor._initialized = True
        tts_processor.tts_engine = Mock()
        tts_processor.max_queue_size = 1
        
        # Fill queue
        await tts_processor.synthesize_async("Hello 1", voice_model)
        
        # Try to add another request
        with pytest.raises(PipelineError, match="queue is full"):
            await tts_processor.synthesize_async("Hello 2", voice_model)
    
    @pytest.mark.asyncio
    async def test_synthesize_sync_success(self, tts_processor, voice_model, sample_audio):
        """Test successful synchronous synthesis."""
        mock_tts = AsyncMock()
        mock_tts.synthesize.return_value = sample_audio
        
        tts_processor._initialized = True
        tts_processor.tts_engine = mock_tts
        
        result = await tts_processor.synthesize_sync("Hello world", voice_model)
        
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        mock_tts.synthesize.assert_called_once_with("Hello world", voice_model)
    
    @pytest.mark.asyncio
    async def test_synthesize_sync_failure(self, tts_processor, voice_model):
        """Test synchronous synthesis failure."""
        mock_tts = AsyncMock()
        mock_tts.synthesize.side_effect = Exception("Synthesis failed")
        
        tts_processor._initialized = True
        tts_processor.tts_engine = mock_tts
        
        with pytest.raises(Exception, match="Synthesis failed"):
            await tts_processor.synthesize_sync("Hello world", voice_model)
        
        # Check that metrics were updated
        assert tts_processor.failed_requests == 1
    
    @pytest.mark.asyncio
    async def test_get_result_timeout(self, tts_processor):
        """Test getting result with timeout."""
        result = await tts_processor.get_result(timeout=0.1)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_result_success(self, tts_processor, voice_model, sample_audio):
        """Test getting result successfully."""
        # Add a result to the queue
        result = TTSResult(
            request_id="test_request",
            audio_data=sample_audio,
            sample_rate=16000,
            processing_time=1.0,
            voice_model=voice_model,
            success=True
        )
        await tts_processor.result_queue.put(result)
        
        retrieved_result = await tts_processor.get_result(timeout=1.0)
        
        assert retrieved_result is not None
        assert retrieved_result.request_id == "test_request"
        assert retrieved_result.success is True
    
    def test_post_process_audio_empty(self, tts_processor):
        """Test post-processing empty audio."""
        empty_audio = np.array([], dtype=np.float32)
        result = tts_processor._post_process_audio(empty_audio, 22050)
        assert len(result) == 0
    
    def test_post_process_audio_normalization(self, tts_processor):
        """Test audio normalization."""
        # Create loud audio that needs normalization
        loud_audio = np.array([2.0, -2.0, 1.5, -1.5], dtype=np.float32)
        
        tts_processor.audio_normalization = True
        result = tts_processor._post_process_audio(loud_audio, 16000)
        
        # Check that audio was normalized
        assert np.max(np.abs(result)) <= 1.0
    
    def test_post_process_audio_volume_boost(self, tts_processor):
        """Test volume boost."""
        audio = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
        
        tts_processor.volume_boost = 2.0
        tts_processor.audio_normalization = False
        result = tts_processor._post_process_audio(audio, 16000)
        
        # Check that volume was boosted
        assert np.allclose(result, audio * 2.0)
    
    def test_resample_audio_no_librosa(self, tts_processor):
        """Test audio resampling when librosa is not available."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        # Mock import to raise ImportError
        with patch('builtins.__import__', side_effect=ImportError("No module named 'librosa'")):
            result = tts_processor._resample_audio(audio, 16000, 22050)
        
        # Should return original audio when librosa is not available
        assert np.array_equal(result, audio)
    
    def test_resample_audio_same_rate(self, tts_processor):
        """Test resampling with same source and target rate."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        result = tts_processor._resample_audio(audio, 16000, 16000)
        
        assert np.array_equal(result, audio)
    
    def test_apply_noise_gate(self, tts_processor):
        """Test noise gate application."""
        # Create audio with quiet and loud sections
        audio = np.concatenate([
            np.full(1024, 0.005),  # Quiet section
            np.full(1024, 0.5)     # Loud section
        ]).astype(np.float32)
        
        result = tts_processor._apply_noise_gate(audio, 0.01)
        
        # Quiet section should be reduced
        assert np.mean(np.abs(result[:1024])) < np.mean(np.abs(audio[:1024]))
        # Loud section should be mostly unchanged
        assert np.allclose(result[1024:], audio[1024:], rtol=0.1)
    
    def test_normalize_audio(self, tts_processor):
        """Test audio normalization."""
        # Create audio that exceeds target peak
        audio = np.array([1.5, -1.2, 0.8, -0.5], dtype=np.float32)
        
        result = tts_processor._normalize_audio(audio)
        
        # Peak should be reduced to target (0.9)
        assert np.max(np.abs(result)) <= 0.9
    
    def test_normalize_audio_quiet(self, tts_processor):
        """Test normalization of quiet audio."""
        # Create quiet audio that doesn't need normalization
        audio = np.array([0.1, -0.2, 0.05, -0.15], dtype=np.float32)
        
        result = tts_processor._normalize_audio(audio)
        
        # Should be unchanged
        assert np.array_equal(result, audio)
    
    def test_update_metrics(self, tts_processor):
        """Test metrics updating."""
        # Test successful request
        tts_processor._update_metrics(1.5, True)
        assert tts_processor.successful_requests == 1
        assert tts_processor.failed_requests == 0
        assert tts_processor.average_processing_time == 1.5
        
        # Test failed request
        tts_processor._update_metrics(2.0, False)
        assert tts_processor.successful_requests == 1
        assert tts_processor.failed_requests == 1
        assert tts_processor.average_processing_time == 1.75  # (1.5 + 2.0) / 2
    
    def test_get_status(self, tts_processor):
        """Test status reporting."""
        tts_processor._initialized = True
        tts_processor.is_running = True
        tts_processor.successful_requests = 5
        tts_processor.failed_requests = 1
        
        status = tts_processor.get_status()
        
        assert status["initialized"] is True
        assert status["running"] is True
        assert status["metrics"]["successful_requests"] == 5
        assert status["metrics"]["failed_requests"] == 1
        assert status["metrics"]["success_rate"] == 5/6
    
    def test_clear_queue(self, tts_processor, voice_model):
        """Test clearing the request queue."""
        # Add some requests to queue
        for i in range(3):
            request = TTSRequest(
                request_id=f"req_{i}",
                text=f"Text {i}",
                voice_model=voice_model,
                timestamp=time.time()
            )
            tts_processor.request_queue.put_nowait(request)
        
        cleared_count = tts_processor.clear_queue()
        
        assert cleared_count == 3
        assert tts_processor.request_queue.qsize() == 0
    
    def test_set_audio_settings(self, tts_processor):
        """Test updating audio settings."""
        tts_processor.set_audio_settings(
            normalization=False,
            volume_boost=1.5,
            noise_gate_threshold=0.02
        )
        
        assert tts_processor.audio_normalization is False
        assert tts_processor.volume_boost == 1.5
        assert tts_processor.noise_gate_threshold == 0.02
    
    def test_set_audio_settings_clamping(self, tts_processor):
        """Test that audio settings are clamped to reasonable ranges."""
        tts_processor.set_audio_settings(
            volume_boost=10.0,  # Too high
            noise_gate_threshold=1.0  # Too high
        )
        
        assert tts_processor.volume_boost == 3.0  # Clamped to max
        assert tts_processor.noise_gate_threshold == 0.5  # Clamped to max
    
    @pytest.mark.asyncio
    async def test_save_audio_to_file(self, tts_processor, sample_audio):
        """Test saving audio to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_audio.wav"
            
            with patch('src.tts.processor.sf.write') as mock_write:
                await tts_processor.save_audio_to_file(sample_audio, str(file_path))
                
                mock_write.assert_called_once_with(
                    str(file_path),
                    sample_audio,
                    tts_processor.target_sample_rate
                )
    
    @pytest.mark.asyncio
    async def test_save_audio_to_file_failure(self, tts_processor, sample_audio):
        """Test saving audio to file failure."""
        with patch('src.tts.processor.sf.write', side_effect=Exception("Write failed")):
            with pytest.raises(PipelineError, match="Failed to save audio"):
                await tts_processor.save_audio_to_file(sample_audio, "/invalid/path.wav")
    
    @pytest.mark.asyncio
    async def test_process_request_success(self, tts_processor, voice_model, sample_audio):
        """Test processing a request successfully."""
        # Setup
        mock_tts = AsyncMock()
        mock_tts.synthesize.return_value = sample_audio
        tts_processor.tts_engine = mock_tts
        
        request = TTSRequest(
            request_id="test_request",
            text="Hello world",
            voice_model=voice_model,
            timestamp=time.time()
        )
        
        # Process request
        await tts_processor._process_request(request)
        
        # Check result was queued
        assert tts_processor.result_queue.qsize() == 1
        result = await tts_processor.result_queue.get()
        
        assert result.request_id == "test_request"
        assert result.success is True
        assert len(result.audio_data) > 0
    
    @pytest.mark.asyncio
    async def test_process_request_failure(self, tts_processor, voice_model):
        """Test processing a request that fails."""
        # Setup
        mock_tts = AsyncMock()
        mock_tts.synthesize.side_effect = Exception("Synthesis failed")
        tts_processor.tts_engine = mock_tts
        
        request = TTSRequest(
            request_id="test_request",
            text="Hello world",
            voice_model=voice_model,
            timestamp=time.time()
        )
        
        # Process request
        await tts_processor._process_request(request)
        
        # Check result was queued
        assert tts_processor.result_queue.qsize() == 1
        result = await tts_processor.result_queue.get()
        
        assert result.request_id == "test_request"
        assert result.success is False
        assert result.error_message == "Synthesis failed"
    
    @pytest.mark.asyncio
    async def test_process_request_timeout(self, tts_processor, voice_model):
        """Test processing a request that has timed out."""
        # Setup
        tts_processor.request_timeout = 0.1  # Very short timeout
        
        request = TTSRequest(
            request_id="test_request",
            text="Hello world",
            voice_model=voice_model,
            timestamp=time.time() - 1.0  # Request from 1 second ago
        )
        
        # Process request
        await tts_processor._process_request(request)
        
        # Check result was queued with timeout error
        assert tts_processor.result_queue.qsize() == 1
        result = await tts_processor.result_queue.get()
        
        assert result.request_id == "test_request"
        assert result.success is False
        assert "timed out" in result.error_message
    
    @pytest.mark.asyncio
    async def test_process_request_with_callback(self, tts_processor, voice_model, sample_audio):
        """Test processing a request with callback."""
        # Setup
        mock_tts = AsyncMock()
        mock_tts.synthesize.return_value = sample_audio
        tts_processor.tts_engine = mock_tts
        
        callback_called = False
        callback_audio = None
        
        def test_callback(audio_data):
            nonlocal callback_called, callback_audio
            callback_called = True
            callback_audio = audio_data
        
        request = TTSRequest(
            request_id="test_request",
            text="Hello world",
            voice_model=voice_model,
            timestamp=time.time(),
            callback=test_callback
        )
        
        # Process request
        await tts_processor._process_request(request)
        
        # Check callback was called
        assert callback_called is True
        assert callback_audio is not None
        assert len(callback_audio) > 0


class TestTTSRequest:
    """Test cases for TTSRequest dataclass."""
    
    def test_tts_request_creation(self, voice_model):
        """Test creating a TTS request."""
        request = TTSRequest(
            request_id="test_id",
            text="Hello world",
            voice_model=voice_model,
            timestamp=123456.789,
            priority=1
        )
        
        assert request.request_id == "test_id"
        assert request.text == "Hello world"
        assert request.voice_model == voice_model
        assert request.timestamp == 123456.789
        assert request.priority == 1
        assert request.callback is None


class TestTTSResult:
    """Test cases for TTSResult dataclass."""
    
    def test_tts_result_creation(self, voice_model, sample_audio):
        """Test creating a TTS result."""
        result = TTSResult(
            request_id="test_id",
            audio_data=sample_audio,
            sample_rate=16000,
            processing_time=1.5,
            voice_model=voice_model,
            success=True,
            error_message=None
        )
        
        assert result.request_id == "test_id"
        assert np.array_equal(result.audio_data, sample_audio)
        assert result.sample_rate == 16000
        assert result.processing_time == 1.5
        assert result.voice_model == voice_model
        assert result.success is True
        assert result.error_message is None


if __name__ == "__main__":
    pytest.main([__file__])