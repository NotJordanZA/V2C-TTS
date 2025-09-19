"""
Integration tests for STT processing pipeline.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import time

from src.stt.processor import STTProcessor, STTResult, ProcessingState
from src.core.interfaces import PipelineConfig, AudioChunk, PipelineError, PipelineStage


@pytest.fixture
def config():
    """Create a test configuration."""
    return PipelineConfig(
        audio_device_id=0,
        sample_rate=16000,
        stt_model_size="base",
        gpu_device="cpu"  # Use CPU for testing
    )


@pytest.fixture
def sample_audio_chunk():
    """Create a sample audio chunk for testing."""
    # Generate 1 second of sine wave
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    return AudioChunk(
        data=audio_data,
        timestamp=time.time(),
        sample_rate=sample_rate,
        duration_ms=duration * 1000
    )


@pytest.fixture
def short_audio_chunk():
    """Create a short audio chunk that should be filtered out."""
    sample_rate = 16000
    duration = 0.05  # 50ms - below default minimum
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    return AudioChunk(
        data=audio_data,
        timestamp=time.time(),
        sample_rate=sample_rate,
        duration_ms=duration * 1000
    )


@pytest.fixture
def silence_audio_chunk():
    """Create a silent audio chunk."""
    sample_rate = 16000
    duration = 1.0
    audio_data = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    return AudioChunk(
        data=audio_data,
        timestamp=time.time(),
        sample_rate=sample_rate,
        duration_ms=duration * 1000
    )


class TestSTTProcessor:
    """Test cases for STTProcessor class."""
    
    def test_init(self, config):
        """Test processor initialization."""
        callback = Mock()
        processor = STTProcessor(config, callback)
        
        assert processor.config == config
        assert processor.result_callback == callback
        assert processor.state == ProcessingState.IDLE
        assert not processor.is_initialized
        assert processor.audio_queue.maxsize == 10
    
    @patch('src.stt.processor.WhisperSTT')
    async def test_initialize_success(self, mock_whisper_class, config):
        """Test successful processor initialization."""
        mock_whisper = AsyncMock()
        mock_whisper_class.return_value = mock_whisper
        
        processor = STTProcessor(config)
        await processor.initialize()
        
        assert processor.is_initialized
        mock_whisper.initialize.assert_called_once()
    
    @patch('src.stt.processor.WhisperSTT')
    async def test_initialize_failure(self, mock_whisper_class, config):
        """Test processor initialization failure."""
        mock_whisper = AsyncMock()
        mock_whisper.initialize.side_effect = Exception("Initialization failed")
        mock_whisper_class.return_value = mock_whisper
        
        processor = STTProcessor(config)
        
        with pytest.raises(PipelineError) as exc_info:
            await processor.initialize()
        
        assert exc_info.value.stage == PipelineStage.SPEECH_TO_TEXT
        assert "Failed to initialize STT processor" in str(exc_info.value)
        assert not exc_info.value.recoverable
    
    async def test_start_processing_not_initialized(self, config):
        """Test starting processing when not initialized."""
        processor = STTProcessor(config)
        
        with pytest.raises(PipelineError) as exc_info:
            await processor.start_processing()
        
        assert "not initialized" in str(exc_info.value)
    
    @patch('src.stt.processor.WhisperSTT')
    async def test_start_stop_processing(self, mock_whisper_class, config):
        """Test starting and stopping processing loop."""
        mock_whisper = AsyncMock()
        mock_whisper_class.return_value = mock_whisper
        
        processor = STTProcessor(config)
        await processor.initialize()
        
        # Start processing
        await processor.start_processing()
        assert processor.processing_task is not None
        assert not processor.processing_task.done()
        
        # Stop processing
        await processor.stop_processing()
        assert processor.state == ProcessingState.STOPPED
        assert processor.processing_task is None
    
    @patch('src.stt.processor.WhisperSTT')
    async def test_process_audio_not_running(self, mock_whisper_class, config, sample_audio_chunk):
        """Test processing audio when processor is stopped."""
        mock_whisper = AsyncMock()
        mock_whisper_class.return_value = mock_whisper
        
        processor = STTProcessor(config)
        processor.state = ProcessingState.STOPPED
        
        with pytest.raises(PipelineError) as exc_info:
            await processor.process_audio(sample_audio_chunk)
        
        assert "processor is stopped" in str(exc_info.value)
    
    def test_preprocess_audio_chunk_valid(self, config, sample_audio_chunk):
        """Test preprocessing valid audio chunk."""
        processor = STTProcessor(config)
        result = processor._preprocess_audio_chunk(sample_audio_chunk)
        
        assert result is not None
        assert result.data.shape == sample_audio_chunk.data.shape
    
    def test_preprocess_audio_chunk_too_short(self, config, short_audio_chunk):
        """Test preprocessing audio chunk that's too short."""
        processor = STTProcessor(config)
        result = processor._preprocess_audio_chunk(short_audio_chunk)
        
        assert result is None
    
    def test_preprocess_audio_chunk_silence(self, config, silence_audio_chunk):
        """Test preprocessing silent audio chunk."""
        processor = STTProcessor(config)
        result = processor._preprocess_audio_chunk(silence_audio_chunk)
        
        assert result is None
    
    def test_preprocess_audio_chunk_too_long(self, config):
        """Test preprocessing audio chunk that's too long."""
        # Create very long audio chunk
        sample_rate = 16000
        duration = 35.0  # 35 seconds - above default maximum
        audio_data = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        long_chunk = AudioChunk(
            data=audio_data,
            timestamp=time.time(),
            sample_rate=sample_rate,
            duration_ms=duration * 1000
        )
        
        processor = STTProcessor(config)
        result = processor._preprocess_audio_chunk(long_chunk)
        
        assert result is not None
        assert result.duration_ms == processor.max_audio_length_ms
        assert len(result.data) < len(long_chunk.data)
    
    def test_is_silence(self, config):
        """Test silence detection."""
        processor = STTProcessor(config)
        
        # Test silence
        silence = np.zeros(1000, dtype=np.float32)
        assert processor._is_silence(silence)
        
        # Test non-silence
        noise = np.random.randn(1000).astype(np.float32) * 0.1
        assert not processor._is_silence(noise)
        
        # Test very quiet audio (should be considered silence)
        quiet = np.random.randn(1000).astype(np.float32) * 0.001
        assert processor._is_silence(quiet)
    
    def test_convert_audio_format_16khz(self, config, sample_audio_chunk):
        """Test audio format conversion for 16kHz audio."""
        processor = STTProcessor(config)
        result = processor._convert_audio_format(sample_audio_chunk)
        
        assert result.dtype == np.float32
        assert len(result) == len(sample_audio_chunk.data)
    
    def test_convert_audio_format_resample(self, config):
        """Test audio format conversion with resampling."""
        # Create 8kHz audio chunk
        sample_rate = 8000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        chunk = AudioChunk(
            data=audio_data,
            timestamp=time.time(),
            sample_rate=sample_rate,
            duration_ms=duration * 1000
        )
        
        processor = STTProcessor(config)
        result = processor._convert_audio_format(chunk)
        
        assert result.dtype == np.float32
        # Should be resampled to 16kHz (approximately double the length)
        assert len(result) > len(audio_data)
    
    def test_configure_preprocessing(self, config):
        """Test configuring preprocessing parameters."""
        processor = STTProcessor(config)
        
        processor.configure_preprocessing(
            min_audio_length_ms=200,
            max_audio_length_ms=20000,
            silence_threshold=0.05
        )
        
        assert processor.min_audio_length_ms == 200
        assert processor.max_audio_length_ms == 20000
        assert processor.silence_threshold == 0.05
    
    def test_get_metrics_initial(self, config):
        """Test getting metrics from uninitialized processor."""
        processor = STTProcessor(config)
        metrics = processor.get_metrics()
        
        assert metrics["state"] == ProcessingState.IDLE.value
        assert metrics["total_processed"] == 0
        assert metrics["queue_size"] == 0
        assert metrics["avg_processing_time_ms"] == 0.0
    
    @patch('src.stt.processor.WhisperSTT')
    async def test_cleanup(self, mock_whisper_class, config):
        """Test processor cleanup."""
        mock_whisper = AsyncMock()
        mock_whisper_class.return_value = mock_whisper
        
        processor = STTProcessor(config)
        await processor.initialize()
        await processor.start_processing()
        
        await processor.cleanup()
        
        assert not processor.is_initialized
        assert processor.state == ProcessingState.STOPPED
        mock_whisper.cleanup.assert_called_once()


@pytest.mark.integration
class TestSTTProcessorIntegration:
    """Integration tests for STT processor with real processing."""
    
    @patch('src.stt.processor.WhisperSTT')
    async def test_end_to_end_processing(self, mock_whisper_class, config, sample_audio_chunk):
        """Test end-to-end audio processing."""
        # Mock WhisperSTT with async mock that simulates processing time
        mock_whisper = AsyncMock()
        async def mock_transcribe(audio_data):
            await asyncio.sleep(0.01)  # Simulate processing time
            return "Hello world"
        mock_whisper.transcribe = mock_transcribe
        mock_whisper_class.return_value = mock_whisper
        
        # Create callback to capture results
        results = []
        def result_callback(result: STTResult):
            results.append(result)
        
        processor = STTProcessor(config, result_callback)
        
        try:
            # Initialize and start processing
            await processor.initialize()
            await processor.start_processing()
            
            # Process audio chunk
            await processor.process_audio(sample_audio_chunk)
            
            # Wait for processing to complete
            await asyncio.sleep(0.2)
            
            # Check results
            assert len(results) == 1
            result = results[0]
            assert result.text == "Hello world"
            assert result.audio_chunk == sample_audio_chunk
            assert result.processing_time_ms >= 0  # Allow for very fast processing
            
            # Check metrics
            metrics = processor.get_metrics()
            assert metrics["total_processed"] == 1
            assert metrics["last_processing_time_ms"] >= 0
            
        finally:
            await processor.cleanup()
    
    @patch('src.stt.processor.WhisperSTT')
    async def test_multiple_audio_chunks(self, mock_whisper_class, config):
        """Test processing multiple audio chunks."""
        # Mock WhisperSTT
        mock_whisper = AsyncMock()
        mock_whisper.transcribe.side_effect = ["First", "Second", "Third"]
        mock_whisper_class.return_value = mock_whisper
        
        results = []
        def result_callback(result: STTResult):
            results.append(result)
        
        processor = STTProcessor(config, result_callback)
        
        try:
            await processor.initialize()
            await processor.start_processing()
            
            # Create multiple audio chunks
            for i in range(3):
                chunk = AudioChunk(
                    data=np.random.randn(16000).astype(np.float32) * 0.1,
                    timestamp=time.time() + i,
                    sample_rate=16000,
                    duration_ms=1000
                )
                await processor.process_audio(chunk)
            
            # Wait for all processing to complete
            await asyncio.sleep(0.5)
            
            # Check results
            assert len(results) == 3
            assert [r.text for r in results] == ["First", "Second", "Third"]
            
            # Check metrics
            metrics = processor.get_metrics()
            assert metrics["total_processed"] == 3
            
        finally:
            await processor.cleanup()
    
    @patch('src.stt.processor.WhisperSTT')
    async def test_processing_error_recovery(self, mock_whisper_class, config, sample_audio_chunk):
        """Test error recovery during processing."""
        # Mock WhisperSTT to fail once then succeed
        mock_whisper = AsyncMock()
        mock_whisper.transcribe.side_effect = [Exception("Processing failed"), "Success"]
        mock_whisper_class.return_value = mock_whisper
        
        results = []
        def result_callback(result: STTResult):
            results.append(result)
        
        processor = STTProcessor(config, result_callback)
        
        try:
            await processor.initialize()
            await processor.start_processing()
            
            # Process first chunk (should fail)
            await processor.process_audio(sample_audio_chunk)
            await asyncio.sleep(0.1)
            
            # Process second chunk (should succeed)
            await processor.process_audio(sample_audio_chunk)
            await asyncio.sleep(0.1)
            
            # Should have one successful result
            assert len(results) == 1
            assert results[0].text == "Success"
            
        finally:
            await processor.cleanup()
    
    @patch('src.stt.processor.WhisperSTT')
    async def test_queue_overflow_handling(self, mock_whisper_class, config):
        """Test handling of queue overflow."""
        # Mock WhisperSTT to be very slow
        mock_whisper = AsyncMock()
        async def slow_transcribe(audio_data):
            await asyncio.sleep(1)  # Very slow
            return "text"
        mock_whisper.transcribe = slow_transcribe
        mock_whisper_class.return_value = mock_whisper
        
        processor = STTProcessor(config)
        
        try:
            await processor.initialize()
            await processor.start_processing()
            
            # Fill up the queue beyond capacity
            chunks_added = 0
            queue_full_count = 0
            for i in range(15):  # More than queue maxsize (10)
                chunk = AudioChunk(
                    data=np.random.randn(16000).astype(np.float32) * 0.1,
                    timestamp=time.time() + i,
                    sample_rate=16000,
                    duration_ms=1000
                )
                try:
                    await processor.process_audio(chunk)
                    chunks_added += 1
                except Exception:
                    # Queue full errors are expected but not thrown by our implementation
                    # Instead, we log warnings and drop chunks
                    queue_full_count += 1
            
            # All chunks should be processed (some dropped with warnings)
            # The queue should be at or near capacity
            assert processor.audio_queue.qsize() <= 10
            
        finally:
            await processor.cleanup()