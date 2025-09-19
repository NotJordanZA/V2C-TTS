"""
Unit tests for the pipeline orchestration system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from datetime import datetime

from src.core.pipeline import VoicePipeline, PipelineState, PipelineMetrics, ProcessingItem
from src.core.interfaces import (
    PipelineError, PipelineStage, AudioChunk, CharacterProfile, VoiceModel
)
from src.core.config import AppConfig, AudioConfig, STTConfig, CharacterConfig, TTSConfig, PerformanceConfig, LoggingConfig


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return AppConfig(
        audio=AudioConfig(
            sample_rate=16000,
            chunk_size=1024,
            input_device_id=0,
            output_device_id=0
        ),
        stt=STTConfig(model_size="base", device="cpu"),
        character=CharacterConfig(
            default_character="test",
            llm_model_path="test_model.gguf"
        ),
        tts=TTSConfig(model_path="test_tts.pth", device="cpu"),
        performance=PerformanceConfig(max_latency_ms=2000),
        logging=LoggingConfig(level="INFO")
    )


@pytest.fixture
def mock_components():
    """Create mock pipeline components."""
    audio_capture = AsyncMock()
    stt_processor = AsyncMock()
    character_transformer = Mock()  # Synchronous methods
    tts_processor = Mock()  # Synchronous methods for load_voice_model
    audio_output = AsyncMock()
    
    # Set up async methods
    character_transformer.initialize = AsyncMock()
    character_transformer.cleanup = AsyncMock()
    character_transformer.transform_text = AsyncMock()
    
    tts_processor.initialize = AsyncMock()
    tts_processor.cleanup = AsyncMock()
    tts_processor.synthesize = AsyncMock()
    
    return {
        'audio_capture': audio_capture,
        'stt_processor': stt_processor,
        'character_transformer': character_transformer,
        'tts_processor': tts_processor,
        'audio_output': audio_output
    }


@pytest.fixture
def sample_character():
    """Create a sample character profile."""
    return CharacterProfile(
        name="test_character",
        description="Test character",
        personality_traits=["friendly", "energetic"],
        speech_patterns={"hello": "hiya"},
        vocabulary_preferences={"greetings": ["hiya", "hey"]},
        transformation_prompt="Transform text to be friendly",
        voice_model_path="test_voice.pth"
    )


@pytest.fixture
def sample_audio_chunk():
    """Create a sample audio chunk."""
    return AudioChunk(
        data=np.random.random(1024).astype(np.float32),
        timestamp=1234567890.0,
        sample_rate=16000,
        duration_ms=64.0
    )


class TestPipelineMetrics:
    """Test pipeline metrics functionality."""
    
    def test_metrics_initialization(self):
        """Test metrics are initialized correctly."""
        metrics = PipelineMetrics()
        
        assert metrics.total_latency_ms == 0.0
        assert metrics.stt_latency_ms == 0.0
        assert metrics.transform_latency_ms == 0.0
        assert metrics.tts_latency_ms == 0.0
        assert metrics.audio_output_latency_ms == 0.0
        assert metrics.processed_chunks == 0
        assert metrics.successful_transformations == 0
        assert metrics.failed_transformations == 0
        assert metrics.start_time is None
        assert metrics.last_update is None
    
    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        metrics = PipelineMetrics()
        
        # Set some values
        metrics.total_latency_ms = 100.0
        metrics.processed_chunks = 5
        metrics.successful_transformations = 3
        
        # Reset
        metrics.reset()
        
        assert metrics.total_latency_ms == 0.0
        assert metrics.processed_chunks == 0
        assert metrics.successful_transformations == 0
        assert isinstance(metrics.start_time, datetime)
        assert isinstance(metrics.last_update, datetime)


class TestProcessingItem:
    """Test processing item functionality."""
    
    def test_processing_item_initialization(self, sample_audio_chunk):
        """Test processing item is initialized correctly."""
        item = ProcessingItem(audio_chunk=sample_audio_chunk)
        
        assert item.audio_chunk == sample_audio_chunk
        assert item.transcribed_text is None
        assert item.transformed_text is None
        assert item.generated_audio is None
        assert isinstance(item.timestamp, float)
        assert item.stage == PipelineStage.AUDIO_CAPTURE
    
    def test_processing_item_stage_progression(self):
        """Test processing item stage can be updated."""
        item = ProcessingItem()
        
        item.stage = PipelineStage.SPEECH_TO_TEXT
        assert item.stage == PipelineStage.SPEECH_TO_TEXT
        
        item.stage = PipelineStage.CHARACTER_TRANSFORM
        assert item.stage == PipelineStage.CHARACTER_TRANSFORM


class TestVoicePipeline:
    """Test voice pipeline orchestrator."""
    
    def test_pipeline_initialization(self, mock_config):
        """Test pipeline is initialized correctly."""
        pipeline = VoicePipeline(mock_config)
        
        assert pipeline.config == mock_config
        assert pipeline._state == PipelineState.STOPPED
        assert pipeline._current_character is None
        assert isinstance(pipeline._metrics, PipelineMetrics)
        assert pipeline._stt_queue.maxsize == 10
        assert pipeline._transform_queue.maxsize == 10
        assert pipeline._tts_queue.maxsize == 10
        assert pipeline._output_queue.maxsize == 10
    
    def test_set_components(self, mock_config, mock_components):
        """Test setting pipeline components."""
        pipeline = VoicePipeline(mock_config)
        
        pipeline.set_components(**mock_components)
        
        assert pipeline._audio_capture == mock_components['audio_capture']
        assert pipeline._stt_processor == mock_components['stt_processor']
        assert pipeline._character_transformer == mock_components['character_transformer']
        assert pipeline._tts_processor == mock_components['tts_processor']
        assert pipeline._audio_output == mock_components['audio_output']
    
    @pytest.mark.asyncio
    async def test_start_pipeline_success(self, mock_config, mock_components):
        """Test successful pipeline startup."""
        pipeline = VoicePipeline(mock_config)
        pipeline.set_components(**mock_components)
        
        # Mock component initialization
        for component in mock_components.values():
            component.initialize = AsyncMock()
        
        await pipeline.start_pipeline()
        
        assert pipeline._state == PipelineState.RUNNING
        assert pipeline._metrics.start_time is not None
        
        # Verify all components were initialized
        for component in mock_components.values():
            component.initialize.assert_called_once()
        
        # Cleanup
        await pipeline.stop_pipeline()
    
    @pytest.mark.asyncio
    async def test_start_pipeline_already_running(self, mock_config):
        """Test starting pipeline when already running raises error."""
        pipeline = VoicePipeline(mock_config)
        pipeline._state = PipelineState.RUNNING
        
        with pytest.raises(PipelineError) as exc_info:
            await pipeline.start_pipeline()
        
        assert "Cannot start pipeline in state" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_start_pipeline_component_failure(self, mock_config, mock_components):
        """Test pipeline startup failure when component initialization fails."""
        pipeline = VoicePipeline(mock_config)
        pipeline.set_components(**mock_components)
        
        # Make one component fail initialization
        mock_components['stt_processor'].initialize.side_effect = Exception("Init failed")
        
        with pytest.raises(PipelineError) as exc_info:
            await pipeline.start_pipeline()
        
        assert "Component initialization failed" in str(exc_info.value)
        assert pipeline._state == PipelineState.ERROR
    
    @pytest.mark.asyncio
    async def test_stop_pipeline_success(self, mock_config, mock_components):
        """Test successful pipeline shutdown."""
        pipeline = VoicePipeline(mock_config)
        pipeline.set_components(**mock_components)
        
        # Start pipeline first
        for component in mock_components.values():
            component.initialize = AsyncMock()
            component.cleanup = AsyncMock()
        
        await pipeline.start_pipeline()
        assert pipeline._state == PipelineState.RUNNING
        
        # Stop pipeline
        await pipeline.stop_pipeline()
        
        assert pipeline._state == PipelineState.STOPPED
        
        # Verify all components were cleaned up
        for component in mock_components.values():
            component.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_pipeline_already_stopped(self, mock_config):
        """Test stopping pipeline when already stopped."""
        pipeline = VoicePipeline(mock_config)
        
        # Should not raise error
        await pipeline.stop_pipeline()
        assert pipeline._state == PipelineState.STOPPED
    
    def test_set_character_success(self, mock_config, mock_components, sample_character):
        """Test successful character setting."""
        pipeline = VoicePipeline(mock_config)
        pipeline.set_components(**mock_components)
        
        # Mock character transformer
        mock_components['character_transformer'].load_character.return_value = sample_character
        
        pipeline.set_character("test_character")
        
        assert pipeline._current_character == sample_character
        mock_components['character_transformer'].load_character.assert_called_once_with("test_character")
    
    def test_set_character_no_transformer(self, mock_config):
        """Test setting character when no transformer is available."""
        pipeline = VoicePipeline(mock_config)
        
        # Should not raise error, just log warning
        pipeline.set_character("test_character")
        assert pipeline._current_character is None
    
    def test_set_character_failure(self, mock_config, mock_components):
        """Test character setting failure."""
        pipeline = VoicePipeline(mock_config)
        pipeline.set_components(**mock_components)
        
        # Mock character transformer failure
        mock_components['character_transformer'].load_character.side_effect = Exception("Load failed")
        
        with pytest.raises(PipelineError) as exc_info:
            pipeline.set_character("test_character")
        
        assert "Failed to load character" in str(exc_info.value)
    
    def test_get_pipeline_status(self, mock_config, sample_character):
        """Test getting pipeline status."""
        pipeline = VoicePipeline(mock_config)
        pipeline._current_character = sample_character
        pipeline._metrics.processed_chunks = 5
        pipeline._metrics.successful_transformations = 3
        pipeline._metrics.start_time = datetime.now()
        
        status = pipeline.get_pipeline_status()
        
        assert status["state"] == PipelineState.STOPPED.value
        assert status["current_character"] == "test_character"
        assert status["legacy_metrics"]["processed_chunks"] == 5
        assert status["legacy_metrics"]["successful_transformations"] == 3
        assert "uptime_seconds" in status["legacy_metrics"]
        assert "queue_sizes" in status
        assert "stt_queue" in status["queue_sizes"]
        assert "performance_metrics" in status
        assert "optimization_suggestions" in status
    
    def test_on_audio_captured(self, mock_config, sample_audio_chunk):
        """Test audio capture callback."""
        pipeline = VoicePipeline(mock_config)
        
        # Call the callback
        pipeline._on_audio_captured(sample_audio_chunk)
        
        # Check that item was added to STT queue
        assert pipeline._stt_queue.qsize() == 1
        assert pipeline._metrics.processed_chunks == 1
    
    def test_on_audio_captured_queue_full(self, mock_config, sample_audio_chunk):
        """Test audio capture callback when queue is full."""
        pipeline = VoicePipeline(mock_config)
        
        # Fill the queue
        for _ in range(10):  # maxsize is 10
            pipeline._stt_queue.put_nowait(ProcessingItem())
        
        # This should not raise error, just log warning
        pipeline._on_audio_captured(sample_audio_chunk)
        
        # Queue should still be full
        assert pipeline._stt_queue.qsize() == 10
    
    @pytest.mark.asyncio
    async def test_stt_worker(self, mock_config, mock_components, sample_audio_chunk):
        """Test STT worker processing."""
        pipeline = VoicePipeline(mock_config)
        pipeline.set_components(**mock_components)
        
        # Mock STT processor
        mock_components['stt_processor'].transcribe.return_value = "Hello world"
        
        # Add item to STT queue
        item = ProcessingItem(audio_chunk=sample_audio_chunk)
        await pipeline._stt_queue.put(item)
        
        # Set shutdown event after processing
        async def delayed_shutdown():
            await asyncio.sleep(0.1)
            pipeline._shutdown_event.set()
        
        # Start worker and shutdown task
        worker_task = asyncio.create_task(pipeline._stt_worker())
        shutdown_task = asyncio.create_task(delayed_shutdown())
        
        await asyncio.gather(worker_task, shutdown_task, return_exceptions=True)
        
        # Check that item was processed and moved to transform queue
        assert pipeline._transform_queue.qsize() == 1
        processed_item = await pipeline._transform_queue.get()
        assert processed_item.transcribed_text == "Hello world"
        assert processed_item.stage == PipelineStage.SPEECH_TO_TEXT
    
    @pytest.mark.asyncio
    async def test_transform_worker(self, mock_config, mock_components, sample_character):
        """Test character transform worker processing."""
        pipeline = VoicePipeline(mock_config)
        pipeline.set_components(**mock_components)
        pipeline._current_character = sample_character
        
        # Mock character transformer
        mock_components['character_transformer'].transform_text.return_value = "Hiya world!"
        
        # Add item to transform queue
        item = ProcessingItem(transcribed_text="Hello world")
        await pipeline._transform_queue.put(item)
        
        # Set shutdown event after processing
        async def delayed_shutdown():
            await asyncio.sleep(0.1)
            pipeline._shutdown_event.set()
        
        # Start worker and shutdown task
        worker_task = asyncio.create_task(pipeline._transform_worker())
        shutdown_task = asyncio.create_task(delayed_shutdown())
        
        await asyncio.gather(worker_task, shutdown_task, return_exceptions=True)
        
        # Check that item was processed and moved to TTS queue
        assert pipeline._tts_queue.qsize() == 1
        processed_item = await pipeline._tts_queue.get()
        assert processed_item.transformed_text == "Hiya world!"
        assert processed_item.stage == PipelineStage.CHARACTER_TRANSFORM
        assert pipeline._metrics.successful_transformations == 1
    
    @pytest.mark.asyncio
    async def test_tts_worker(self, mock_config, mock_components, sample_character):
        """Test TTS worker processing."""
        pipeline = VoicePipeline(mock_config)
        pipeline.set_components(**mock_components)
        pipeline._current_character = sample_character
        
        # Mock TTS processor and voice model
        mock_voice_model = VoiceModel(
            name="test_voice",
            model_path="test_voice.pth",
            sample_rate=22050,
            language="en",
            gender="female"
        )
        mock_components['tts_processor'].load_voice_model.return_value = mock_voice_model
        mock_components['tts_processor'].synthesize.return_value = b"audio_data"
        
        # Add item to TTS queue
        item = ProcessingItem(transformed_text="Hiya world!")
        await pipeline._tts_queue.put(item)
        
        # Set shutdown event after processing
        async def delayed_shutdown():
            await asyncio.sleep(0.1)
            pipeline._shutdown_event.set()
        
        # Start worker and shutdown task
        worker_task = asyncio.create_task(pipeline._tts_worker())
        shutdown_task = asyncio.create_task(delayed_shutdown())
        
        await asyncio.gather(worker_task, shutdown_task, return_exceptions=True)
        
        # Check that item was processed and moved to output queue
        assert pipeline._output_queue.qsize() == 1
        processed_item = await pipeline._output_queue.get()
        assert processed_item.generated_audio == b"audio_data"
        assert processed_item.stage == PipelineStage.TEXT_TO_SPEECH
    
    @pytest.mark.asyncio
    async def test_output_worker(self, mock_config, mock_components):
        """Test audio output worker processing."""
        pipeline = VoicePipeline(mock_config)
        pipeline.set_components(**mock_components)
        
        # Add item to output queue
        item = ProcessingItem(generated_audio=b"audio_data")
        await pipeline._output_queue.put(item)
        
        # Set shutdown event after processing
        async def delayed_shutdown():
            await asyncio.sleep(0.1)
            pipeline._shutdown_event.set()
        
        # Start worker and shutdown task
        worker_task = asyncio.create_task(pipeline._output_worker())
        shutdown_task = asyncio.create_task(delayed_shutdown())
        
        await asyncio.gather(worker_task, shutdown_task, return_exceptions=True)
        
        # Check that audio was played
        mock_components['audio_output'].play_audio.assert_called_once_with(
            b"audio_data",
            mock_config.audio.sample_rate
        )
        assert item.stage == PipelineStage.AUDIO_OUTPUT


if __name__ == "__main__":
    pytest.main([__file__])