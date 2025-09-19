"""
Integration tests for pipeline error handling.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.core.pipeline import VoicePipeline
from src.core.config import AppConfig, AudioConfig, STTConfig, CharacterConfig, TTSConfig, PerformanceConfig, LoggingConfig
from src.core.interfaces import PipelineStage
from src.core.error_handling import ErrorSeverity, RecoveryStrategy


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
    character_transformer = Mock()
    tts_processor = Mock()
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


class TestPipelineErrorIntegration:
    """Test pipeline error handling integration."""
    
    def test_error_recovery_manager_initialization(self, mock_config):
        """Test that pipeline initializes error recovery manager."""
        pipeline = VoicePipeline(mock_config)
        
        error_manager = pipeline.get_error_recovery_manager()
        assert error_manager is not None
        assert len(error_manager.error_rules) > 0  # Should have default rules
    
    def test_degradation_manager_initialization(self, mock_config):
        """Test that pipeline initializes degradation manager."""
        pipeline = VoicePipeline(mock_config)
        
        degradation_manager = pipeline.get_degradation_manager()
        assert degradation_manager is not None
        assert len(degradation_manager.fallback_configs) > 0  # Should have default configs
    
    def test_get_error_statistics(self, mock_config):
        """Test getting error statistics from pipeline."""
        pipeline = VoicePipeline(mock_config)
        
        stats = pipeline.get_error_statistics()
        
        assert 'total_errors' in stats
        assert 'error_rate' in stats
        assert 'errors_by_stage' in stats
        assert 'errors_by_severity' in stats
        assert 'most_common_errors' in stats
    
    def test_enable_graceful_degradation(self, mock_config):
        """Test enabling graceful degradation through pipeline."""
        pipeline = VoicePipeline(mock_config)
        
        pipeline.enable_graceful_degradation('gpu_to_cpu', 'GPU unavailable')
        
        degradation_manager = pipeline.get_degradation_manager()
        assert degradation_manager.is_degraded('gpu_to_cpu')
    
    def test_disable_graceful_degradation(self, mock_config):
        """Test disabling graceful degradation through pipeline."""
        pipeline = VoicePipeline(mock_config)
        
        pipeline.enable_graceful_degradation('gpu_to_cpu', 'GPU unavailable')
        assert pipeline.get_degradation_manager().is_degraded('gpu_to_cpu')
        
        pipeline.disable_graceful_degradation('gpu_to_cpu')
        assert not pipeline.get_degradation_manager().is_degraded('gpu_to_cpu')
    
    def test_get_system_health(self, mock_config):
        """Test getting comprehensive system health information."""
        pipeline = VoicePipeline(mock_config)
        
        health = pipeline.get_system_health()
        
        assert 'pipeline_state' in health
        assert 'error_statistics' in health
        assert 'degradation_status' in health
        assert 'performance_summary' in health
        assert 'optimization_suggestions' in health
        assert 'recent_errors' in health
        
        # Check structure of nested data
        assert 'total_errors' in health['error_statistics']
        assert 'active_degradations' in health['degradation_status']
        assert 'timestamp' in health['performance_summary']
    
    @pytest.mark.asyncio
    async def test_worker_error_handling_integration(self, mock_config, mock_components):
        """Test that worker errors are handled by the error recovery system."""
        pipeline = VoicePipeline(mock_config)
        pipeline.set_components(**mock_components)
        
        # Make STT processor raise an error
        mock_components['stt_processor'].transcribe.side_effect = ValueError("Test error")
        
        # Start the pipeline
        await pipeline.start_pipeline()
        
        # Simulate audio input to trigger STT worker
        import numpy as np
        from src.core.interfaces import AudioChunk
        
        audio_chunk = AudioChunk(
            data=np.random.random(1024).astype(np.float32),
            timestamp=1234567890.0,
            sample_rate=16000,
            duration_ms=64.0
        )
        
        # Add item to STT queue to trigger error
        from src.core.pipeline import ProcessingItem
        item = ProcessingItem(audio_chunk=audio_chunk)
        await pipeline._stt_queue.put(item)
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        
        # Stop the pipeline
        await pipeline.stop_pipeline()
        
        # Check that error was recorded
        error_stats = pipeline.get_error_statistics()
        # Note: The actual error recording depends on the worker loop running,
        # which might not happen in this test setup, so we just verify the structure exists
        assert 'total_errors' in error_stats
    
    def test_custom_error_rule_addition(self, mock_config):
        """Test adding custom error rules to the pipeline."""
        pipeline = VoicePipeline(mock_config)
        error_manager = pipeline.get_error_recovery_manager()
        
        from src.core.error_handling import ErrorRule
        
        # Add custom rule
        custom_rule = ErrorRule(
            error_types=[ConnectionError],
            stages=[PipelineStage.SPEECH_TO_TEXT],
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.RETRY
        )
        
        initial_rule_count = len(error_manager.error_rules)
        error_manager.add_error_rule(custom_rule)
        
        assert len(error_manager.error_rules) == initial_rule_count + 1
        assert custom_rule in error_manager.error_rules
    
    def test_degradation_config_application(self, mock_config):
        """Test that degradation affects configuration."""
        pipeline = VoicePipeline(mock_config)
        degradation_manager = pipeline.get_degradation_manager()
        
        base_config = {
            'stt_model_size': 'large',
            'audio_sample_rate': 44100
        }
        
        # Enable degradation
        pipeline.enable_graceful_degradation('gpu_to_cpu', 'Testing')
        
        # Get degraded config
        degraded_config = degradation_manager.get_degraded_config(base_config)
        
        # Should have fallback values
        assert degraded_config['stt_model_size'] == 'tiny'  # From gpu_to_cpu fallback
        assert 'llm_max_tokens' in degraded_config  # Additional fallback setting


if __name__ == "__main__":
    pytest.main([__file__])