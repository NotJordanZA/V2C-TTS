"""
Integration tests for application startup and initialization.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import threading
import time

from src.core.application import ApplicationLifecycleManager, ApplicationState, InitializationProgress
from src.core.config import ConfigManager, AppConfig
from src.core.interfaces import PipelineError


class TestApplicationStartup:
    """Test application startup and initialization scenarios."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        temp_dir = tempfile.mkdtemp()
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create minimal config file
        config_content = """
audio:
  sample_rate: 16000
  chunk_size: 1024
  input_device_id: -1
  output_device_id: -1

stt:
  model_size: "base"
  device: "cpu"

character:
  default_character: "neutral"
  llm_model_path: "models/test_model.gguf"

tts:
  model_path: "models/test_tts.pth"
  device: "cpu"

performance:
  max_latency_ms: 2000

logging:
  level: "INFO"
  file: "logs/test.log"
"""
        
        config_file = config_dir / "default_config.yaml"
        config_file.write_text(config_content)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def app_manager(self, temp_config_dir):
        """Create application manager with test configuration."""
        with patch('src.core.config.ConfigManager') as mock_config_manager:
            # Mock config manager to use test directory
            mock_instance = Mock()
            mock_config_manager.return_value = mock_instance
            
            # Create test config
            test_config = AppConfig(
                audio=Mock(input_device_id=-1, output_device_id=-1, sample_rate=16000, chunk_size=1024),
                stt=Mock(model_size="base", device="cpu", language="auto"),
                character=Mock(default_character="neutral", llm_model_path="models/test.gguf", 
                             max_tokens=256, temperature=0.7),
                tts=Mock(model_path="models/test.pth", device="cpu", sample_rate=22050),
                performance=Mock(max_latency_ms=2000),
                logging=Mock(level="INFO", file="logs/test.log", max_file_size="10MB", backup_count=5)
            )
            
            mock_instance.load_config.return_value = test_config
            mock_instance.validate_config.return_value = True
            mock_instance.get_config.return_value = test_config
            
            app_manager = ApplicationLifecycleManager()
            return app_manager
    
    @pytest.mark.asyncio
    async def test_successful_initialization(self, app_manager):
        """Test successful application initialization."""
        progress_updates = []
        
        def progress_callback(progress: InitializationProgress):
            progress_updates.append(progress)
        
        app_manager.set_progress_callback(progress_callback)
        
        # Mock all component classes
        with patch('src.core.application.AudioCapture') as mock_audio_capture:
            with patch('src.core.application.AudioOutput') as mock_audio_output:
                with patch('src.core.application.STTProcessor') as mock_stt:
                    with patch('src.core.application.CharacterTransformer') as mock_char:
                        with patch('src.core.application.TTSProcessor') as mock_tts:
                            with patch('src.core.application.VoicePipeline') as mock_pipeline:
                                with patch('src.core.application.MainWindow') as mock_window:
                                    # Setup mock instances with async methods
                                    for mock_class in [mock_audio_capture, mock_audio_output, 
                                                     mock_stt, mock_char, mock_tts]:
                                        mock_instance = Mock()
                                        mock_instance.initialize = AsyncMock()
                                        mock_class.return_value = mock_instance
                                    
                                    # Mock UI initialization
                                    with patch('threading.Thread') as mock_thread:
                                        mock_thread_instance = Mock()
                                        mock_thread.return_value = mock_thread_instance
                                        
                                        with patch.object(app_manager, '_ui_ready_event') as mock_event:
                                            mock_event.wait.return_value = True
                                            app_manager._main_window = Mock()  # Simulate successful UI creation
                                            
                                            # Run initialization
                                            result = await app_manager.initialize_application()
        
        # Verify initialization succeeded
        assert result is True
        assert app_manager.get_application_state() == ApplicationState.READY
        
        # Verify progress updates
        assert len(progress_updates) > 0
        assert progress_updates[0].stage == "config"
        assert progress_updates[-1].progress == 1.0
        
        # Verify no errors in progress
        for progress in progress_updates:
            assert progress.error is None
    
    @pytest.mark.asyncio
    async def test_configuration_error(self, app_manager):
        """Test initialization failure due to configuration error."""
        progress_updates = []
        
        def progress_callback(progress: InitializationProgress):
            progress_updates.append(progress)
        
        app_manager.set_progress_callback(progress_callback)
        
        # Mock configuration error
        with patch.object(app_manager, '_initialize_configuration') as mock_init_config:
            mock_init_config.side_effect = Exception("Configuration validation failed")
            
            result = await app_manager.initialize_application()
        
        # Verify initialization failed
        assert result is False
        assert app_manager.get_application_state() == ApplicationState.ERROR
        
        # Verify error was reported
        error_progress = [p for p in progress_updates if p.error is not None]
        assert len(error_progress) > 0
        assert "Configuration validation failed" in error_progress[0].error
    
    @pytest.mark.asyncio
    async def test_audio_system_error(self, app_manager):
        """Test initialization failure due to audio system error."""
        progress_updates = []
        
        def progress_callback(progress: InitializationProgress):
            progress_updates.append(progress)
        
        app_manager.set_progress_callback(progress_callback)
        
        # Mock successful config initialization
        with patch.object(app_manager, '_initialize_configuration'):
            with patch.object(app_manager, '_validate_directories'):
                # Mock audio system error
                with patch.object(app_manager, '_initialize_audio_system') as mock_audio:
                    mock_audio.side_effect = Exception("Audio device not found")
                    
                    result = await app_manager.initialize_application()
        
        # Verify initialization failed
        assert result is False
        assert app_manager.get_application_state() == ApplicationState.ERROR
        
        # Verify error was reported
        error_progress = [p for p in progress_updates if p.error is not None]
        assert len(error_progress) > 0
        assert "Audio device not found" in error_progress[0].error
    
    @pytest.mark.asyncio
    async def test_model_loading_error(self, app_manager):
        """Test initialization failure due to model loading error."""
        progress_updates = []
        
        def progress_callback(progress: InitializationProgress):
            progress_updates.append(progress)
        
        app_manager.set_progress_callback(progress_callback)
        
        # Mock successful early stages
        with patch.object(app_manager, '_initialize_configuration'):
            with patch.object(app_manager, '_validate_directories'):
                with patch.object(app_manager, '_initialize_audio_system'):
                    # Mock model loading error
                    with patch.object(app_manager, '_initialize_ai_models') as mock_models:
                        mock_models.side_effect = Exception("Model file not found")
                        
                        result = await app_manager.initialize_application()
        
        # Verify initialization failed
        assert result is False
        assert app_manager.get_application_state() == ApplicationState.ERROR
        
        # Verify error was reported
        error_progress = [p for p in progress_updates if p.error is not None]
        assert len(error_progress) > 0
        assert "Model file not found" in error_progress[0].error
    
    @pytest.mark.asyncio
    async def test_ui_initialization_timeout(self, app_manager):
        """Test initialization failure due to UI timeout."""
        progress_updates = []
        
        def progress_callback(progress: InitializationProgress):
            progress_updates.append(progress)
        
        app_manager.set_progress_callback(progress_callback)
        
        # Mock successful early stages
        with patch.object(app_manager, '_initialize_configuration'):
            with patch.object(app_manager, '_validate_directories'):
                with patch.object(app_manager, '_initialize_audio_system'):
                    with patch.object(app_manager, '_initialize_ai_models'):
                        with patch.object(app_manager, '_initialize_pipeline'):
                            # Mock UI timeout
                            with patch.object(app_manager, '_ui_ready_event') as mock_event:
                                mock_event.wait.return_value = False  # Timeout
                                
                                result = await app_manager.initialize_application()
        
        # Verify initialization failed
        assert result is False
        assert app_manager.get_application_state() == ApplicationState.ERROR
        
        # Verify error was reported
        error_progress = [p for p in progress_updates if p.error is not None]
        assert len(error_progress) > 0
        assert "UI initialization timeout" in error_progress[0].error
    
    @pytest.mark.asyncio
    async def test_progress_reporting(self, app_manager):
        """Test that progress is reported correctly during initialization."""
        progress_updates = []
        
        def progress_callback(progress: InitializationProgress):
            progress_updates.append(progress)
        
        app_manager.set_progress_callback(progress_callback)
        
        # Mock all stages to succeed
        with patch.multiple(
            app_manager,
            _initialize_configuration=AsyncMock(),
            _validate_directories=AsyncMock(),
            _initialize_audio_system=AsyncMock(),
            _initialize_ai_models=AsyncMock(),
            _initialize_pipeline=AsyncMock(),
            _initialize_ui=AsyncMock(),
            _perform_final_validation=AsyncMock()
        ):
            result = await app_manager.initialize_application()
        
        # Verify initialization succeeded
        assert result is True
        
        # Verify progress stages
        expected_stages = ["config", "directories", "audio", "models", "pipeline", "ui", "validation"]
        reported_stages = [p.stage for p in progress_updates]
        
        for stage in expected_stages:
            assert stage in reported_stages
        
        # Verify progress increases
        progress_values = [p.progress for p in progress_updates]
        assert progress_values[0] == 0.1  # First stage
        assert progress_values[-1] == 1.0  # Final stage
        
        # Verify progress is monotonically increasing
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i-1]
    
    def test_system_status_reporting(self, app_manager):
        """Test system status reporting."""
        # Test initial state
        status = app_manager.get_system_status()
        assert status["application_state"] == ApplicationState.INITIALIZING.value
        assert status["initialization_complete"] is False
        assert status["pipeline_available"] is False
        assert status["ui_available"] is False
        
        # Mock successful initialization
        app_manager._state = ApplicationState.READY
        app_manager._pipeline = Mock()
        app_manager._main_window = Mock()
        
        # Mock pipeline status
        mock_pipeline_status = {
            "state": "stopped",
            "current_character": None,
            "queue_sizes": {"stt_queue": 0}
        }
        app_manager._pipeline.get_pipeline_status.return_value = mock_pipeline_status
        
        status = app_manager.get_system_status()
        assert status["application_state"] == ApplicationState.READY.value
        assert status["initialization_complete"] is True
        assert status["pipeline_available"] is True
        assert status["ui_available"] is True
        assert "pipeline_status" in status
    
    def test_initialization_progress_history(self, app_manager):
        """Test initialization progress history tracking."""
        # Initially no progress
        progress_history = app_manager.get_initialization_progress()
        assert len(progress_history) == 0
        
        # Report some progress
        app_manager._report_progress("test_stage", 0.5, "Test message")
        app_manager._report_progress("test_stage2", 1.0, "Complete", "Test error")
        
        # Verify history
        progress_history = app_manager.get_initialization_progress()
        assert len(progress_history) == 2
        
        assert progress_history[0].stage == "test_stage"
        assert progress_history[0].progress == 0.5
        assert progress_history[0].message == "Test message"
        assert progress_history[0].error is None
        
        assert progress_history[1].stage == "test_stage2"
        assert progress_history[1].progress == 1.0
        assert progress_history[1].message == "Complete"
        assert progress_history[1].error == "Test error"


class TestApplicationValidation:
    """Test application configuration validation scenarios."""
    
    @pytest.mark.asyncio
    async def test_directory_creation(self):
        """Test that required directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            app_manager = ApplicationLifecycleManager()
            
            # Change to temp directory
            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(temp_dir)
                
                # Run directory validation
                await app_manager._validate_directories()
                
                # Verify directories were created
                required_dirs = ["logs", "models", "characters", "config"]
                for dir_name in required_dirs:
                    assert Path(dir_name).exists()
                    assert Path(dir_name).is_dir()
                    
            finally:
                os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test configuration validation during startup."""
        app_manager = ApplicationLifecycleManager()
        
        # Mock valid configuration
        with patch('src.core.application.ConfigManager') as mock_config_manager:
            mock_instance = Mock()
            mock_config_manager.return_value = mock_instance
            
            # Test valid config
            valid_config = Mock()
            mock_instance.load_config.return_value = valid_config
            mock_instance.validate_config.return_value = True
            
            await app_manager._initialize_configuration()
            assert app_manager._config == valid_config
            
            # Test invalid config
            mock_instance.validate_config.return_value = False
            
            with pytest.raises(Exception) as exc_info:
                await app_manager._initialize_configuration()
            
            assert "Configuration validation failed" in str(exc_info.value)


class TestApplicationIntegration:
    """Integration tests for complete application scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_initialization_cycle(self):
        """Test complete initialization and shutdown cycle."""
        app_manager = ApplicationLifecycleManager()
        
        # Setup mocks
        config_manager_mock = Mock()
        config_mock = Mock()
        config_manager_mock.load_config.return_value = config_mock
        config_manager_mock.validate_config.return_value = True
        
        with patch('src.core.application.ConfigManager', return_value=config_manager_mock):
            # Mock all component classes
            with patch('src.core.application.AudioCapture') as mock_audio_capture:
                with patch('src.core.application.AudioOutput') as mock_audio_output:
                    with patch('src.core.application.STTProcessor') as mock_stt:
                        with patch('src.core.application.CharacterTransformer') as mock_char:
                            with patch('src.core.application.TTSProcessor') as mock_tts:
                                with patch('src.core.application.VoicePipeline') as mock_pipeline:
                                    with patch('src.core.application.MainWindow') as mock_window:
                                        # Setup mock instances with async methods
                                        for mock_class in [mock_audio_capture, mock_audio_output, 
                                                         mock_stt, mock_char, mock_tts]:
                                            mock_instance = Mock()
                                            mock_instance.initialize = AsyncMock()
                                            mock_instance.cleanup = AsyncMock()
                                            mock_class.return_value = mock_instance
                                        
                                        # Mock UI thread
                                        with patch('threading.Thread'):
                                            with patch.object(app_manager, '_ui_ready_event') as mock_event:
                                                mock_event.wait.return_value = True
                                                app_manager._main_window = Mock()
                                                
                                                # Test initialization
                                                result = await app_manager.initialize_application()
                                                assert result is True
                                                assert app_manager.get_application_state() == ApplicationState.READY
                                                
                                                # Test shutdown
                                                await app_manager.shutdown_application()
                                                assert app_manager.get_application_state() == ApplicationState.STOPPED
    
    @pytest.mark.asyncio
    async def test_error_recovery_during_initialization(self):
        """Test error recovery mechanisms during initialization."""
        app_manager = ApplicationLifecycleManager()
        
        # Test that errors in one stage don't prevent cleanup
        with patch.object(app_manager, '_initialize_configuration') as mock_config:
            mock_config.side_effect = Exception("Config error")
            
            result = await app_manager.initialize_application()
            assert result is False
            assert app_manager.get_application_state() == ApplicationState.ERROR
            
            # Verify we can still attempt shutdown
            await app_manager.shutdown_application()
            # Should not raise exception even in error state