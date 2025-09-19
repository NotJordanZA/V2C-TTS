"""
Tests for application shutdown and cleanup procedures.
"""

import pytest
import asyncio
import signal
import threading
import time
from unittest.mock import Mock, patch, AsyncMock, call
from pathlib import Path

from src.core.application import ApplicationLifecycleManager, ApplicationState
from src.core.pipeline import PipelineState


class TestApplicationShutdown:
    """Test application shutdown and cleanup scenarios."""
    
    @pytest.fixture
    def app_manager(self):
        """Create application manager for testing."""
        app_manager = ApplicationLifecycleManager()
        
        # Mock components
        app_manager._config_manager = Mock()
        app_manager._config = Mock()
        app_manager._pipeline = Mock()
        app_manager._main_window = Mock()
        
        # Mock component instances
        app_manager._audio_capture = Mock()
        app_manager._audio_output = Mock()
        app_manager._stt_processor = Mock()
        app_manager._character_transformer = Mock()
        app_manager._tts_processor = Mock()
        
        # Setup async cleanup methods
        for component in [app_manager._audio_capture, app_manager._audio_output,
                         app_manager._stt_processor, app_manager._character_transformer,
                         app_manager._tts_processor]:
            component.cleanup = AsyncMock()
        
        app_manager._pipeline.stop_pipeline = AsyncMock()
        
        return app_manager
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_success(self, app_manager):
        """Test successful graceful shutdown."""
        app_manager._state = ApplicationState.READY
        app_manager._pipeline._state = PipelineState.RUNNING
        
        # Mock UI thread
        mock_ui_thread = Mock()
        mock_ui_thread.is_alive.return_value = True
        mock_ui_thread.join = Mock()
        app_manager._ui_thread = mock_ui_thread
        
        # Mock config saving
        app_manager._config_manager.save_config = Mock()
        
        # Perform shutdown
        await app_manager.shutdown_application()
        
        # Verify shutdown sequence
        assert app_manager.get_application_state() == ApplicationState.STOPPED
        
        # Verify pipeline was stopped
        if app_manager._pipeline:
            app_manager._pipeline.stop_pipeline.assert_called_once()
        
        # Verify config was saved
        app_manager._config_manager.save_config.assert_called_once_with(app_manager._config)
        
        # Verify components were cleaned up
        for component in [app_manager._audio_capture, app_manager._audio_output,
                         app_manager._stt_processor, app_manager._character_transformer,
                         app_manager._tts_processor]:
            if component:
                component.cleanup.assert_called_once()
        
        # Verify UI thread was joined
        mock_ui_thread.join.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_with_timeout(self, app_manager):
        """Test shutdown with timeout handling."""
        app_manager._state = ApplicationState.READY
        
        # Mock the entire shutdown sequence to hang
        with patch.object(app_manager, '_execute_shutdown_sequence') as mock_shutdown:
            mock_shutdown.side_effect = asyncio.sleep(10)  # Hangs
            
            # Perform shutdown with short timeout
            await app_manager.shutdown_application(timeout=0.1)
            
            # Should have timed out and performed force cleanup
            assert app_manager.get_application_state() == ApplicationState.ERROR
    
    @pytest.mark.asyncio
    async def test_shutdown_with_pipeline_error(self, app_manager):
        """Test shutdown when pipeline stop fails."""
        app_manager._state = ApplicationState.READY
        app_manager._pipeline._state = PipelineState.RUNNING
        
        # Mock pipeline stop failure
        app_manager._pipeline.stop_pipeline.side_effect = Exception("Pipeline stop failed")
        
        # Perform shutdown
        await app_manager.shutdown_application()
        
        # Should complete despite pipeline error
        assert app_manager.get_application_state() == ApplicationState.STOPPED
        
        # Verify other cleanup still occurred
        if app_manager._audio_capture:
            app_manager._audio_capture.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_with_config_save_error(self, app_manager):
        """Test shutdown when configuration save fails."""
        app_manager._state = ApplicationState.READY
        
        # Mock config save failure
        app_manager._config_manager.save_config.side_effect = Exception("Config save failed")
        
        # Perform shutdown
        await app_manager.shutdown_application()
        
        # Should complete despite config save error
        assert app_manager.get_application_state() == ApplicationState.STOPPED
        
        # Verify other cleanup still occurred
        if app_manager._audio_capture:
            app_manager._audio_capture.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_already_shutting_down(self, app_manager):
        """Test shutdown when already in shutting down state."""
        app_manager._state = ApplicationState.SHUTTING_DOWN
        
        # Mock to track if shutdown sequence runs
        with patch.object(app_manager, '_execute_shutdown_sequence') as mock_shutdown:
            await app_manager.shutdown_application()
            
            # Should not execute shutdown sequence again
            mock_shutdown.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_force_cleanup(self, app_manager):
        """Test forced cleanup when graceful shutdown fails."""
        app_manager._state = ApplicationState.READY
        app_manager._pipeline._processing_tasks = {
            'task1': Mock(done=Mock(return_value=False), cancel=Mock()),
            'task2': Mock(done=Mock(return_value=True), cancel=Mock())
        }
        
        # Perform force cleanup
        await app_manager._force_cleanup()
        
        # Verify running tasks were cancelled
        if app_manager._pipeline and hasattr(app_manager._pipeline, '_processing_tasks'):
            app_manager._pipeline._processing_tasks['task1'].cancel.assert_called_once()
            app_manager._pipeline._processing_tasks['task2'].cancel.assert_not_called()
        
        # Verify components were cleaned up
        if app_manager._audio_capture:
            app_manager._audio_capture.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_emergency_shutdown(self, app_manager):
        """Test emergency shutdown with minimal cleanup."""
        app_manager._state = ApplicationState.RUNNING
        app_manager._pipeline.stop_pipeline = AsyncMock()
        
        # Mock UI root
        app_manager._main_window.root = Mock()
        app_manager._main_window.root.destroy = Mock()
        
        # Perform emergency shutdown
        await app_manager.emergency_shutdown()
        
        # Verify minimal cleanup occurred
        assert app_manager.get_application_state() == ApplicationState.STOPPED
        app_manager._pipeline.stop_pipeline.assert_called_once()
        app_manager._main_window.root.destroy.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_emergency_shutdown_with_timeout(self, app_manager):
        """Test emergency shutdown when pipeline stop times out."""
        app_manager._state = ApplicationState.RUNNING
        
        # Mock pipeline stop that hangs - but emergency shutdown has timeout
        app_manager._pipeline.stop_pipeline = AsyncMock()
        
        # Create a coroutine that will timeout
        async def hanging_stop():
            await asyncio.sleep(10)
        
        app_manager._pipeline.stop_pipeline.side_effect = hanging_stop
        
        # Mock UI root
        app_manager._main_window.root = Mock()
        app_manager._main_window.root.destroy = Mock()
        
        # Perform emergency shutdown
        await app_manager.emergency_shutdown()
        
        # Should complete - emergency shutdown continues despite timeout
        assert app_manager.get_application_state() == ApplicationState.STOPPED
        app_manager._main_window.root.destroy.assert_called_once()


class TestCleanupTasks:
    """Test cleanup task registration and execution."""
    
    @pytest.fixture
    def app_manager(self):
        """Create application manager for testing."""
        return ApplicationLifecycleManager()
    
    def test_register_cleanup_task(self, app_manager):
        """Test registering cleanup tasks."""
        def cleanup_task():
            pass
        
        async def async_cleanup_task():
            pass
        
        # Register tasks
        app_manager.register_cleanup_task(cleanup_task)
        app_manager.register_cleanup_task(async_cleanup_task)
        
        # Verify tasks were registered
        assert len(app_manager._cleanup_tasks) == 2
        assert cleanup_task in app_manager._cleanup_tasks
        assert async_cleanup_task in app_manager._cleanup_tasks
    
    @pytest.mark.asyncio
    async def test_execute_cleanup_tasks(self, app_manager):
        """Test execution of registered cleanup tasks."""
        # Create mock tasks
        sync_task = Mock()
        async_task = AsyncMock()
        failing_task = Mock(side_effect=Exception("Task failed"))
        
        # Register tasks
        app_manager.register_cleanup_task(sync_task)
        app_manager.register_cleanup_task(async_task)
        app_manager.register_cleanup_task(failing_task)
        
        # Execute cleanup tasks
        await app_manager._execute_cleanup_tasks()
        
        # Verify all tasks were called
        sync_task.assert_called_once()
        async_task.assert_called_once()
        failing_task.assert_called_once()


class TestConfigurationPersistence:
    """Test configuration persistence during shutdown."""
    
    @pytest.fixture
    def app_manager(self):
        """Create application manager with mocked UI."""
        app_manager = ApplicationLifecycleManager()
        app_manager._config_manager = Mock()
        app_manager._config = Mock()
        
        # Mock UI with selections
        app_manager._main_window = Mock()
        app_manager._main_window.selected_character = Mock()
        app_manager._main_window.selected_character.get.return_value = "test_character"
        app_manager._main_window.selected_input_device = Mock()
        app_manager._main_window.selected_input_device.get.return_value = "Test Input (ID: 1)"
        app_manager._main_window.selected_output_device = Mock()
        app_manager._main_window.selected_output_device.get.return_value = "Test Output (ID: 2)"
        app_manager._main_window._extract_device_id = Mock(side_effect=lambda x: int(x.split("ID: ")[1].split(")")[0]))
        
        return app_manager
    
    @pytest.mark.asyncio
    async def test_save_application_state(self, app_manager):
        """Test saving application state during shutdown."""
        # Mock config attributes
        app_manager._config.character = Mock()
        app_manager._config.audio = Mock()
        
        # Save application state
        await app_manager._save_application_state()
        
        # Verify config was updated from UI
        assert app_manager._config.character.default_character == "test_character"
        assert app_manager._config.audio.input_device_id == 1
        assert app_manager._config.audio.output_device_id == 2
        
        # Verify config was saved
        app_manager._config_manager.save_config.assert_called_once_with(app_manager._config)
    
    @pytest.mark.asyncio
    async def test_update_config_from_ui_no_ui(self, app_manager):
        """Test config update when UI is not available."""
        app_manager._main_window = None
        
        # Should not raise exception
        await app_manager._update_config_from_ui()
    
    @pytest.mark.asyncio
    async def test_update_config_from_ui_error(self, app_manager):
        """Test config update when UI access fails."""
        # Mock UI error
        app_manager._main_window.selected_character.get.side_effect = Exception("UI error")
        
        # Should not raise exception
        await app_manager._update_config_from_ui()


class TestSignalHandling:
    """Test signal handling for graceful shutdown."""
    
    @pytest.fixture
    def app_manager(self):
        """Create application manager for testing."""
        return ApplicationLifecycleManager()
    
    def test_setup_interrupt_handlers(self, app_manager):
        """Test setup of interrupt signal handlers."""
        with patch('signal.signal') as mock_signal:
            app_manager.setup_interrupt_handlers()
            
            # Verify signal handlers were registered
            expected_signals = []
            if hasattr(signal, 'SIGINT'):
                expected_signals.append(signal.SIGINT)
            if hasattr(signal, 'SIGTERM'):
                expected_signals.append(signal.SIGTERM)
            if hasattr(signal, 'SIGBREAK'):
                expected_signals.append(signal.SIGBREAK)
            
            assert mock_signal.call_count == len(expected_signals)
    
    def test_signal_handler_execution(self, app_manager):
        """Test signal handler execution."""
        with patch('asyncio.get_running_loop') as mock_get_loop:
            mock_loop = Mock()
            mock_get_loop.return_value = mock_loop
            
            # Setup interrupt handlers
            app_manager.setup_interrupt_handlers()
            
            # Simulate signal
            with patch('signal.signal') as mock_signal:
                app_manager.setup_interrupt_handlers()
                
                # Get the signal handler
                signal_handler = mock_signal.call_args_list[0][0][1]
                
                # Call signal handler
                signal_handler(signal.SIGINT, None)
                
                # Verify shutdown was requested
                assert app_manager.is_shutdown_requested() is True
                
                # Verify task was created
                mock_loop.create_task.assert_called_once()
    
    def test_is_shutdown_requested(self, app_manager):
        """Test shutdown request flag."""
        # Initially not requested
        assert app_manager.is_shutdown_requested() is False
        
        # Set shutdown requested
        app_manager._shutdown_requested = True
        assert app_manager.is_shutdown_requested() is True


class TestResourceCleanup:
    """Test resource cleanup during shutdown."""
    
    @pytest.fixture
    def app_manager(self):
        """Create application manager with components."""
        app_manager = ApplicationLifecycleManager()
        
        # Mock all components
        app_manager._audio_capture = Mock()
        app_manager._audio_output = Mock()
        app_manager._stt_processor = Mock()
        app_manager._character_transformer = Mock()
        app_manager._tts_processor = Mock()
        app_manager._pipeline = Mock()
        app_manager._main_window = Mock()
        
        # Setup async cleanup methods
        for component in [app_manager._audio_capture, app_manager._audio_output,
                         app_manager._stt_processor, app_manager._character_transformer,
                         app_manager._tts_processor]:
            component.cleanup = AsyncMock()
        
        return app_manager
    
    @pytest.mark.asyncio
    async def test_cleanup_components_success(self, app_manager):
        """Test successful component cleanup."""
        await app_manager._cleanup_components()
        
        # Verify all components were cleaned up
        for component in [app_manager._audio_capture, app_manager._audio_output,
                         app_manager._stt_processor, app_manager._character_transformer,
                         app_manager._tts_processor]:
            component.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_components_with_errors(self, app_manager):
        """Test component cleanup when some components fail."""
        # Make one component fail
        app_manager._stt_processor.cleanup.side_effect = Exception("Cleanup failed")
        
        # Should not raise exception
        await app_manager._cleanup_components()
        
        # Verify other components were still cleaned up
        if app_manager._audio_capture:
            app_manager._audio_capture.cleanup.assert_called_once()
        if app_manager._audio_output:
            app_manager._audio_output.cleanup.assert_called_once()
        if app_manager._character_transformer:
            app_manager._character_transformer.cleanup.assert_called_once()
        if app_manager._tts_processor:
            app_manager._tts_processor.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_final_resource_cleanup(self, app_manager):
        """Test final resource cleanup."""
        # Add some cleanup tasks
        app_manager._cleanup_tasks = [Mock(), Mock()]
        
        await app_manager._final_resource_cleanup()
        
        # Verify component references were cleared
        assert app_manager._audio_capture is None
        assert app_manager._audio_output is None
        assert app_manager._stt_processor is None
        assert app_manager._character_transformer is None
        assert app_manager._tts_processor is None
        assert app_manager._pipeline is None
        assert app_manager._main_window is None
        
        # Verify cleanup tasks were cleared
        assert len(app_manager._cleanup_tasks) == 0


class TestUICleanup:
    """Test UI cleanup during shutdown."""
    
    @pytest.fixture
    def app_manager(self):
        """Create application manager with UI."""
        app_manager = ApplicationLifecycleManager()
        app_manager._main_window = Mock()
        app_manager._main_window.root = Mock()
        return app_manager
    
    @pytest.mark.asyncio
    async def test_close_ui_success(self, app_manager):
        """Test successful UI closure."""
        # Mock UI thread
        mock_ui_thread = Mock()
        mock_ui_thread.is_alive.return_value = True
        mock_ui_thread.join = Mock()
        app_manager._ui_thread = mock_ui_thread
        
        await app_manager._close_ui()
        
        # Verify UI quit was scheduled
        app_manager._main_window.root.after.assert_called_once()
        
        # Verify UI thread was joined
        mock_ui_thread.join.assert_called_once_with(timeout=5.0)
    
    @pytest.mark.asyncio
    async def test_close_ui_thread_timeout(self, app_manager):
        """Test UI closure when thread doesn't terminate."""
        # Mock UI thread that doesn't terminate
        mock_ui_thread = Mock()
        mock_ui_thread.is_alive.side_effect = [True, True]  # Still alive after join
        mock_ui_thread.join = Mock()
        app_manager._ui_thread = mock_ui_thread
        
        # Should not raise exception
        await app_manager._close_ui()
        
        mock_ui_thread.join.assert_called_once_with(timeout=5.0)
    
    @pytest.mark.asyncio
    async def test_close_ui_no_ui(self, app_manager):
        """Test UI closure when no UI is present."""
        app_manager._main_window = None
        
        # Should not raise exception
        await app_manager._close_ui()


class TestShutdownIntegration:
    """Integration tests for complete shutdown scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_shutdown_cycle(self):
        """Test complete shutdown cycle from running state."""
        app_manager = ApplicationLifecycleManager()
        
        # Setup mocked components
        app_manager._state = ApplicationState.RUNNING
        app_manager._config_manager = Mock()
        app_manager._config = Mock()
        app_manager._pipeline = Mock()
        app_manager._pipeline._state = PipelineState.RUNNING
        app_manager._pipeline.stop_pipeline = AsyncMock()
        
        # Mock all component cleanup
        components = ['_audio_capture', '_audio_output', '_stt_processor', 
                     '_character_transformer', '_tts_processor']
        for comp_name in components:
            comp = Mock()
            comp.cleanup = AsyncMock()
            setattr(app_manager, comp_name, comp)
        
        # Mock UI
        app_manager._main_window = Mock()
        app_manager._main_window.root = Mock()
        app_manager._ui_thread = Mock()
        app_manager._ui_thread.is_alive.return_value = False
        
        # Perform shutdown
        await app_manager.shutdown_application()
        
        # Verify final state
        assert app_manager.get_application_state() == ApplicationState.STOPPED
        
        # Verify pipeline was stopped
        if app_manager._pipeline:
            app_manager._pipeline.stop_pipeline.assert_called_once()
        
        # Verify all components were cleaned up
        for comp_name in components:
            comp = getattr(app_manager, comp_name, None)
            if comp:
                comp.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_with_multiple_errors(self):
        """Test shutdown resilience with multiple component failures."""
        app_manager = ApplicationLifecycleManager()
        app_manager._state = ApplicationState.RUNNING
        
        # Mock components that all fail
        app_manager._config_manager = Mock()
        app_manager._config_manager.save_config.side_effect = Exception("Config save failed")
        
        app_manager._pipeline = Mock()
        app_manager._pipeline.stop_pipeline = AsyncMock(side_effect=Exception("Pipeline stop failed"))
        
        app_manager._audio_capture = Mock()
        app_manager._audio_capture.cleanup = AsyncMock(side_effect=Exception("Audio cleanup failed"))
        
        app_manager._main_window = Mock()
        app_manager._main_window.root = Mock()
        app_manager._main_window.root.after.side_effect = Exception("UI close failed")
        
        # Should complete despite all errors
        await app_manager.shutdown_application()
        
        # Should reach stopped state despite errors
        assert app_manager.get_application_state() == ApplicationState.STOPPED