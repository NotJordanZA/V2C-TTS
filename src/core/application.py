"""
Application lifecycle management for the Voice Character Transformation System.

This module handles application startup, initialization, shutdown, and cleanup
with proper error handling and progress reporting.
"""

import asyncio
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import tkinter as tk
from tkinter import messagebox

from .config import ConfigManager, AppConfig, ConfigValidationError
from .pipeline import VoicePipeline, PipelineState
from .interfaces import PipelineError
from ..ui.main_window import MainWindow
from ..audio.device_manager import AudioDeviceManager
from ..audio.capture import AudioCapture
from ..audio.output import AudioOutput
from ..stt.processor import STTProcessor
from ..character.transformer import CharacterTransformer
from ..tts.processor import TTSProcessor


class ApplicationState(str, Enum):
    """Application lifecycle states."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class InitializationProgress:
    """Progress tracking for application initialization."""
    stage: str
    progress: float  # 0.0 to 1.0
    message: str
    error: Optional[str] = None


class ApplicationError(Exception):
    """Application-specific error."""
    pass


class ApplicationLifecycleManager:
    """Manages the complete application lifecycle."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._state = ApplicationState.INITIALIZING
        self._config_manager: Optional[ConfigManager] = None
        self._config: Optional[AppConfig] = None
        self._pipeline: Optional[VoicePipeline] = None
        self._main_window: Optional[MainWindow] = None
        
        # Component instances
        self._audio_capture: Optional[AudioCapture] = None
        self._audio_output: Optional[AudioOutput] = None
        self._stt_processor: Optional[STTProcessor] = None
        self._character_transformer: Optional[CharacterTransformer] = None
        self._tts_processor: Optional[TTSProcessor] = None
        
        # Progress tracking
        self._initialization_progress: List[InitializationProgress] = []
        self._progress_callback: Optional[Callable[[InitializationProgress], None]] = None
        
        # Shutdown handling
        self._shutdown_requested = False
        self._cleanup_tasks: List[Callable] = []
        
        # UI thread handling
        self._ui_thread: Optional[threading.Thread] = None
        self._ui_ready_event = threading.Event()
    
    def set_progress_callback(self, callback: Callable[[InitializationProgress], None]):
        """Set callback for initialization progress updates."""
        self._progress_callback = callback
    
    def _report_progress(self, stage: str, progress: float, message: str, error: Optional[str] = None):
        """Report initialization progress."""
        progress_info = InitializationProgress(
            stage=stage,
            progress=progress,
            message=message,
            error=error
        )
        
        self._initialization_progress.append(progress_info)
        self.logger.info(f"Initialization progress: {stage} ({progress*100:.1f}%) - {message}")
        
        if error:
            self.logger.error(f"Initialization error in {stage}: {error}")
        
        if self._progress_callback:
            self._progress_callback(progress_info)
    
    async def initialize_application(self) -> bool:
        """
        Initialize the application with proper error handling and progress reporting.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self._state = ApplicationState.INITIALIZING
            self.logger.info("Starting application initialization")
            
            # Stage 1: Configuration validation (10%)
            self._report_progress("config", 0.1, "Loading and validating configuration")
            await self._initialize_configuration()
            
            # Stage 2: Directory structure validation (20%)
            self._report_progress("directories", 0.2, "Validating directory structure")
            await self._validate_directories()
            
            # Stage 3: Audio system initialization (30%)
            self._report_progress("audio", 0.3, "Initializing audio system")
            await self._initialize_audio_system()
            
            # Stage 4: Model loading (60%)
            self._report_progress("models", 0.4, "Loading AI models")
            await self._initialize_ai_models()
            
            # Stage 5: Pipeline setup (80%)
            self._report_progress("pipeline", 0.8, "Setting up processing pipeline")
            await self._initialize_pipeline()
            
            # Stage 6: UI initialization (90%)
            self._report_progress("ui", 0.9, "Initializing user interface")
            await self._initialize_ui()
            
            # Stage 7: Final validation (100%)
            self._report_progress("validation", 1.0, "Performing final validation")
            await self._perform_final_validation()
            
            self._state = ApplicationState.READY
            self.logger.info("Application initialization completed successfully")
            return True
            
        except Exception as e:
            self._state = ApplicationState.ERROR
            error_msg = f"Application initialization failed: {e}"
            self.logger.error(error_msg)
            self._report_progress("error", 0.0, "Initialization failed", str(e))
            return False
    
    async def _initialize_configuration(self):
        """Initialize and validate configuration."""
        try:
            self._config_manager = ConfigManager()
            self._config = self._config_manager.load_config()
            
            # Validate configuration
            if not self._config_manager.validate_config(self._config):
                raise ApplicationError("Configuration validation failed")
            
            self.logger.info("Configuration loaded and validated successfully")
            
        except ConfigValidationError as e:
            raise ApplicationError(f"Configuration error: {e}")
        except Exception as e:
            raise ApplicationError(f"Failed to load configuration: {e}")
    
    async def _validate_directories(self):
        """Validate required directory structure."""
        required_dirs = [
            Path("logs"),
            Path("models"),
            Path("characters"),
            Path("config")
        ]
        
        for dir_path in required_dirs:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Validated directory: {dir_path}")
            except Exception as e:
                raise ApplicationError(f"Failed to create directory {dir_path}: {e}")
    
    async def _initialize_audio_system(self):
        """Initialize audio capture and output systems."""
        try:
            # Initialize audio device manager
            device_manager = AudioDeviceManager()
            
            # Initialize audio capture
            self._audio_capture = AudioCapture(
                device_id=self._config.audio.input_device_id,
                sample_rate=self._config.audio.sample_rate,
                chunk_size=self._config.audio.chunk_size
            )
            
            # Initialize audio output
            self._audio_output = AudioOutput(
                device_id=self._config.audio.output_device_id,
                sample_rate=self._config.audio.sample_rate
            )
            
            # Test audio system
            await self._audio_capture.initialize()
            await self._audio_output.initialize()
            
            self.logger.info("Audio system initialized successfully")
            
        except Exception as e:
            raise ApplicationError(f"Audio system initialization failed: {e}")
    
    async def _initialize_ai_models(self):
        """Initialize AI models with progress reporting."""
        try:
            # Initialize STT processor (20% of model loading)
            self._report_progress("models", 0.45, "Loading speech-to-text model")
            self._stt_processor = STTProcessor(
                model_size=self._config.stt.model_size,
                device=self._config.stt.device,
                language=self._config.stt.language
            )
            await self._stt_processor.initialize()
            
            # Initialize character transformer (40% of model loading)
            self._report_progress("models", 0.55, "Loading character transformation model")
            self._character_transformer = CharacterTransformer(
                model_path=self._config.character.llm_model_path,
                max_tokens=self._config.character.max_tokens,
                temperature=self._config.character.temperature
            )
            await self._character_transformer.initialize()
            
            # Initialize TTS processor (40% of model loading)
            self._report_progress("models", 0.7, "Loading text-to-speech model")
            self._tts_processor = TTSProcessor(
                model_path=self._config.tts.model_path,
                device=self._config.tts.device,
                sample_rate=self._config.tts.sample_rate
            )
            await self._tts_processor.initialize()
            
            self.logger.info("AI models loaded successfully")
            
        except Exception as e:
            raise ApplicationError(f"AI model initialization failed: {e}")
    
    async def _initialize_pipeline(self):
        """Initialize the processing pipeline."""
        try:
            self._pipeline = VoicePipeline(self._config)
            
            # Inject components into pipeline
            self._pipeline.set_components(
                audio_capture=self._audio_capture,
                stt_processor=self._stt_processor,
                character_transformer=self._character_transformer,
                tts_processor=self._tts_processor,
                audio_output=self._audio_output
            )
            
            self.logger.info("Processing pipeline initialized successfully")
            
        except Exception as e:
            raise ApplicationError(f"Pipeline initialization failed: {e}")
    
    async def _initialize_ui(self):
        """Initialize the user interface."""
        try:
            # Create main window in a separate thread
            def create_ui():
                try:
                    self._main_window = MainWindow(self._config_manager)
                    self._setup_ui_callbacks()
                    self._ui_ready_event.set()
                    
                    # Start UI main loop
                    self._main_window.root.mainloop()
                    
                except Exception as e:
                    self.logger.error(f"UI initialization error: {e}")
                    self._ui_ready_event.set()
            
            # Start UI thread
            self._ui_thread = threading.Thread(target=create_ui, daemon=True)
            self._ui_thread.start()
            
            # Wait for UI to be ready (with timeout)
            if not self._ui_ready_event.wait(timeout=10.0):
                raise ApplicationError("UI initialization timeout")
            
            if self._main_window is None:
                raise ApplicationError("UI initialization failed")
            
            self.logger.info("User interface initialized successfully")
            
        except Exception as e:
            raise ApplicationError(f"UI initialization failed: {e}")
    
    def _setup_ui_callbacks(self):
        """Setup callbacks between UI and pipeline."""
        if not self._main_window or not self._pipeline:
            return
        
        # Set up UI callbacks
        self._main_window.on_character_changed = self._on_character_changed
        self._main_window.on_input_device_changed = self._on_input_device_changed
        self._main_window.on_output_device_changed = self._on_output_device_changed
        self._main_window.on_start_pipeline = self._on_start_pipeline
        self._main_window.on_stop_pipeline = self._on_stop_pipeline
    
    async def _perform_final_validation(self):
        """Perform final validation of all systems."""
        try:
            # Validate all components are initialized
            components = [
                ("Configuration", self._config),
                ("Audio Capture", self._audio_capture),
                ("Audio Output", self._audio_output),
                ("STT Processor", self._stt_processor),
                ("Character Transformer", self._character_transformer),
                ("TTS Processor", self._tts_processor),
                ("Pipeline", self._pipeline),
                ("Main Window", self._main_window)
            ]
            
            for name, component in components:
                if component is None:
                    raise ApplicationError(f"{name} not initialized")
            
            # Test basic functionality
            await self._run_system_tests()
            
            self.logger.info("Final validation completed successfully")
            
        except Exception as e:
            raise ApplicationError(f"Final validation failed: {e}")
    
    async def _run_system_tests(self):
        """Run basic system tests to ensure everything works."""
        try:
            # Test character loading
            if self._character_transformer:
                available_characters = self._character_transformer.get_available_characters()
                if not available_characters:
                    self.logger.warning("No character profiles available")
                else:
                    # Load default character
                    default_char = self._config.character.default_character
                    if default_char in available_characters:
                        self._pipeline.set_character(default_char)
                    else:
                        self._pipeline.set_character(available_characters[0])
            
            self.logger.info("System tests completed successfully")
            
        except Exception as e:
            self.logger.warning(f"System test warning: {e}")
            # Don't fail initialization for test warnings
    
    def _on_character_changed(self, character_name: str):
        """Handle character change from UI."""
        try:
            if self._pipeline:
                self._pipeline.set_character(character_name)
                self.logger.info(f"Character changed to: {character_name}")
        except Exception as e:
            self.logger.error(f"Failed to change character: {e}")
            if self._main_window:
                messagebox.showerror("Character Error", f"Failed to change character: {e}")
    
    def _on_input_device_changed(self, device_id: int):
        """Handle input device change from UI."""
        try:
            if self._audio_capture:
                # Update configuration
                self._config.audio.input_device_id = device_id
                # Note: Device change while running would require pipeline restart
                self.logger.info(f"Input device changed to ID: {device_id}")
        except Exception as e:
            self.logger.error(f"Failed to change input device: {e}")
    
    def _on_output_device_changed(self, device_id: int):
        """Handle output device change from UI."""
        try:
            if self._audio_output:
                # Update configuration
                self._config.audio.output_device_id = device_id
                # Note: Device change while running would require pipeline restart
                self.logger.info(f"Output device changed to ID: {device_id}")
        except Exception as e:
            self.logger.error(f"Failed to change output device: {e}")
    
    def _on_start_pipeline(self):
        """Handle pipeline start request from UI."""
        try:
            if self._pipeline:
                # Start pipeline in background
                asyncio.create_task(self._start_pipeline_async())
        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}")
            if self._main_window:
                messagebox.showerror("Pipeline Error", f"Failed to start pipeline: {e}")
    
    def _on_stop_pipeline(self):
        """Handle pipeline stop request from UI."""
        try:
            if self._pipeline:
                # Stop pipeline in background
                asyncio.create_task(self._stop_pipeline_async())
        except Exception as e:
            self.logger.error(f"Failed to stop pipeline: {e}")
            if self._main_window:
                messagebox.showerror("Pipeline Error", f"Failed to stop pipeline: {e}")
    
    async def _start_pipeline_async(self):
        """Start pipeline asynchronously."""
        try:
            if self._pipeline:
                await self._pipeline.start_pipeline()
                self._state = ApplicationState.RUNNING
                self.logger.info("Pipeline started successfully")
        except Exception as e:
            self.logger.error(f"Pipeline start failed: {e}")
            if self._main_window:
                # Schedule UI update in main thread
                self._main_window.root.after(0, lambda: messagebox.showerror(
                    "Pipeline Error", f"Failed to start pipeline: {e}"
                ))
    
    async def _stop_pipeline_async(self):
        """Stop pipeline asynchronously."""
        try:
            if self._pipeline:
                await self._pipeline.stop_pipeline()
                self._state = ApplicationState.READY
                self.logger.info("Pipeline stopped successfully")
        except Exception as e:
            self.logger.error(f"Pipeline stop failed: {e}")
            if self._main_window:
                # Schedule UI update in main thread
                self._main_window.root.after(0, lambda: messagebox.showerror(
                    "Pipeline Error", f"Failed to stop pipeline: {e}"
                ))
    
    def run_application(self):
        """Run the application main loop."""
        try:
            if self._state != ApplicationState.READY:
                raise ApplicationError("Application not properly initialized")
            
            self.logger.info("Starting application main loop")
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Wait for UI thread to complete or shutdown signal
            if self._ui_thread:
                self._ui_thread.join()
            
            self.logger.info("Application main loop completed")
            
        except Exception as e:
            self.logger.error(f"Application runtime error: {e}")
            raise ApplicationError(f"Application runtime error: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown")
            self._shutdown_requested = True
            asyncio.create_task(self.shutdown_application())
        
        # Setup handlers for common shutdown signals
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown_application(self, timeout: float = 30.0):
        """
        Gracefully shutdown the application with timeout and resource cleanup.
        
        Args:
            timeout: Maximum time to wait for shutdown completion in seconds
        """
        if self._state == ApplicationState.SHUTTING_DOWN:
            return
        
        self._state = ApplicationState.SHUTTING_DOWN
        self.logger.info("Starting graceful application shutdown")
        
        shutdown_start_time = time.time()
        
        try:
            # Execute shutdown sequence with timeout
            await asyncio.wait_for(
                self._execute_shutdown_sequence(),
                timeout=timeout
            )
            
            self._state = ApplicationState.STOPPED
            shutdown_duration = time.time() - shutdown_start_time
            self.logger.info(f"Application shutdown completed successfully in {shutdown_duration:.2f}s")
            
        except asyncio.TimeoutError:
            self.logger.error(f"Shutdown timeout after {timeout}s, forcing cleanup")
            await self._force_cleanup()
            self._state = ApplicationState.ERROR
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            await self._force_cleanup()
            self._state = ApplicationState.ERROR
    
    async def _execute_shutdown_sequence(self):
        """Execute the complete shutdown sequence."""
        shutdown_tasks = []
        
        # Step 1: Stop pipeline gracefully
        if self._pipeline:
            self.logger.info("Stopping processing pipeline")
            try:
                if hasattr(self._pipeline, '_state') and self._pipeline._state == PipelineState.RUNNING:
                    await self._pipeline.stop_pipeline()
                self.logger.info("Pipeline stopped successfully")
            except Exception as e:
                self.logger.error(f"Error stopping pipeline: {e}")
        
        # Step 2: Save current configuration and settings
        await self._save_application_state()
        
        # Step 3: Cleanup components in reverse order of initialization
        await self._cleanup_components()
        
        # Step 4: Execute custom cleanup tasks
        await self._execute_cleanup_tasks()
        
        # Step 5: Close UI gracefully
        await self._close_ui()
        
        # Step 6: Final resource cleanup
        await self._final_resource_cleanup()
    
    async def _save_application_state(self):
        """Save application configuration and current state."""
        try:
            if self._config_manager and self._config:
                self.logger.info("Saving application configuration")
                
                # Update config with current UI selections if available
                if self._main_window:
                    await self._update_config_from_ui()
                
                # Save configuration
                self._config_manager.save_config(self._config)
                self.logger.info("Configuration saved successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to save application state: {e}")
    
    async def _update_config_from_ui(self):
        """Update configuration with current UI selections."""
        try:
            if not self._main_window:
                return
            
            # Update character selection
            if hasattr(self._main_window, 'selected_character'):
                current_character = self._main_window.selected_character.get()
                if current_character:
                    self._config.character.default_character = current_character
            
            # Update audio device selections
            if hasattr(self._main_window, 'selected_input_device'):
                input_device_text = self._main_window.selected_input_device.get()
                if input_device_text:
                    device_id = self._main_window._extract_device_id(input_device_text)
                    if device_id is not None:
                        self._config.audio.input_device_id = device_id
            
            if hasattr(self._main_window, 'selected_output_device'):
                output_device_text = self._main_window.selected_output_device.get()
                if output_device_text:
                    device_id = self._main_window._extract_device_id(output_device_text)
                    if device_id is not None:
                        self._config.audio.output_device_id = device_id
            
            self.logger.debug("Configuration updated from UI selections")
            
        except Exception as e:
            self.logger.error(f"Failed to update config from UI: {e}")
    
    async def _execute_cleanup_tasks(self):
        """Execute registered cleanup tasks."""
        self.logger.info(f"Executing {len(self._cleanup_tasks)} cleanup tasks")
        
        for i, cleanup_task in enumerate(self._cleanup_tasks):
            try:
                if asyncio.iscoroutinefunction(cleanup_task):
                    await cleanup_task()
                else:
                    cleanup_task()
                self.logger.debug(f"Cleanup task {i+1} completed")
            except Exception as e:
                self.logger.error(f"Cleanup task {i+1} failed: {e}")
    
    async def _close_ui(self):
        """Close the user interface gracefully."""
        if self._main_window:
            try:
                self.logger.info("Closing user interface")
                
                # Schedule UI close in the main thread
                if hasattr(self._main_window, 'root'):
                    self._main_window.root.after(0, self._main_window.root.quit)
                
                # Wait for UI thread to complete (with timeout)
                if self._ui_thread and self._ui_thread.is_alive():
                    self._ui_thread.join(timeout=5.0)
                    
                    if self._ui_thread.is_alive():
                        self.logger.warning("UI thread did not terminate gracefully")
                
                self.logger.info("User interface closed")
                
            except Exception as e:
                self.logger.error(f"Error closing UI: {e}")
    
    async def _final_resource_cleanup(self):
        """Perform final resource cleanup."""
        try:
            # Clear component references
            self._audio_capture = None
            self._audio_output = None
            self._stt_processor = None
            self._character_transformer = None
            self._tts_processor = None
            self._pipeline = None
            self._main_window = None
            
            # Clear cleanup tasks
            self._cleanup_tasks.clear()
            
            self.logger.debug("Final resource cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error in final cleanup: {e}")
    
    async def _force_cleanup(self):
        """Force cleanup when graceful shutdown fails."""
        self.logger.warning("Performing forced cleanup")
        
        try:
            # Force stop pipeline
            if self._pipeline:
                try:
                    # Cancel any running tasks
                    if hasattr(self._pipeline, '_processing_tasks'):
                        for task in self._pipeline._processing_tasks.values():
                            if not task.done():
                                task.cancel()
                except Exception as e:
                    self.logger.error(f"Error force-stopping pipeline: {e}")
            
            # Force cleanup components
            await self._cleanup_components()
            
            # Force close UI
            if self._main_window and hasattr(self._main_window, 'root'):
                try:
                    self._main_window.root.destroy()
                except Exception as e:
                    self.logger.error(f"Error force-closing UI: {e}")
            
            # Final cleanup
            await self._final_resource_cleanup()
            
        except Exception as e:
            self.logger.error(f"Error in forced cleanup: {e}")
    
    def register_cleanup_task(self, cleanup_task: Callable):
        """
        Register a cleanup task to be executed during shutdown.
        
        Args:
            cleanup_task: Function or coroutine to execute during cleanup
        """
        self._cleanup_tasks.append(cleanup_task)
        task_name = getattr(cleanup_task, '__name__', str(cleanup_task))
        self.logger.debug(f"Registered cleanup task: {task_name}")
    
    def setup_interrupt_handlers(self):
        """Setup interrupt signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received interrupt signal {signum}")
            self._shutdown_requested = True
            
            # Create shutdown task
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, create new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop:
                loop.create_task(self.shutdown_application())
        
        # Register signal handlers
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)
        
        self.logger.info("Interrupt handlers registered")
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested
    
    async def emergency_shutdown(self):
        """Perform emergency shutdown with minimal cleanup."""
        self.logger.critical("Performing emergency shutdown")
        self._state = ApplicationState.SHUTTING_DOWN
        
        try:
            # Only essential cleanup
            if self._pipeline:
                try:
                    await asyncio.wait_for(self._pipeline.stop_pipeline(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.error("Pipeline stop timeout during emergency shutdown")
            
            # Force close UI
            if self._main_window and hasattr(self._main_window, 'root'):
                self._main_window.root.destroy()
            
            self._state = ApplicationState.STOPPED
            
        except Exception as e:
            self.logger.error(f"Error during emergency shutdown: {e}")
            self._state = ApplicationState.ERROR
    
    async def _cleanup_components(self):
        """Cleanup all application components."""
        components = [
            ("TTS Processor", self._tts_processor),
            ("Character Transformer", self._character_transformer),
            ("STT Processor", self._stt_processor),
            ("Audio Output", self._audio_output),
            ("Audio Capture", self._audio_capture)
        ]
        
        for name, component in components:
            if component:
                try:
                    await component.cleanup()
                    self.logger.debug(f"Cleaned up {name}")
                except Exception as e:
                    self.logger.error(f"Error cleaning up {name}: {e}")
    
    def get_application_state(self) -> ApplicationState:
        """Get current application state."""
        return self._state
    
    def get_initialization_progress(self) -> List[InitializationProgress]:
        """Get initialization progress history."""
        return self._initialization_progress.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "application_state": self._state.value,
            "initialization_complete": self._state in [ApplicationState.READY, ApplicationState.RUNNING],
            "pipeline_available": self._pipeline is not None,
            "ui_available": self._main_window is not None
        }
        
        if self._pipeline:
            status["pipeline_status"] = self._pipeline.get_pipeline_status()
        
        return status