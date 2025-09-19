"""
TTS processing pipeline with async audio generation queue.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import soundfile as sf
from pathlib import Path

from ..core.interfaces import (
    TTSInterface, VoiceModel, PipelineConfig, PipelineError, PipelineStage,
    AudioChunk, PipelineComponent
)
from .coqui_tts import CoquiTTS
from .voice_model import VoiceModelManager


logger = logging.getLogger(__name__)


class TTSProcessingState(str, Enum):
    """TTS processing states."""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class TTSRequest:
    """Represents a TTS synthesis request."""
    request_id: str
    text: str
    voice_model: VoiceModel
    timestamp: float
    priority: int = 0  # Higher numbers = higher priority
    callback: Optional[Callable[[np.ndarray], None]] = None


@dataclass
class TTSResult:
    """Represents a TTS synthesis result."""
    request_id: str
    audio_data: np.ndarray
    sample_rate: int
    processing_time: float
    voice_model: VoiceModel
    success: bool
    error_message: Optional[str] = None


class TTSProcessor(PipelineComponent):
    """
    TTS processing pipeline with async audio generation queue.
    Handles text-to-speech synthesis with queue management and audio post-processing.
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.tts_engine: Optional[CoquiTTS] = None
        self.voice_manager: Optional[VoiceModelManager] = None
        
        # Processing queue and state
        self.request_queue: asyncio.Queue[TTSRequest] = asyncio.Queue()
        self.result_queue: asyncio.Queue[TTSResult] = asyncio.Queue()
        self.processing_state = TTSProcessingState.IDLE
        self.current_request: Optional[TTSRequest] = None
        
        # Worker task
        self.worker_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Performance metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        self.average_processing_time = 0.0
        
        # Audio processing settings
        self.target_sample_rate = config.sample_rate
        self.audio_normalization = True
        self.volume_boost = 1.0
        self.noise_gate_threshold = 0.01  # Silence threshold
        
        # Queue management
        self.max_queue_size = 10
        self.request_timeout = 30.0  # seconds
    
    async def initialize(self) -> None:
        """Initialize the TTS processor."""
        try:
            logger.info("Initializing TTS processor...")
            
            # Initialize TTS engine
            self.tts_engine = CoquiTTS(self.config)
            await self.tts_engine.initialize()
            
            # Initialize voice model manager
            self.voice_manager = VoiceModelManager()
            self.voice_manager.initialize()
            
            # Start worker task
            await self.start_processing()
            
            self._initialized = True
            logger.info("TTS processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS processor: {e}")
            raise PipelineError(
                PipelineStage.TEXT_TO_SPEECH,
                f"TTS processor initialization failed: {e}",
                recoverable=False
            )
    
    async def cleanup(self) -> None:
        """Clean up TTS processor resources."""
        try:
            logger.info("Cleaning up TTS processor...")
            
            # Stop processing
            await self.stop_processing()
            
            # Clean up TTS engine
            if self.tts_engine:
                await self.tts_engine.cleanup()
                self.tts_engine = None
            
            # Clear queues
            while not self.request_queue.empty():
                try:
                    self.request_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            self.voice_manager = None
            self._initialized = False
            
            logger.info("TTS processor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during TTS processor cleanup: {e}")
    
    async def start_processing(self) -> None:
        """Start the TTS processing worker."""
        if self.is_running:
            logger.warning("TTS processor is already running")
            return
        
        self.is_running = True
        self.worker_task = asyncio.create_task(self._processing_worker())
        logger.info("TTS processing worker started")
    
    async def stop_processing(self) -> None:
        """Stop the TTS processing worker."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        
        self.worker_task = None
        self.processing_state = TTSProcessingState.IDLE
        logger.info("TTS processing worker stopped")
    
    async def synthesize_async(
        self,
        text: str,
        voice_model: VoiceModel,
        request_id: Optional[str] = None,
        priority: int = 0,
        callback: Optional[Callable[[np.ndarray], None]] = None
    ) -> str:
        """
        Queue text for asynchronous synthesis.
        
        Args:
            text: Text to synthesize
            voice_model: Voice model to use
            request_id: Optional request ID (auto-generated if not provided)
            priority: Request priority (higher = more urgent)
            callback: Optional callback for when synthesis completes
            
        Returns:
            Request ID for tracking
        """
        if not self._initialized or not self.tts_engine:
            raise PipelineError(
                PipelineStage.TEXT_TO_SPEECH,
                "TTS processor not initialized",
                recoverable=True
            )
        
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"tts_{int(time.time() * 1000)}_{self.total_requests}"
        
        # Check queue size
        if self.request_queue.qsize() >= self.max_queue_size:
            raise PipelineError(
                PipelineStage.TEXT_TO_SPEECH,
                "TTS request queue is full",
                recoverable=True
            )
        
        # Create request
        request = TTSRequest(
            request_id=request_id,
            text=text,
            voice_model=voice_model,
            timestamp=time.time(),
            priority=priority,
            callback=callback
        )
        
        # Queue request
        await self.request_queue.put(request)
        self.total_requests += 1
        
        logger.debug(f"Queued TTS request: {request_id}")
        return request_id
    
    async def get_result(self, timeout: Optional[float] = None) -> Optional[TTSResult]:
        """
        Get the next synthesis result from the queue.
        
        Args:
            timeout: Maximum time to wait for result
            
        Returns:
            TTSResult or None if timeout
        """
        try:
            if timeout is None:
                return await self.result_queue.get()
            else:
                return await asyncio.wait_for(self.result_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    async def synthesize_sync(self, text: str, voice_model: VoiceModel) -> np.ndarray:
        """
        Synchronous synthesis (blocks until complete).
        
        Args:
            text: Text to synthesize
            voice_model: Voice model to use
            
        Returns:
            Audio data as numpy array
        """
        if not self._initialized or not self.tts_engine:
            raise PipelineError(
                PipelineStage.TEXT_TO_SPEECH,
                "TTS processor not initialized",
                recoverable=True
            )
        
        # Perform synthesis directly
        start_time = time.time()
        try:
            audio_data = await self.tts_engine.synthesize(text, voice_model)
            
            # Post-process audio
            processed_audio = self._post_process_audio(audio_data, voice_model.sample_rate)
            
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, True)
            
            return processed_audio
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, False)
            raise
    
    async def _processing_worker(self) -> None:
        """Main processing worker loop."""
        logger.info("TTS processing worker started")
        
        while self.is_running:
            try:
                # Get next request with timeout
                try:
                    request = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process request
                await self._process_request(request)
                
            except asyncio.CancelledError:
                logger.info("TTS processing worker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in TTS processing worker: {e}")
                await asyncio.sleep(0.1)  # Brief pause before retrying
        
        logger.info("TTS processing worker stopped")
    
    async def _process_request(self, request: TTSRequest) -> None:
        """Process a single TTS request."""
        self.current_request = request
        self.processing_state = TTSProcessingState.PROCESSING
        
        start_time = time.time()
        success = False
        error_message = None
        audio_data = np.array([], dtype=np.float32)
        
        try:
            logger.debug(f"Processing TTS request: {request.request_id}")
            
            # Check request timeout
            if time.time() - request.timestamp > self.request_timeout:
                raise TimeoutError(f"Request {request.request_id} timed out")
            
            # Synthesize audio
            audio_data = await self.tts_engine.synthesize(request.text, request.voice_model)
            
            # Post-process audio
            audio_data = self._post_process_audio(audio_data, request.voice_model.sample_rate)
            
            success = True
            self.processing_state = TTSProcessingState.COMPLETED
            
        except Exception as e:
            error_message = str(e)
            success = False
            self.processing_state = TTSProcessingState.ERROR
            logger.error(f"Failed to process TTS request {request.request_id}: {e}")
        
        finally:
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, success)
            
            # Create result
            result = TTSResult(
                request_id=request.request_id,
                audio_data=audio_data,
                sample_rate=self.target_sample_rate,
                processing_time=processing_time,
                voice_model=request.voice_model,
                success=success,
                error_message=error_message
            )
            
            # Queue result
            try:
                await self.result_queue.put(result)
            except Exception as e:
                logger.error(f"Failed to queue TTS result: {e}")
            
            # Call callback if provided
            if request.callback and success:
                try:
                    request.callback(audio_data)
                except Exception as e:
                    logger.error(f"Error in TTS callback: {e}")
            
            self.current_request = None
            self.processing_state = TTSProcessingState.IDLE
    
    def _post_process_audio(self, audio_data: np.ndarray, original_sample_rate: int) -> np.ndarray:
        """
        Post-process synthesized audio.
        
        Args:
            audio_data: Raw audio data
            original_sample_rate: Original sample rate
            
        Returns:
            Processed audio data
        """
        if len(audio_data) == 0:
            return audio_data
        
        try:
            # Ensure audio is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Resample if needed
            if original_sample_rate != self.target_sample_rate:
                audio_data = self._resample_audio(audio_data, original_sample_rate, self.target_sample_rate)
            
            # Apply noise gate (remove very quiet sections)
            if self.noise_gate_threshold > 0:
                audio_data = self._apply_noise_gate(audio_data, self.noise_gate_threshold)
            
            # Normalize audio
            if self.audio_normalization:
                audio_data = self._normalize_audio(audio_data)
            
            # Apply volume boost
            if self.volume_boost != 1.0:
                audio_data = audio_data * self.volume_boost
                # Clip to prevent distortion
                audio_data = np.clip(audio_data, -1.0, 1.0)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error in audio post-processing: {e}")
            return audio_data  # Return original on error
    
    def _resample_audio(self, audio_data: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """
        Resample audio data.
        
        Args:
            audio_data: Input audio data
            from_rate: Source sample rate
            to_rate: Target sample rate
            
        Returns:
            Resampled audio data
        """
        if from_rate == to_rate:
            return audio_data
        
        try:
            import librosa
            return librosa.resample(audio_data, orig_sr=from_rate, target_sr=to_rate)
        except ImportError:
            logger.warning("librosa not available for resampling, returning original audio")
            return audio_data
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return audio_data
    
    def _apply_noise_gate(self, audio_data: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply noise gate to remove quiet sections.
        
        Args:
            audio_data: Input audio data
            threshold: Amplitude threshold below which audio is silenced
            
        Returns:
            Gated audio data
        """
        try:
            # Calculate RMS in small windows
            window_size = 1024
            gated_audio = audio_data.copy()
            
            for i in range(0, len(audio_data) - window_size, window_size // 2):
                window = audio_data[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                
                if rms < threshold:
                    gated_audio[i:i + window_size] *= 0.1  # Reduce but don't completely silence
            
            return gated_audio
            
        except Exception as e:
            logger.error(f"Error applying noise gate: {e}")
            return audio_data
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio to prevent clipping while maintaining dynamics.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Normalized audio data
        """
        try:
            if len(audio_data) == 0:
                return audio_data
            
            # Find peak amplitude
            peak = np.max(np.abs(audio_data))
            
            if peak > 0:
                # Normalize to 90% of maximum to leave headroom
                target_peak = 0.9
                normalization_factor = target_peak / peak
                
                # Only normalize if the audio is too loud
                if peak > target_peak:
                    audio_data = audio_data * normalization_factor
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return audio_data
    
    def _update_metrics(self, processing_time: float, success: bool) -> None:
        """Update performance metrics."""
        self.total_processing_time += processing_time
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        total_completed = self.successful_requests + self.failed_requests
        if total_completed > 0:
            self.average_processing_time = self.total_processing_time / total_completed
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current processor status and metrics.
        
        Returns:
            Status dictionary
        """
        return {
            "initialized": self._initialized,
            "running": self.is_running,
            "processing_state": self.processing_state.value,
            "current_request": self.current_request.request_id if self.current_request else None,
            "queue_size": self.request_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
            "metrics": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": (
                    self.successful_requests / max(1, self.successful_requests + self.failed_requests)
                ),
                "average_processing_time": self.average_processing_time,
                "total_processing_time": self.total_processing_time
            },
            "settings": {
                "target_sample_rate": self.target_sample_rate,
                "audio_normalization": self.audio_normalization,
                "volume_boost": self.volume_boost,
                "noise_gate_threshold": self.noise_gate_threshold,
                "max_queue_size": self.max_queue_size,
                "request_timeout": self.request_timeout
            }
        }
    
    def clear_queue(self) -> int:
        """
        Clear all pending requests from the queue.
        
        Returns:
            Number of requests cleared
        """
        cleared_count = 0
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break
        
        logger.info(f"Cleared {cleared_count} requests from TTS queue")
        return cleared_count
    
    def set_audio_settings(
        self,
        normalization: Optional[bool] = None,
        volume_boost: Optional[float] = None,
        noise_gate_threshold: Optional[float] = None
    ) -> None:
        """
        Update audio processing settings.
        
        Args:
            normalization: Enable/disable audio normalization
            volume_boost: Volume multiplier (1.0 = no change)
            noise_gate_threshold: Noise gate threshold (0.0 = disabled)
        """
        if normalization is not None:
            self.audio_normalization = normalization
        
        if volume_boost is not None:
            self.volume_boost = max(0.1, min(3.0, volume_boost))  # Clamp to reasonable range
        
        if noise_gate_threshold is not None:
            self.noise_gate_threshold = max(0.0, min(0.5, noise_gate_threshold))
        
        logger.info(f"Updated TTS audio settings: normalization={self.audio_normalization}, "
                   f"volume_boost={self.volume_boost}, noise_gate={self.noise_gate_threshold}")
    
    async def save_audio_to_file(self, audio_data: np.ndarray, file_path: str) -> None:
        """
        Save audio data to file.
        
        Args:
            audio_data: Audio data to save
            file_path: Output file path
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(str(file_path), audio_data, self.target_sample_rate)
            logger.info(f"Saved audio to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save audio to {file_path}: {e}")
            raise PipelineError(
                PipelineStage.TEXT_TO_SPEECH,
                f"Failed to save audio: {e}",
                recoverable=True
            )