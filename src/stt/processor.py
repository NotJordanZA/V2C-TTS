"""
STT processing pipeline with async queue management and real-time transcription.
"""

import asyncio
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import time

from ..core.interfaces import PipelineConfig, PipelineError, PipelineStage, AudioChunk
from .whisper_stt import WhisperSTT


logger = logging.getLogger(__name__)


class ProcessingState(str, Enum):
    """STT processor states."""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class STTResult:
    """Result from STT processing."""
    text: str
    audio_chunk: AudioChunk
    processing_time_ms: float
    confidence: float = 1.0


class STTProcessor:
    """
    Asynchronous STT processing pipeline with queue management.
    
    Handles real-time audio transcription with preprocessing, format conversion,
    and streaming support for continuous audio input.
    """
    
    def __init__(self, config: PipelineConfig, result_callback: Optional[Callable[[STTResult], None]] = None):
        self.config = config
        self.result_callback = result_callback
        
        # Initialize Whisper STT engine
        self.whisper_stt = WhisperSTT(config)
        
        # Processing queue and state
        self.audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=10)
        self.state = ProcessingState.IDLE
        self.processing_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.total_processed = 0
        self.total_processing_time = 0.0
        self.last_processing_time = 0.0
        
        # Audio preprocessing settings
        self.min_audio_length_ms = 100  # Minimum audio length to process
        self.max_audio_length_ms = 30000  # Maximum audio length (30 seconds)
        self.silence_threshold = 0.01  # Threshold for silence detection
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the STT processor."""
        try:
            logger.info("Initializing STT processor")
            
            # Initialize Whisper STT
            await self.whisper_stt.initialize()
            
            self._initialized = True
            logger.info("STT processor initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize STT processor: {str(e)}"
            logger.error(error_msg)
            raise PipelineError(PipelineStage.SPEECH_TO_TEXT, error_msg, recoverable=False)
    
    async def start_processing(self) -> None:
        """Start the async processing loop."""
        if not self._initialized:
            raise PipelineError(
                PipelineStage.SPEECH_TO_TEXT,
                "STT processor not initialized",
                recoverable=True
            )
        
        if self.processing_task is not None and not self.processing_task.done():
            logger.warning("Processing already started")
            return
        
        logger.info("Starting STT processing loop")
        self.state = ProcessingState.IDLE
        self.processing_task = asyncio.create_task(self._processing_loop())
    
    async def stop_processing(self) -> None:
        """Stop the processing loop."""
        logger.info("Stopping STT processing")
        self.state = ProcessingState.STOPPED
        
        if self.processing_task is not None:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None
        
        # Clear any remaining items in queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        logger.info("STT processing stopped")
    
    async def process_audio(self, audio_chunk: AudioChunk) -> None:
        """
        Queue audio chunk for processing.
        
        Args:
            audio_chunk: Audio data to process
            
        Raises:
            PipelineError: If queue is full or processor is not running
        """
        if self.state == ProcessingState.STOPPED:
            raise PipelineError(
                PipelineStage.SPEECH_TO_TEXT,
                "STT processor is stopped",
                recoverable=True
            )
        
        # Preprocess and validate audio
        processed_chunk = self._preprocess_audio_chunk(audio_chunk)
        
        if processed_chunk is None:
            logger.debug("Audio chunk filtered out during preprocessing")
            return
        
        try:
            # Add to queue (non-blocking)
            self.audio_queue.put_nowait(processed_chunk)
            logger.debug(f"Audio chunk queued (queue size: {self.audio_queue.qsize()})")
            
        except asyncio.QueueFull:
            logger.warning("STT queue full, dropping audio chunk")
            # Optionally could drop oldest item and add new one
            # try:
            #     self.audio_queue.get_nowait()
            #     self.audio_queue.put_nowait(processed_chunk)
            # except asyncio.QueueEmpty:
            #     pass
    
    async def _processing_loop(self) -> None:
        """Main processing loop that handles queued audio chunks."""
        logger.info("STT processing loop started")
        
        while self.state != ProcessingState.STOPPED:
            try:
                # Wait for audio chunk with timeout
                audio_chunk = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=1.0
                )
                
                # Process the audio chunk
                await self._process_single_chunk(audio_chunk)
                
                # Mark task as done
                self.audio_queue.task_done()
                
            except asyncio.TimeoutError:
                # No audio to process, continue loop
                continue
                
            except asyncio.CancelledError:
                logger.info("Processing loop cancelled")
                break
                
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                self.state = ProcessingState.ERROR
                # Continue processing despite errors
                continue
        
        logger.info("STT processing loop ended")
    
    async def _process_single_chunk(self, audio_chunk: AudioChunk) -> None:
        """Process a single audio chunk."""
        start_time = time.time()
        self.state = ProcessingState.PROCESSING
        
        try:
            logger.debug(f"Processing audio chunk: {audio_chunk.duration_ms:.1f}ms")
            
            # Convert audio chunk to format expected by Whisper
            audio_data = self._convert_audio_format(audio_chunk)
            
            # Transcribe audio
            text = await self.whisper_stt.transcribe(audio_data)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.last_processing_time = processing_time
            self.total_processing_time += processing_time
            self.total_processed += 1
            
            # Create result
            result = STTResult(
                text=text,
                audio_chunk=audio_chunk,
                processing_time_ms=processing_time
            )
            
            logger.debug(f"Transcription completed: '{text}' ({processing_time:.1f}ms)")
            
            # Call result callback if provided
            if self.result_callback:
                try:
                    self.result_callback(result)
                except Exception as e:
                    logger.error(f"Error in result callback: {str(e)}")
            
            self.state = ProcessingState.IDLE
            
        except Exception as e:
            self.state = ProcessingState.ERROR
            error_msg = f"Failed to process audio chunk: {str(e)}"
            logger.error(error_msg)
            raise PipelineError(PipelineStage.SPEECH_TO_TEXT, error_msg, recoverable=True)
    
    def _preprocess_audio_chunk(self, audio_chunk: AudioChunk) -> Optional[AudioChunk]:
        """
        Preprocess audio chunk before transcription.
        
        Returns None if chunk should be filtered out.
        """
        # Check audio length constraints
        if audio_chunk.duration_ms < self.min_audio_length_ms:
            logger.debug(f"Audio chunk too short: {audio_chunk.duration_ms:.1f}ms")
            return None
        
        if audio_chunk.duration_ms > self.max_audio_length_ms:
            logger.warning(f"Audio chunk too long: {audio_chunk.duration_ms:.1f}ms, truncating")
            # Truncate to max length
            max_samples = int(self.max_audio_length_ms * audio_chunk.sample_rate / 1000)
            truncated_data = audio_chunk.data[:max_samples]
            audio_chunk = AudioChunk(
                data=truncated_data,
                timestamp=audio_chunk.timestamp,
                sample_rate=audio_chunk.sample_rate,
                duration_ms=self.max_audio_length_ms
            )
        
        # Check for silence (basic energy-based detection)
        if self._is_silence(audio_chunk.data):
            logger.debug("Audio chunk appears to be silence, skipping")
            return None
        
        return audio_chunk
    
    def _is_silence(self, audio_data: np.ndarray) -> bool:
        """Simple silence detection based on RMS energy."""
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms < self.silence_threshold
    
    def _convert_audio_format(self, audio_chunk: AudioChunk) -> np.ndarray:
        """
        Convert audio chunk to format expected by Whisper.
        
        Whisper expects:
        - Float32 samples
        - Mono channel
        - 16kHz sample rate
        """
        audio_data = audio_chunk.data
        
        # Resample if necessary (basic resampling - for production use librosa)
        if audio_chunk.sample_rate != 16000:
            logger.debug(f"Resampling from {audio_chunk.sample_rate}Hz to 16000Hz")
            # Simple resampling (not ideal for production)
            ratio = 16000 / audio_chunk.sample_rate
            new_length = int(len(audio_data) * ratio)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), new_length),
                np.arange(len(audio_data)),
                audio_data
            )
        
        return audio_data.astype(np.float32)
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.stop_processing()
        await self.whisper_stt.cleanup()
        self._initialized = False
        logger.info("STT processor cleanup completed")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        avg_processing_time = (
            self.total_processing_time / self.total_processed
            if self.total_processed > 0 else 0.0
        )
        
        return {
            "state": self.state.value,
            "total_processed": self.total_processed,
            "queue_size": self.audio_queue.qsize(),
            "last_processing_time_ms": self.last_processing_time,
            "avg_processing_time_ms": avg_processing_time,
            "total_processing_time_ms": self.total_processing_time,
            "whisper_model_info": self.whisper_stt.get_model_info()
        }
    
    def configure_preprocessing(
        self,
        min_audio_length_ms: Optional[int] = None,
        max_audio_length_ms: Optional[int] = None,
        silence_threshold: Optional[float] = None
    ) -> None:
        """Configure audio preprocessing parameters."""
        if min_audio_length_ms is not None:
            self.min_audio_length_ms = min_audio_length_ms
        if max_audio_length_ms is not None:
            self.max_audio_length_ms = max_audio_length_ms
        if silence_threshold is not None:
            self.silence_threshold = silence_threshold
        
        logger.info(f"Preprocessing configured: min={self.min_audio_length_ms}ms, "
                   f"max={self.max_audio_length_ms}ms, silence_threshold={self.silence_threshold}")
    
    @property
    def is_processing(self) -> bool:
        """Check if processor is currently processing audio."""
        return self.state == ProcessingState.PROCESSING
    
    @property
    def is_initialized(self) -> bool:
        """Check if processor is initialized."""
        return self._initialized