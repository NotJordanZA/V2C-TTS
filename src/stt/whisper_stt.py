"""
Whisper-based Speech-to-Text implementation using faster-whisper.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import numpy as np
from faster_whisper import WhisperModel
import torch

from ..core.interfaces import STTInterface, PipelineConfig, PipelineError, PipelineStage


logger = logging.getLogger(__name__)


class WhisperSTT(STTInterface):
    """
    Speech-to-Text implementation using OpenAI Whisper via faster-whisper.
    
    Supports GPU acceleration and configurable model sizes for optimal
    performance on high-end consumer hardware.
    """
    
    # Available Whisper model sizes
    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.model: Optional[WhisperModel] = None
        self.model_size = config.stt_model_size
        self.device = config.gpu_device
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        # Validate model size
        if self.model_size not in self.AVAILABLE_MODELS:
            logger.warning(f"Unknown model size '{self.model_size}', defaulting to 'base'")
            self.model_size = "base"
    
    async def initialize(self) -> None:
        """Initialize the Whisper model with GPU acceleration if available."""
        try:
            logger.info(f"Initializing Whisper model: {self.model_size} on {self.device}")
            
            # Check GPU availability
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
                self.compute_type = "int8"
            
            await self.load_model()
            self._initialized = True
            logger.info("Whisper STT initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize Whisper model: {str(e)}"
            logger.error(error_msg)
            raise PipelineError(PipelineStage.SPEECH_TO_TEXT, error_msg, recoverable=False)
    
    async def load_model(self) -> None:
        """Load the Whisper model with specified configuration."""
        try:
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                self._load_model_sync
            )
            logger.info(f"Whisper model '{self.model_size}' loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load Whisper model: {str(e)}"
            logger.error(error_msg)
            raise PipelineError(PipelineStage.SPEECH_TO_TEXT, error_msg, recoverable=True)
    
    def _load_model_sync(self) -> WhisperModel:
        """Synchronous model loading helper."""
        return WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=4 if self.device == "cpu" else 0,
            num_workers=1  # Single worker for real-time processing
        )
    
    async def unload_model(self) -> None:
        """Unload the model to free GPU memory."""
        if self.model is not None:
            # faster-whisper doesn't have explicit cleanup, but we can clear the reference
            self.model = None
            
            # Force garbage collection to free GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("Whisper model unloaded")
    
    async def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Convert audio data to text using Whisper.
        
        Args:
            audio_data: Audio samples as numpy array (float32, mono, 16kHz)
            
        Returns:
            Transcribed text string
            
        Raises:
            PipelineError: If transcription fails
        """
        if not self._initialized or self.model is None:
            raise PipelineError(
                PipelineStage.SPEECH_TO_TEXT,
                "Whisper model not initialized",
                recoverable=True
            )
        
        try:
            # Ensure audio is in correct format
            audio_data = self._preprocess_audio(audio_data)
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                audio_data
            )
            
            # Combine all segments into single text
            text = " ".join(segment.text.strip() for segment in segments)
            
            logger.debug(f"Transcribed audio: '{text}' (language: {info.language}, probability: {info.language_probability:.2f})")
            
            return text.strip()
            
        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            logger.error(error_msg)
            raise PipelineError(PipelineStage.SPEECH_TO_TEXT, error_msg, recoverable=True)
    
    def _transcribe_sync(self, audio_data: np.ndarray):
        """Synchronous transcription helper."""
        return self.model.transcribe(
            audio_data,
            beam_size=1,  # Faster inference for real-time
            best_of=1,
            temperature=0.0,  # Deterministic output
            condition_on_previous_text=False,  # Better for real-time chunks
            vad_filter=True,  # Voice activity detection
            vad_parameters=dict(min_silence_duration_ms=500)
        )
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data for Whisper input.
        
        Whisper expects:
        - Float32 samples
        - Mono channel
        - 16kHz sample rate
        - Values in range [-1, 1]
        """
        # Ensure float32 type
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Ensure mono (take first channel if stereo)
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        # Normalize to [-1, 1] range if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.unload_model()
        self._initialized = False
        logger.info("Whisper STT cleanup completed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "initialized": self._initialized,
            "available_models": self.AVAILABLE_MODELS
        }
    
    def set_model_size(self, model_size: str) -> None:
        """
        Change the model size (requires reinitialization).
        
        Args:
            model_size: New model size from AVAILABLE_MODELS
        """
        if model_size not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model size. Available: {self.AVAILABLE_MODELS}")
        
        if model_size != self.model_size:
            self.model_size = model_size
            logger.info(f"Model size changed to '{model_size}'. Reinitialization required.")