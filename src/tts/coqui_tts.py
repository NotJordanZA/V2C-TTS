"""
Coqui TTS implementation for local text-to-speech generation.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import torch
from TTS.api import TTS
from TTS.utils.manage import ModelManager

from ..core.interfaces import TTSInterface, VoiceModel, PipelineConfig, PipelineError, PipelineStage


logger = logging.getLogger(__name__)


class CoquiTTS(TTSInterface):
    """
    Coqui TTS implementation with XTTS v2 support for local speech synthesis.
    Supports GPU acceleration and voice cloning capabilities.
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.tts_model: Optional[TTS] = None
        self.device = self._get_device()
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.loaded_voices: Dict[str, VoiceModel] = {}
        self.model_manager = ModelManager()
        
    def _get_device(self) -> str:
        """Determine the best device for TTS inference."""
        if self.config.gpu_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif self.config.gpu_device == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            logger.warning("GPU not available, falling back to CPU")
            return "cpu"
    
    async def initialize(self) -> None:
        """Initialize the TTS model and load default configuration."""
        try:
            logger.info(f"Initializing Coqui TTS with model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            # Initialize TTS model
            self.tts_model = TTS(
                model_name=self.model_name,
                gpu=(self.device == "cuda")
            )
            
            # Move model to appropriate device
            if hasattr(self.tts_model, 'synthesizer') and hasattr(self.tts_model.synthesizer, 'tts_model'):
                self.tts_model.synthesizer.tts_model.to(self.device)
            
            self._initialized = True
            logger.info("Coqui TTS initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Coqui TTS: {e}")
            raise PipelineError(
                PipelineStage.TEXT_TO_SPEECH,
                f"TTS initialization failed: {e}",
                recoverable=False
            )
    
    async def cleanup(self) -> None:
        """Clean up TTS resources."""
        try:
            if self.tts_model is not None:
                # Clear GPU memory
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                self.tts_model = None
                self.loaded_voices.clear()
                
            self._initialized = False
            logger.info("Coqui TTS cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during TTS cleanup: {e}")
    
    async def synthesize(self, text: str, voice_model: VoiceModel) -> np.ndarray:
        """
        Generate speech audio from text using the specified voice model.
        
        Args:
            text: Text to synthesize
            voice_model: Voice model to use for synthesis
            
        Returns:
            Audio data as numpy array
        """
        if not self._initialized or self.tts_model is None:
            raise PipelineError(
                PipelineStage.TEXT_TO_SPEECH,
                "TTS model not initialized",
                recoverable=True
            )
        
        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return np.array([], dtype=np.float32)
        
        try:
            logger.debug(f"Synthesizing text: '{text[:50]}...' with voice: {voice_model.name}")
            
            # Run synthesis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                None,
                self._synthesize_sync,
                text,
                voice_model
            )
            
            return audio_data
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise PipelineError(
                PipelineStage.TEXT_TO_SPEECH,
                f"Speech synthesis failed: {e}",
                recoverable=True
            )
    
    def _synthesize_sync(self, text: str, voice_model: VoiceModel) -> np.ndarray:
        """Synchronous synthesis method for thread pool execution."""
        try:
            # Check if this is a voice cloning model (requires speaker reference)
            if voice_model.model_path and os.path.exists(voice_model.model_path):
                # Use voice cloning with reference audio
                audio_data = self.tts_model.tts(
                    text=text,
                    speaker_wav=voice_model.model_path,
                    language=voice_model.language
                )
            else:
                # Use built-in speaker
                audio_data = self.tts_model.tts(
                    text=text,
                    language=voice_model.language
                )
            
            # Convert to numpy array and ensure correct format
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()
            
            audio_data = np.array(audio_data, dtype=np.float32)
            
            # Ensure audio is in the correct sample rate
            if hasattr(self.tts_model, 'synthesizer'):
                model_sample_rate = getattr(self.tts_model.synthesizer.output_sample_rate, 'value', 22050)
                if model_sample_rate != voice_model.sample_rate:
                    # Note: In a full implementation, we'd resample here
                    logger.warning(f"Sample rate mismatch: model={model_sample_rate}, expected={voice_model.sample_rate}")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Synchronous synthesis failed: {e}")
            raise
    
    def load_voice_model(self, model_path: str) -> VoiceModel:
        """
        Load a voice model from file path.
        
        Args:
            model_path: Path to voice model file or reference audio
            
        Returns:
            VoiceModel instance
        """
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Voice model not found: {model_path}")
            
            # Extract model information
            model_name = model_path.stem
            
            # Determine language and other properties from filename or metadata
            # This is a simplified implementation - in practice, you'd parse metadata
            language = "en"  # Default to English
            gender = "neutral"  # Default gender
            sample_rate = 22050  # Default Coqui TTS sample rate
            
            # Check if it's an audio file for voice cloning
            audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
            if model_path.suffix.lower() in audio_extensions:
                # This is a reference audio file for voice cloning
                voice_model = VoiceModel(
                    name=model_name,
                    model_path=str(model_path),
                    sample_rate=sample_rate,
                    language=language,
                    gender=gender
                )
            else:
                # This might be a model file or config
                voice_model = VoiceModel(
                    name=model_name,
                    model_path=str(model_path),
                    sample_rate=sample_rate,
                    language=language,
                    gender=gender
                )
            
            # Cache the loaded voice model
            self.loaded_voices[model_name] = voice_model
            
            logger.info(f"Loaded voice model: {model_name}")
            return voice_model
            
        except Exception as e:
            logger.error(f"Failed to load voice model from {model_path}: {e}")
            raise PipelineError(
                PipelineStage.TEXT_TO_SPEECH,
                f"Voice model loading failed: {e}",
                recoverable=True
            )
    
    def get_available_voices(self) -> List[VoiceModel]:
        """
        Get list of available voice models.
        
        Returns:
            List of available VoiceModel instances
        """
        voices = []
        
        # Add loaded voice models
        voices.extend(self.loaded_voices.values())
        
        # Add default built-in voices if available
        try:
            if self.tts_model and hasattr(self.tts_model, 'speakers'):
                speakers = getattr(self.tts_model, 'speakers', [])
                for speaker in speakers:
                    if isinstance(speaker, str):
                        voice_model = VoiceModel(
                            name=f"builtin_{speaker}",
                            model_path="",  # Built-in, no file path
                            sample_rate=22050,
                            language="en",
                            gender="neutral"
                        )
                        voices.append(voice_model)
        except Exception as e:
            logger.warning(f"Could not enumerate built-in voices: {e}")
        
        return voices
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded TTS model."""
        if not self._initialized or self.tts_model is None:
            return {}
        
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "initialized": self._initialized,
            "loaded_voices": len(self.loaded_voices)
        }
        
        try:
            if hasattr(self.tts_model, 'synthesizer'):
                synthesizer = self.tts_model.synthesizer
                info.update({
                    "sample_rate": getattr(synthesizer, 'output_sample_rate', 22050),
                    "languages": getattr(synthesizer, 'languages', []),
                    "speakers": getattr(synthesizer, 'speakers', [])
                })
        except Exception as e:
            logger.warning(f"Could not get detailed model info: {e}")
        
        return info
    
    async def warmup(self, sample_text: str = "Hello, this is a test.") -> None:
        """
        Warm up the TTS model with a sample synthesis to reduce first-call latency.
        
        Args:
            sample_text: Text to use for warmup
        """
        if not self._initialized:
            logger.warning("Cannot warmup: TTS not initialized")
            return
        
        try:
            logger.info("Warming up TTS model...")
            
            # Create a dummy voice model for warmup
            dummy_voice = VoiceModel(
                name="warmup",
                model_path="",
                sample_rate=22050,
                language="en",
                gender="neutral"
            )
            
            # Perform warmup synthesis
            await self.synthesize(sample_text, dummy_voice)
            logger.info("TTS warmup completed")
            
        except Exception as e:
            logger.warning(f"TTS warmup failed: {e}")