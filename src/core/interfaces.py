"""
Core interfaces and abstract base classes for the voice transformation pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
import numpy as np
from enum import Enum


class PipelineStage(str, Enum):
    """Pipeline processing stages."""
    AUDIO_CAPTURE = "audio_capture"
    SPEECH_TO_TEXT = "speech_to_text"
    CHARACTER_TRANSFORM = "character_transform"
    TEXT_TO_SPEECH = "text_to_speech"
    AUDIO_OUTPUT = "audio_output"


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    duration_ms: float


@dataclass
class AudioDevice:
    """Represents an audio input/output device."""
    id: int
    name: str
    channels: int
    sample_rate: int
    is_input: bool


@dataclass
class CharacterProfile:
    """Defines a character's transformation rules and voice model."""
    name: str
    description: str
    personality_traits: List[str]
    speech_patterns: Dict[str, str]
    vocabulary_preferences: Dict[str, List[str]]
    transformation_prompt: str
    voice_model_path: str
    intensity_multiplier: float = 1.0


@dataclass
class VoiceModel:
    """Represents a voice model for TTS generation."""
    name: str
    model_path: str
    sample_rate: int
    language: str
    gender: str


@dataclass
class PipelineConfig:
    """Configuration for the voice transformation pipeline."""
    audio_device_id: int
    sample_rate: int = 16000
    chunk_size: int = 1024
    stt_model_size: str = "base"
    llm_model_path: str = ""
    tts_model_path: str = ""
    gpu_device: str = "cuda"
    max_latency_ms: int = 2000


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    
    def __init__(self, stage: PipelineStage, message: str, recoverable: bool = True):
        self.stage = stage
        self.message = message
        self.recoverable = recoverable
        super().__init__(f"{stage.value}: {message}")


class PipelineComponent(ABC):
    """Abstract base class for all pipeline components."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized


class AudioCaptureInterface(PipelineComponent):
    """Interface for audio capture components."""
    
    @abstractmethod
    def get_available_devices(self) -> List[AudioDevice]:
        """Get list of available audio input devices."""
        pass
    
    @abstractmethod
    async def start_capture(self, device_id: int, callback: Callable[[AudioChunk], None]) -> None:
        """Start capturing audio from specified device."""
        pass
    
    @abstractmethod
    async def stop_capture(self) -> None:
        """Stop audio capture."""
        pass
    
    @abstractmethod
    def set_sensitivity(self, threshold: float) -> None:
        """Set voice activity detection sensitivity."""
        pass


class STTInterface(PipelineComponent):
    """Interface for Speech-to-Text components."""
    
    @abstractmethod
    async def transcribe(self, audio_data: np.ndarray) -> str:
        """Convert audio to text."""
        pass
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the STT model."""
        pass
    
    @abstractmethod
    async def unload_model(self) -> None:
        """Unload the STT model to free resources."""
        pass


class CharacterTransformInterface(PipelineComponent):
    """Interface for character transformation components."""
    
    @abstractmethod
    async def transform_text(self, text: str, character: CharacterProfile) -> str:
        """Transform text according to character profile."""
        pass
    
    @abstractmethod
    def load_character(self, character_name: str) -> CharacterProfile:
        """Load character profile by name."""
        pass
    
    @abstractmethod
    def get_available_characters(self) -> List[str]:
        """Get list of available character names."""
        pass


class TTSInterface(PipelineComponent):
    """Interface for Text-to-Speech components."""
    
    @abstractmethod
    async def synthesize(self, text: str, voice_model: VoiceModel) -> np.ndarray:
        """Generate speech audio from text."""
        pass
    
    @abstractmethod
    def load_voice_model(self, model_path: str) -> VoiceModel:
        """Load voice model from file."""
        pass
    
    @abstractmethod
    def get_available_voices(self) -> List[VoiceModel]:
        """Get list of available voice models."""
        pass


class AudioOutputInterface(PipelineComponent):
    """Interface for audio output components."""
    
    @abstractmethod
    def get_available_devices(self) -> List[AudioDevice]:
        """Get list of available audio output devices."""
        pass
    
    @abstractmethod
    async def play_audio(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Play audio through output device."""
        pass
    
    @abstractmethod
    async def stop_playback(self) -> None:
        """Stop current audio playback."""
        pass


class PipelineOrchestrator(ABC):
    """Interface for pipeline orchestration."""
    
    @abstractmethod
    async def start_pipeline(self) -> None:
        """Start the processing pipeline."""
        pass
    
    @abstractmethod
    async def stop_pipeline(self) -> None:
        """Stop the processing pipeline."""
        pass
    
    @abstractmethod
    def set_character(self, character_name: str) -> None:
        """Change the active character profile."""
        pass
    
    @abstractmethod
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics."""
        pass