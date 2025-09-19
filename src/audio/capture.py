"""
Real-time audio capture with voice activity detection and circular buffering.
"""

import pyaudio
import numpy as np
import threading
import time
import logging
from typing import Callable, Optional, List
from collections import deque
import asyncio

from ..core.interfaces import (
    AudioCaptureInterface, AudioChunk, AudioDevice, PipelineConfig, 
    PipelineError, PipelineStage
)
from .device_manager import AudioDeviceManager

logger = logging.getLogger(__name__)


class CircularAudioBuffer:
    """Thread-safe circular buffer for audio data."""
    
    def __init__(self, max_duration_seconds: float = 10.0, sample_rate: int = 16000):
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.buffer = deque(maxlen=self.max_samples)
        self._lock = threading.Lock()
    
    def write(self, audio_data: np.ndarray) -> None:
        """Write audio data to the buffer."""
        with self._lock:
            # Convert to list and extend buffer
            self.buffer.extend(audio_data.flatten())
    
    def read(self, num_samples: int) -> np.ndarray:
        """Read the most recent audio data from buffer."""
        with self._lock:
            if len(self.buffer) < num_samples:
                # Pad with zeros at the beginning if not enough data
                padding_needed = num_samples - len(self.buffer)
                data = [0] * padding_needed + list(self.buffer)
            else:
                # Get the most recent samples
                data = list(self.buffer)[-num_samples:]
            
            return np.array(data, dtype=np.float32)
    
    def get_recent_chunk(self, duration_seconds: float) -> np.ndarray:
        """Get a chunk of recent audio data."""
        num_samples = int(duration_seconds * self.sample_rate)
        return self.read(num_samples)
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.buffer.clear()
    
    @property
    def current_size(self) -> int:
        """Get current buffer size in samples."""
        with self._lock:
            return len(self.buffer)


class VoiceActivityDetector:
    """Simple voice activity detection based on energy and zero-crossing rate."""
    
    def __init__(self, 
                 energy_threshold: float = 0.01,
                 zcr_threshold: float = 0.1,
                 min_speech_duration: float = 0.1,
                 min_silence_duration: float = 0.5):
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        
        self._speech_start_time = None
        self._silence_start_time = None
        self._is_speaking = False
    
    def _calculate_energy(self, audio_data: np.ndarray) -> float:
        """Calculate RMS energy of audio signal."""
        return np.sqrt(np.mean(audio_data ** 2))
    
    def _calculate_zcr(self, audio_data: np.ndarray) -> float:
        """Calculate zero-crossing rate."""
        if len(audio_data) <= 1:
            return 0.0
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data))))
        return zero_crossings / (2 * (len(audio_data) - 1))
    
    def detect_speech(self, audio_data: np.ndarray, timestamp: float) -> tuple[bool, bool]:
        """
        Detect speech activity in audio data.
        
        Returns:
            tuple: (is_speech_active, speech_ended)
                - is_speech_active: True if speech is currently detected
                - speech_ended: True if speech just ended (useful for triggering processing)
        """
        energy = self._calculate_energy(audio_data)
        zcr = self._calculate_zcr(audio_data)
        
        # Simple heuristic: speech has higher energy and moderate ZCR
        has_speech_characteristics = (
            energy > self.energy_threshold and 
            zcr > self.zcr_threshold * 0.5 and 
            zcr < self.zcr_threshold * 2.0
        )
        
        speech_ended = False
        
        if has_speech_characteristics:
            if not self._is_speaking:
                if self._speech_start_time is None:
                    self._speech_start_time = timestamp
                elif timestamp - self._speech_start_time >= self.min_speech_duration:
                    self._is_speaking = True
                    self._silence_start_time = None
                    logger.debug("Speech started")
            else:
                # Continue speech
                self._silence_start_time = None
        else:
            if self._is_speaking:
                if self._silence_start_time is None:
                    self._silence_start_time = timestamp
                elif timestamp - self._silence_start_time >= self.min_silence_duration:
                    self._is_speaking = False
                    self._speech_start_time = None
                    speech_ended = True
                    logger.debug("Speech ended")
            else:
                # Continue silence
                self._speech_start_time = None
        
        return self._is_speaking, speech_ended
    
    def set_sensitivity(self, threshold: float) -> None:
        """Adjust voice activity detection sensitivity (0.0 to 1.0)."""
        # Scale the energy threshold based on sensitivity
        # Higher sensitivity (closer to 1.0) = lower threshold (more sensitive)
        # Lower sensitivity (closer to 0.0) = higher threshold (less sensitive)
        base_threshold = 0.01
        # Use exponential scaling: threshold 0.5 = base, 1.0 = base/2, 0.0 = base*2
        self.energy_threshold = base_threshold * (2.0 ** (0.5 - threshold))
        logger.info(f"VAD sensitivity set to {threshold}, energy threshold: {self.energy_threshold}")


class AudioCapture(AudioCaptureInterface):
    """Real-time audio capture with voice activity detection."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._pyaudio = None
        self._stream = None
        self._device_manager = None
        self._capture_thread = None
        self._stop_event = threading.Event()
        
        # Audio processing components
        self._circular_buffer = CircularAudioBuffer(
            max_duration_seconds=10.0,
            sample_rate=config.sample_rate
        )
        self._vad = VoiceActivityDetector()
        
        # Callback for processed audio chunks
        self._audio_callback: Optional[Callable[[AudioChunk], None]] = None
        
        # Current capture settings
        self._current_device_id: Optional[int] = None
        self._is_capturing = False
    
    async def initialize(self) -> None:
        """Initialize the audio capture system."""
        try:
            self._pyaudio = pyaudio.PyAudio()
            self._device_manager = AudioDeviceManager()
            self._device_manager.__enter__()
            self._initialized = True
            logger.info("AudioCapture initialized successfully")
        except Exception as e:
            raise PipelineError(
                PipelineStage.AUDIO_CAPTURE,
                f"Failed to initialize audio capture: {str(e)}",
                recoverable=False
            )
    
    async def cleanup(self) -> None:
        """Clean up audio capture resources."""
        await self.stop_capture()
        
        if self._device_manager:
            self._device_manager.__exit__(None, None, None)
            self._device_manager = None
        
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
        
        self._initialized = False
        logger.info("AudioCapture cleaned up")
    
    def get_available_devices(self) -> List[AudioDevice]:
        """Get list of available audio input devices."""
        if not self._device_manager:
            raise PipelineError(
                PipelineStage.AUDIO_CAPTURE,
                "AudioCapture not initialized",
                recoverable=False
            )
        
        return self._device_manager.get_available_input_devices()
    
    async def start_capture(self, device_id: int, callback: Callable[[AudioChunk], None]) -> None:
        """Start capturing audio from specified device."""
        if not self._initialized:
            raise PipelineError(
                PipelineStage.AUDIO_CAPTURE,
                "AudioCapture not initialized",
                recoverable=False
            )
        
        if self._is_capturing:
            await self.stop_capture()
        
        # Validate device
        if not self._device_manager.validate_device(
            device_id, is_input=True, 
            sample_rate=self.config.sample_rate, 
            channels=1
        ):
            raise PipelineError(
                PipelineStage.AUDIO_CAPTURE,
                f"Device {device_id} does not support required audio format",
                recoverable=True
            )
        
        self._audio_callback = callback
        self._current_device_id = device_id
        self._stop_event.clear()
        
        try:
            # Open audio stream
            self._stream = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=device_id,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._audio_stream_callback
            )
            
            # Start capture thread for processing
            self._capture_thread = threading.Thread(
                target=self._capture_worker,
                daemon=True
            )
            self._capture_thread.start()
            
            self._stream.start_stream()
            self._is_capturing = True
            
            logger.info(f"Started audio capture on device {device_id}")
            
        except Exception as e:
            raise PipelineError(
                PipelineStage.AUDIO_CAPTURE,
                f"Failed to start audio capture: {str(e)}",
                recoverable=True
            )
    
    async def stop_capture(self) -> None:
        """Stop audio capture."""
        if not self._is_capturing:
            return
        
        self._stop_event.set()
        self._is_capturing = False
        
        # Stop and close stream
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.warning(f"Error stopping audio stream: {e}")
            finally:
                self._stream = None
        
        # Wait for capture thread to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                logger.warning("Capture thread did not stop gracefully")
        
        self._capture_thread = None
        self._current_device_id = None
        self._audio_callback = None
        
        # Clear buffer
        self._circular_buffer.clear()
        
        logger.info("Audio capture stopped")
    
    def set_sensitivity(self, threshold: float) -> None:
        """Set voice activity detection sensitivity (0.0 to 1.0)."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Sensitivity threshold must be between 0.0 and 1.0")
        
        self._vad.set_sensitivity(threshold)
    
    def _audio_stream_callback(self, in_data, frame_count, time_info, status):
        """PyAudio stream callback for real-time audio data."""
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to circular buffer
        self._circular_buffer.write(audio_data)
        
        return (None, pyaudio.paContinue)
    
    def _capture_worker(self) -> None:
        """Worker thread for processing audio chunks and voice activity detection."""
        chunk_duration = 0.1  # Process in 100ms chunks
        chunk_samples = int(chunk_duration * self.config.sample_rate)
        
        last_process_time = time.time()
        
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                
                # Get recent audio chunk
                audio_chunk = self._circular_buffer.get_recent_chunk(chunk_duration)
                
                if len(audio_chunk) > 0:
                    # Perform voice activity detection
                    is_speech, speech_ended = self._vad.detect_speech(audio_chunk, current_time)
                    
                    # If speech ended, process a longer chunk for better transcription
                    if speech_ended and self._audio_callback:
                        # Get a longer chunk (up to 3 seconds) for processing
                        processing_chunk = self._circular_buffer.get_recent_chunk(3.0)
                        
                        if len(processing_chunk) > 0:
                            audio_chunk_obj = AudioChunk(
                                data=processing_chunk,
                                timestamp=current_time,
                                sample_rate=self.config.sample_rate,
                                duration_ms=len(processing_chunk) / self.config.sample_rate * 1000
                            )
                            
                            # Call the callback asynchronously
                            try:
                                self._audio_callback(audio_chunk_obj)
                            except Exception as e:
                                logger.error(f"Error in audio callback: {e}")
                
                # Sleep to maintain processing rate
                sleep_time = chunk_duration - (time.time() - current_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in capture worker: {e}")
                time.sleep(0.1)  # Prevent tight error loop
        
        logger.debug("Capture worker thread stopped")
    
    @property
    def is_capturing(self) -> bool:
        """Check if currently capturing audio."""
        return self._is_capturing
    
    @property
    def current_device_id(self) -> Optional[int]:
        """Get the currently selected device ID."""
        return self._current_device_id
    
    def get_buffer_status(self) -> dict:
        """Get current buffer status for monitoring."""
        return {
            'buffer_size_samples': self._circular_buffer.current_size,
            'buffer_duration_seconds': self._circular_buffer.current_size / self.config.sample_rate,
            'is_capturing': self._is_capturing,
            'device_id': self._current_device_id
        }